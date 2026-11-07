#include "overlay.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using V3 = cv::Vec3d;
using M3 = cv::Matx33d;
using M4 = cv::Matx44d;
using P = cv::Point2d;
using Line = std::array<P, 61>;
constexpr int HEADER = 50;
const cv::Scalar BG(16, 24, 32), WHITE(229, 237, 244), YELLOW(255, 233, 86),
    BLUE(61, 157, 255), ORANGE(255, 162, 58), BLACK(17, 17, 17),
    NOTE(211, 220, 231);
const std::array<cv::Scalar, 3> XYZ = {cv::Scalar(255, 85, 79),
                                       cv::Scalar(86, 239, 119),
                                       cv::Scalar(135, 150, 255)};
thread_local std::string last_error;
void require(bool yes, const std::string &message) {
  if (!yes)
    throw std::invalid_argument(message);
}
template <int N> cv::Vec<double, N> vec(const cv::FileNode &n) {
  require(n.isSeq() && n.size() == N, "Invalid vector in configuration");
  cv::Vec<double, N> a;
  for (int i = 0; i < N; ++i) {
    a[i] = (double)n[i];
    require(std::isfinite(a[i]), "Nonfinite configuration");
  }
  return a;
}
template <int N> cv::Matx<double, N, N> matrix(const cv::FileNode &n) {
  require(n.isSeq() && n.size() == N, "Invalid matrix in configuration");
  cv::Matx<double, N, N> a;
  for (int r = 0; r < N; ++r) {
    auto v = vec<N>(n[r]);
    for (int c = 0; c < N; ++c)
      a(r, c) = v[c];
  }
  return a;
}
M3 rot(const M4 &m) { return m.get_minor<3, 3>(0, 0); }
V3 column(const M3 &m, int i) { return {m(0, i), m(1, i), m(2, i)}; }
V3 transform(const M4 &m, const V3 &v) {
  auto p = m * cv::Vec4d(v[0], v[1], v[2], 1);
  return {p[0], p[1], p[2]};
}
M4 quaternion(const double *q, const double *t) {
  double s = 0;
  for (int i = 0; i < 4; ++i)
    s += q[i] * q[i];
  require(s >= 1e-24, "Zero quaternion");
  s = 1 / std::sqrt(s);
  double x = q[0] * s, y = q[1] * s, z = q[2] * s, w = q[3] * s;
  return {1 - 2 * (y * y + z * z),
          2 * (x * y - z * w),
          2 * (x * z + y * w),
          t[0],
          2 * (x * y + z * w),
          1 - 2 * (x * x + z * z),
          2 * (y * z - x * w),
          t[1],
          2 * (x * z - y * w),
          2 * (y * z + x * w),
          1 - 2 * (x * x + y * y),
          t[2],
          0,
          0,
          0,
          1};
}
struct Joint {
  std::string name, parent, child;
  int axis, input;
  double sign, bind, lo, hi;
  M4 a, tail;
};
struct Jaw {
  std::string body;
  V3 h, j;
};
struct Camera {
  M3 k;
  cv::Vec<double, 5> d;
  P project(const V3 &p) const {
    require(p[2] > 0 && std::isfinite(p[0]) && std::isfinite(p[1]) &&
                std::isfinite(p[2]),
            "Point behind camera or nonfinite");
    double x = p[0] / p[2], y = p[1] / p[2], r = x * x + y * y,
           rad = 1 + d[0] * r + d[1] * r * r + d[4] * r * r * r;
    P uv(k(0, 0) * (x * rad + 2 * d[2] * x * y + d[3] * (r + 2 * x * x)) +
             k(0, 2),
         k(1, 1) * (y * rad + d[2] * (r + 2 * y * y) + 2 * d[3] * x * y) +
             k(1, 2));
    require(std::isfinite(uv.x) && std::isfinite(uv.y) &&
                std::max(std::abs(uv.x), std::abs(uv.y)) <= 1e6,
            "Projection outside drawing range");
    return uv;
  }
  Line curve(const V3 &a, const V3 &b) const {
    Line l;
    for (int i = 0; i < 61; ++i)
      l[i] = project(a + (b - a) * (i / 60.));
    return l;
  }
};
double length(const Line &l) {
  double n = 0;
  for (int i = 1; i < 61; ++i)
    n += cv::norm(l[i] - l[i - 1]);
  return n;
}
struct Axis {
  V3 start, end, direction;
  Line line;
  double pixels = 0;
};
struct Frame {
  V3 origin;
  M3 rotation;
  std::array<Axis, 3> axes;
  double reference = 0, size = 0;
};
struct Eye {
  V3 c, d, t;
  std::array<V3, 2> h, j;
  std::array<M3, 2> rj;
  M3 rs;
  std::array<Line, 3> main;
  std::array<Frame, 3> frames;
  std::array<int, 2> vis{{-1, -1}};
};

/* Glyph masks are retained independently of changing frame numbers. The same
 * mask can be drawn in any color. Static labels are whole-string tiles. */
class Text {
  cv::Ptr<cv::freetype::FreeType2> ft;
  struct Tile {
    cv::Mat mask;
    int advance, height;
  };
  std::map<std::pair<int, std::string>, Tile> tiles;

public:
  explicit Text(const std::string &path) {
    ft = cv::freetype::createFreeType2();
    ft->loadFontData(path, 0);
  }
  const Tile &tile(const std::string &s, int height) {
    auto key = std::make_pair(height, s);
    auto found = tiles.find(key);
    if (found != tiles.end())
      return found->second;
    require(tiles.size() < 512, "Text tile cache limit reached");
    int base = 0;
    auto sz = ft->getTextSize(s, height, -1, &base);
    cv::Mat rgb(std::max(1, sz.height + base + 8), std::max(1, sz.width + 8),
                CV_8UC3, cv::Scalar::all(0));
    ft->putText(rgb, s, {3, sz.height + 3}, height, cv::Scalar::all(255), -1,
                cv::LINE_AA, true);
    cv::Mat mask;
    cv::extractChannel(rgb, mask, 0);
    return tiles.emplace(key, Tile{mask, sz.width, sz.height}).first->second;
  }
  int width(const std::string &s, int h) { return tile(s, h).advance; }
  void draw(cv::Mat &dst, const std::string &s, P at, int height,
            cv::Scalar color) {
    const auto &t = tile(s, height);
    int x = (int)at.x - 3, y = (int)at.y - 3;
    cv::Rect r(x, y, t.mask.cols, t.mask.rows),
        clip = r & cv::Rect(0, 0, dst.cols, dst.rows);
    for (int j = clip.y; j < clip.y + clip.height; ++j) {
      auto *out = dst.ptr<cv::Vec3b>(j);
      const auto *mask = t.mask.ptr<uint8_t>(j - y);
      for (int i = clip.x; i < clip.x + clip.width; ++i) {
        unsigned a = mask[i - x];
        if (a)
          for (int c = 0; c < 3; ++c)
            out[i][c] =
                (out[i][c] * (255 - a) + (unsigned)color[c] * a + 127) / 255;
      }
    }
  }
  void ascii(cv::Mat &dst, const std::string &s, P at, int height,
             cv::Scalar color) {
    const int baseline = tile("国Ag", height).height;
    for (unsigned char c : s) {
      std::string glyph(1, (c >= 32 && c < 127) ? c : '?');
      draw(dst, glyph, {at.x, at.y + baseline - tile(glyph, height).height},
           height, color);
      at.x += width(glyph, height);
    }
  }
};
cv::Point ip(P p) { return {cvRound(p.x * 16), cvRound(p.y * 16)}; }
void segment(cv::Mat &im, P a, P b, cv::Scalar color, int width) {
  cv::line(im, ip(a), ip(b), color, width, cv::LINE_AA, 4);
}
void arrow(cv::Mat &im, const Line &raw, P offset, cv::Scalar color, int width,
           bool broken = false, double head = 19) {
  std::array<cv::Point, 61> pts;
  for (int i = 0; i < 61; ++i)
    pts[i] = ip(raw[i] + offset);
  if (broken) {
    double dist = 0;
    for (int i = 1; i < 61; ++i) {
      if ((int)(dist / 9) % 2 == 0) {
        cv::line(im, pts[i - 1], pts[i], BLACK, width + 3, cv::LINE_AA, 4);
        cv::line(im, pts[i - 1], pts[i], color, width, cv::LINE_AA, 4);
      }
      dist += cv::norm(raw[i] - raw[i - 1]);
    }
  } else {
    const cv::Point *p = pts.data();
    int n = 61;
    cv::polylines(im, &p, &n, 1, false, BLACK, width + 3, cv::LINE_AA, 4);
    cv::polylines(im, &p, &n, 1, false, color, width, cv::LINE_AA, 4);
  }
  P tip = raw[60] + offset, v = raw[60] - raw[56];
  double norm = cv::norm(v);
  if (norm <= 1e-8)
    return;
  v *= 1 / norm;
  double s = std::min(head, length(raw) * .35);
  std::array<cv::Point, 3> tri = {ip(tip),
                                  ip(tip - v * s + P(v.y, -v.x) * (.45 * s)),
                                  ip(tip - v * s - P(v.y, -v.x) * (.45 * s))};
  cv::fillConvexPoly(im, tri.data(), 3, color, cv::LINE_AA, 4);
}
double overlap(cv::Rect2d a, cv::Rect2d b) { return (a & b).area(); }
void rounded(cv::Mat &dst, cv::Rect r, int radius, cv::Scalar fill,
             cv::Scalar border, int width, double alpha = 1) {
  // Only the small panel is blended. No full-frame temporary images.
  cv::Rect clip = r & cv::Rect(0, 0, dst.cols, dst.rows);
  if (clip.empty())
    return;
  cv::Mat mask(r.height, r.width, CV_8U, cv::Scalar(0));
  cv::rectangle(mask, {radius, 0, r.width - 2 * radius, r.height}, 255, -1);
  cv::rectangle(mask, {0, radius, r.width, r.height - 2 * radius}, 255, -1);
  for (int x : {radius, r.width - radius - 1})
    for (int y : {radius, r.height - radius - 1})
      cv::circle(mask, {x, y}, radius, 255, -1, cv::LINE_AA);
  for (int y = clip.y; y < clip.y + clip.height; ++y) {
    auto *out = dst.ptr<cv::Vec3b>(y);
    auto *m = mask.ptr<uint8_t>(y - r.y);
    for (int x = clip.x; x < clip.x + clip.width; ++x) {
      double a = m[x - r.x] * alpha / 255.;
      if (a)
        for (int c = 0; c < 3; ++c)
          out[x][c] =
              cv::saturate_cast<uint8_t>(out[x][c] * (1 - a) + fill[c] * a);
    }
  }
  cv::line(dst, {r.x + radius, r.y}, {r.x + r.width - radius - 1, r.y}, border,
           width, cv::LINE_AA);
  cv::line(dst, {r.x + radius, r.y + r.height - 1},
           {r.x + r.width - radius - 1, r.y + r.height - 1}, border, width,
           cv::LINE_AA);
  cv::line(dst, {r.x, r.y + radius}, {r.x, r.y + r.height - radius - 1}, border,
           width, cv::LINE_AA);
  cv::line(dst, {r.x + r.width - 1, r.y + radius},
           {r.x + r.width - 1, r.y + r.height - radius - 1}, border, width,
           cv::LINE_AA);
  for (int j = 0; j < 4; ++j) {
    int x = (j == 0 || j == 3) ? r.x + radius : r.x + r.width - radius - 1;
    int y = j < 2 ? r.y + radius : r.y + r.height - radius - 1;
    cv::ellipse(dst, {x, y}, {radius, radius}, 0, 180 + 90 * j, 270 + 90 * j,
                border, width, cv::LINE_AA);
  }
}
struct Overlay {
  int w, h;
  std::string shaft;
  std::vector<Joint> joints;
  std::array<Jaw, 2> jaws;
  std::array<Camera, 2> camera;
  std::array<Eye, 2> eyes;
  Text text;
  bool valid = false;
  std::string json;
  std::array<double, 17> values;
  Overlay(const char *cfg, const char *font) : text(font) {
    cv::FileStorage fs(cfg, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                                cv::FileStorage::FORMAT_JSON);
    require(fs.isOpened(), "Cannot parse configuration");
    auto m = fs["model"], cal = fs["calibration"];
    require((std::string)cal["image"]["space"] == "raw_distorted",
            "Expected raw_distorted camera profile");
    w = (int)cal["image"]["width"];
    h = (int)cal["image"]["height"];
    require(w >= 1280 && h >= 720 && w <= 8192 && h <= 8192,
            "Unsupported image dimensions");
    shaft = (std::string)m["shaft_body"];
    std::map<std::string, std::pair<int, double>> inputs = {
        {"psm_tool_pitch_joint", {14, 1}},
        {"psm_tool_yaw_joint", {15, 1}},
        {"psm_tool_gripper1_joint", {16, -1}},
        {"psm_tool_gripper2_joint", {16, 1}}};
    std::map<std::string, bool> known{{shaft, true}};
    for (auto n : m["joints"]) {
      Joint j;
      j.name = (std::string)n["name"];
      j.parent = (std::string)n["parent"];
      j.child = (std::string)n["child"];
      require(inputs.count(j.name) && known.count(j.parent) &&
                  !known.count(j.child),
              "Unsupported joint chain");
      j.input = inputs.at(j.name).first;
      j.sign = inputs.at(j.name).second;
      std::string axis = (std::string)n["axis"];
      require(axis.size() == 1 &&
                  std::string("XYZ").find(axis) != std::string::npos,
              "Invalid joint axis");
      j.axis = (int)std::string("XYZ").find(axis);
      j.bind = (double)n["bind_angle_rad"];
      j.lo = (double)n["lower_rad"];
      j.hi = (double)n["upper_rad"];
      require(std::isfinite(j.bind) && std::isfinite(j.lo) &&
                  std::isfinite(j.hi) && j.lo <= j.hi,
              "Invalid joint limits");
      j.a = matrix<4>(n["A_parent_joint_mm"]);
      bool ok = false;
      M4 inv = j.a.inv(cv::DECOMP_LU, &ok);
      require(ok, "Singular joint anchor");
      j.tail = inv * matrix<4>(n["B_parent_child_bind_mm"]);
      known[j.child] = true;
      joints.push_back(j);
    }
    require(joints.size() == 4, "Expected four articulated joints");
    for (int i = 0; i < 2; ++i) {
      auto n = m["jaws"]["jaw" + std::to_string(i + 1)];
      jaws[i] = {(std::string)n["body"], vec<3>(n["hinge_local_mm"]),
                 vec<3>(n["tip_local_mm"])};
      require(known.count(jaws[i].body), "Missing jaw body");
      auto c = cal[i == 0 ? "left" : "right"];
      camera[i] = {matrix<3>(c["K"]), vec<5>(c["D"])};
      auto k = camera[i].k;
      require(k(0, 0) > 0 && k(1, 1) > 0 && std::abs(k(0, 1)) <= 1e-12 &&
                  std::abs(k(1, 0)) <= 1e-12 &&
                  std::abs(k(2, 0)) + std::abs(k(2, 1)) +
                          std::abs(k(2, 2) - 1) <=
                      1e-12,
              "Unsupported camera matrix");
    }
  }
  void compute(const double *v) {
    valid = false;
    require(v, "Missing prediction");
    for (int i = 0; i < 17; ++i) {
      require(std::isfinite(v[i]), "Nonfinite pose");
      values[i] = v[i];
    }
    for (auto &j : joints)
      require(v[j.input] * j.sign >= j.lo - 1e-8 &&
                  v[j.input] * j.sign <= j.hi + 1e-8,
              "Joint limit exceeded: " + j.name);
    for (int e = 0; e < 2; ++e) {
      auto &g = eyes[e];
      const auto &cam = camera[e];
      M4 root = quaternion(v + e * 7 + 3, v + e * 7);
      std::map<std::string, M4> poses{{shaft, root}};
      for (auto &j : joints) {
        double a = v[j.input] * j.sign - j.bind, c = std::cos(a),
               s = std::sin(a);
        M4 motion = M4::eye();
        int b = (j.axis + 1) % 3, d = (j.axis + 2) % 3;
        motion(b, b) = motion(d, d) = c;
        motion(b, d) = -s;
        motion(d, b) = s;
        poses[j.child] = poses.at(j.parent) * j.a * motion * j.tail;
      }
      g.c = transform(root, {0, 0, 0});
      g.rs = rot(root);
      g.d = column(g.rs, 2);
      g.main[0] = cam.curve(g.c, g.c + 15 * g.d);
      for (int i = 0; i < 2; ++i) {
        auto m = poses.at(jaws[i].body);
        g.h[i] = transform(m, jaws[i].h);
        g.j[i] = transform(m, jaws[i].j);
        g.rj[i] = rot(m);
        g.main[i + 1] = cam.curve(g.h[i], g.j[i]);
        g.vis[i] = -1;
      }
      g.t = (g.j[0] + g.j[1]) * .5;
      g.frames[0].origin = g.c;
      g.frames[0].rotation = g.rs;
      g.frames[0].reference = std::max(
          12., std::min({length(g.main[0]), length(cam.curve(g.c, g.j[0])),
                         length(cam.curve(g.c, g.j[1]))}));
      for (int i = 1; i < 3; ++i) {
        g.frames[i].origin = g.h[i - 1];
        g.frames[i].rotation = g.rj[i - 1];
        g.frames[i].reference = std::max(12., length(g.main[i]));
      }
    }
    for (int group = 0; group < 2; ++group) {
      double size = 1e9;
      for (auto &g : eyes)
        size =
            std::min(size, group == 0 ? std::min({15., cv::norm(g.j[0] - g.c),
                                                  cv::norm(g.j[1] - g.c)})
                                      : std::min(cv::norm(g.j[0] - g.h[0]),
                                                 cv::norm(g.j[1] - g.h[1])));
      size *= .5;
      require(size > 1e-9, "Degenerate arrow geometry");
      bool fits = false;
      for (int trial = 0; trial < 100; ++trial) {
        fits = true;
        for (int e = 0; e < 2; ++e)
          for (int f = group == 0 ? 0 : 1; f < (group == 0 ? 1 : 3); ++f) {
            auto &fr = eyes[e].frames[f];
            fr.size = size;
            for (int a = 0; a < 3; ++a) {
              auto &ax = fr.axes[a];
              ax.start = fr.origin;
              ax.direction = column(fr.rotation, a);
              ax.end = ax.start + size * ax.direction;
              ax.line = camera[e].curve(ax.start, ax.end);
              ax.pixels = length(ax.line);
              fits &= ax.pixels <= .55 * fr.reference;
            }
          }
        if (fits)
          break;
        size *= .9;
      }
      require(fits, "Unable to size XYZ panels");
    }
    valid = true;
  }
  void panel(cv::Mat im, int e, const std::string &fid, int64_t timestamp,
             bool replay) {
    auto &g = eyes[e];
    const auto &cam = camera[e];
    P base(0, HEADER);
    arrow(im, g.main[2], base, ORANGE, 6, g.vis[1] != 1, 21);
    arrow(im, g.main[1], base, BLUE, 3, g.vis[0] != 1, 14);
    arrow(im, g.main[0], base, YELLOW, 3);
    std::array<cv::Rect, 3> rects = {cv::Rect(16, h - 252, 280, 240),
                                     cv::Rect(16, 12, 280, 240),
                                     cv::Rect(w - 296, 12, 280, 240)};
    std::vector<cv::Rect2d> protected_boxes;
    for (auto r : rects)
      protected_boxes.emplace_back(r.x, r.y + HEADER, r.width, r.height);
    auto label = [&](const std::string &s, P point, P offset, cv::Scalar color,
                     bool leader = true) {
      double bw = text.width(s, 25) + 20, bh = 44;
      std::array<P, 6> offs = {offset,
                               offset + P(80, -55),
                               offset + P(120, 55),
                               P(-bw - 80, -140),
                               P(70, 130),
                               P(-bw / 2, -180)};
      cv::Rect2d best;
      double score = 1e100;
      for (int i = 0; i < 6; ++i) {
        cv::Rect2d r(std::max(12., std::min(w - bw - 12, point.x + offs[i].x)),
                     std::max(HEADER + 12., std::min(HEADER + h - bh - 12,
                                                     point.y + offs[i].y)),
                     bw, bh);
        double n = i;
        for (auto b : protected_boxes)
          n += 1000 * overlap(r, b);
        if (n < score) {
          score = n;
          best = r;
        }
      }
      protected_boxes.emplace_back(best.x - 5, best.y - 5, best.width + 10,
                                   best.height + 10);
      if (leader)
        segment(im, point,
                {std::clamp(point.x, best.x, best.x + best.width),
                 std::clamp(point.y, best.y, best.y + best.height)},
                cv::Scalar(211, 217, 224), 1);
      rounded(im, best, 5, BG, color, 1);
      text.draw(im, s, {best.x + 10, best.y + 7}, 25, color);
    };
    label("C 杆体坐标原点", g.main[0][0] + base, {55, 85}, WHITE);
    label("d 杆轴方向", g.main[0][30] + base, {65, -110}, YELLOW, false);
    if (cv::norm(g.h[0] - g.h[1]) < 1e-4)
      label(replay ? "H1 / H2 真实铰接中心" : "H1 / H2 铰接中心（预测）",
            g.main[1][0] + base, {-270, 155}, WHITE);
    else
      for (int i = 0; i < 2; ++i)
        label("H" + std::to_string(i + 1) + " 钳口" + std::to_string(i + 1) +
                  "铰接中心",
              g.main[i + 1][0] + base, {-270., 155. + 60 * i},
              i ? ORANGE : BLUE);
    for (int i = 0; i < 2; ++i) {
      std::string state = g.vis[i] == 0    ? "出画"
                          : g.vis[i] == -1 ? "可见性未知"
                          : g.vis[i] == 1  ? "可见"
                          : g.vis[i] == 2  ? "组织遮挡"
                          : g.vis[i] == 3  ? "自遮挡"
                                           : "遮挡";
      label("J" + std::to_string(i + 1) + " 钳口" + std::to_string(i + 1) +
                "尖端 · " + state,
            g.main[i + 1][60] + base,
            {-390, g.main[i + 1][60].y <= g.main[2 - i][60].y ? -130. : 100.},
            i ? ORANGE : BLUE);
    }
    for (int f = 0; f < 3; ++f) {
      auto r = rects[f];
      auto &fr = g.frames[f];
      auto border = f == 0 ? YELLOW : f == 1 ? BLUE : ORANGE;
      rounded(im, {r.x, r.y + HEADER, r.width, r.height}, 9,
              cv::Scalar(17, 29, 42), border, 2, 102. / 255.);
      text.draw(im,
                f == 0 ? "杆体 · XYZ 姿态"
                       : "钳口 " + std::to_string(f) + " · XYZ 姿态",
                {r.x + 14., r.y + HEADER + 10.}, 22, border);
      std::string anchor = f == 0 ? "C" : "H" + std::to_string(f);
      text.draw(im,
                "仅移位显示；" + std::string(replay ? "真实 " : "预测 ") +
                    anchor + " 见器械",
                {r.x + 12., r.y + r.height + HEADER - 29.}, 18, NOTE);
      P origin(r.x + r.width / 2., r.y + 126),
          offset = origin - cam.project(fr.origin);
      std::vector<cv::Rect2d> boxes;
      for (int a = 0; a < 3; ++a) {
        auto &ax = fr.axes[a];
        arrow(im, ax.line, offset + base, XYZ[a], 3, false, 10);
        P end = ax.line[60] + offset, v = end - origin;
        double n = cv::norm(v);
        v = n > 1e-8 ? v * (1 / n) : P(0, -1);
        std::string name =
            "+" + std::string(1, "XYZ"[a]) + (f ? std::to_string(f) : "");
        int tw = text.width(name, 22);
        P best;
        cv::Rect2d bestbox;
        double score = 1e100;
        std::array<P, 6> options = {P(20, 0),   P(32, 0),  P(22, 20),
                                    P(22, -20), P(34, 30), P(34, -30)};
        for (int i = 0; i < 6; ++i) {
          P p = end + v * options[i].x + P(-v.y, v.x) * options[i].y;
          p.x = std::clamp(p.x, r.x + 32., r.x + r.width - 32.);
          p.y = std::clamp(p.y, r.y + 51., r.y + r.height - 48.) + HEADER;
          cv::Rect2d b(p.x - tw / 2., p.y - 13, tw, 26);
          double s = i;
          for (auto q : boxes)
            s += 1000 * overlap(b, q);
          if (s < score) {
            score = s;
            best = p;
            bestbox = b;
          }
        }
        if (cv::norm(best - (end + base)) > 30)
          segment(im, end + base, best, XYZ[a], 1);
        // A small dark text surround keeps the original outlined-label
        // contrast.
        for (P delta : {P(-1, 0), P(1, 0), P(0, -1), P(0, 1)})
          text.draw(im, name, {bestbox.x + delta.x, bestbox.y + delta.y}, 22,
                    cv::Scalar(17, 29, 42));
        text.draw(im, name, {bestbox.x, bestbox.y}, 22, XYZ[a]);
        boxes.emplace_back(bestbox.x - 5, bestbox.y - 5, bestbox.width + 10,
                           bestbox.height + 10);
      }
    }
    std::string prefix =
        (e == 0 ? "左目 LEFT" : "右目 RIGHT") + std::string(" · ") +
        (replay ? "真值输入回放 · 非模型预测" : "模型预测 PREDICTION") + "   ";
    text.draw(im, prefix, {16, 10}, 25, cv::Scalar::all(255));
    std::ostringstream suffix;
    suffix << fid.substr(0, 48) << " / t=" << std::fixed << std::setprecision(4)
           << timestamp / 1e9 << "s";
    text.ascii(im, suffix.str(), {16. + text.width(prefix, 25), 10}, 25,
               cv::Scalar::all(255));
  }
  void render(const double *v, const uint8_t *l, ptrdiff_t ls, const uint8_t *r,
              ptrdiff_t rs, uint8_t *out, ptrdiff_t os, const int *visibility,
              const char *fid, int64_t stamp, bool replay) {
    valid = false;
    require(l && r && out && fid && visibility, "Null frame buffer");
    require(ls >= w * 3 && rs >= w * 3 && os >= w * 6, "Invalid image stride");
    compute(v);
    for (int e = 0; e < 2; ++e)
      for (int j = 0; j < 2; ++j) {
        int vis = visibility[e * 2 + j];
        require(vis == -1 || (vis >= 1 && vis <= 4), "Invalid visibility code");
        auto uv = camera[e].project(eyes[e].j[j]);
        eyes[e].vis[j] =
            (uv.x >= 0 && uv.x < w && uv.y >= 0 && uv.y < h) ? vis : 0;
      }
    cv::Mat canvas(h + HEADER, w * 2, CV_8UC3, out, os);
    canvas.rowRange(0, HEADER).setTo(BG);
    cv::Mat(h, w, CV_8UC3, const_cast<uint8_t *>(l), ls)
        .copyTo(canvas(cv::Rect(0, HEADER, w, h)));
    cv::Mat(h, w, CV_8UC3, const_cast<uint8_t *>(r), rs)
        .copyTo(canvas(cv::Rect(w, HEADER, w, h)));
    for (int e = 0; e < 2; ++e)
      panel(canvas(cv::Rect(e * w, 0, w, h + HEADER)), e, fid, stamp, replay);
  }
  static void vjson(std::ostream &s, const V3 &v) {
    s << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
  }
  static void pjson(std::ostream &s, P p) {
    s << '[' << p.x << ',' << p.y << ']';
  }
  static void rjson(std::ostream &s, const M3 &r) {
    s << '[';
    for (int i = 0; i < 3; ++i) {
      if (i)
        s << ',';
      vjson(s, {r(i, 0), r(i, 1), r(i, 2)});
    }
    s << ']';
  }
  static void ljson(std::ostream &s, const Line &l) {
    s << '[';
    for (int i = 0; i < 61; ++i) {
      if (i)
        s << ',';
      pjson(s, l[i]);
    }
    s << ']';
  }
  const char *inspect(bool detail) {
    require(valid, "No successful current geometry");
    std::ostringstream s;
    s << std::setprecision(17) << '{';
    for (int e = 0; e < 2; ++e) {
      if (e)
        s << ',';
      auto &g = eyes[e];
      auto &cam = camera[e];
      s << '"' << (e == 0 ? "left" : "right") << "\":{";
      std::array<V3, 7> pts = {g.c, g.d, g.h[0], g.h[1], g.j[0], g.j[1], g.t};
      const char *names[] = {"C", "d", "H1", "H2", "J1", "J2", "T"};
      for (int i = 0; i < 7; ++i) {
        if (i)
          s << ',';
        s << '"' << names[i] << "\":";
        vjson(s, pts[i]);
      }
      s << ",\"R_shaft\":";
      rjson(s, g.rs);
      for (int i = 0; i < 2; ++i) {
        s << ",\"R_jaw" << i + 1 << "\":";
        rjson(s, g.rj[i]);
      }
      s << ",\"pixels\":{";
      for (int i : {0, 2, 3, 4, 5, 6}) {
        if (i)
          s << ',';
        s << '"' << names[i] << "\":";
        pjson(s, cam.project(pts[i]));
      }
      s << "},\"visibility_codes\":[" << g.vis[0] << ',' << g.vis[1] << ']';
      if (detail) {
        s << ",\"main_curves\":[";
        for (int i = 0; i < 3; ++i) {
          if (i)
            s << ',';
          ljson(s, g.main[i]);
        }
        s << "],\"frames\":[";
        for (int f = 0; f < 3; ++f) {
          if (f)
            s << ',';
          auto &fr = g.frames[f];
          s << "{\"length_mm\":" << fr.size
            << ",\"reference_px\":" << fr.reference << ",\"axes\":[";
          for (int a = 0; a < 3; ++a) {
            if (a)
              s << ',';
            auto &ax = fr.axes[a];
            s << "{\"name\":\"" << "XYZ"[a] << "\",\"start_camera_mm\":";
            vjson(s, ax.start);
            s << ",\"end_camera_mm\":";
            vjson(s, ax.end);
            s << ",\"direction_camera_unit\":";
            vjson(s, ax.direction);
            s << ",\"projected_length_px\":" << ax.pixels
              << ",\"projected_polyline_pixels\":";
            ljson(s, ax.line);
            s << '}';
          }
          s << "]}";
        }
        s << ']';
      }
      s << '}';
    }
    s << '}';
    json = s.str();
    return json.c_str();
  }
};
} // namespace
extern "C" {
void *sd_create(const char *cfg, const char *font) {
  try {
    last_error.clear();
    require(cfg && font, "Missing configuration/font");
    return new Overlay(cfg, font);
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
void sd_destroy(void *h) { delete static_cast<Overlay *>(h); }
int sd_compute(void *h, const double *v) {
  try {
    require(h, "Invalid handle");
    static_cast<Overlay *>(h)->compute(v);
    return 0;
  } catch (const std::exception &e) {
    if (h)
      static_cast<Overlay *>(h)->valid = false;
    last_error = e.what();
    return -1;
  }
}
int sd_render(void *h, const double *v, const uint8_t *l, ptrdiff_t ls,
              const uint8_t *r, ptrdiff_t rs, uint8_t *out, ptrdiff_t os,
              const int *vis, const char *fid, int64_t stamp, int replay) {
  try {
    require(h, "Invalid handle");
    static_cast<Overlay *>(h)->render(v, l, ls, r, rs, out, os, vis, fid, stamp,
                                      replay != 0);
    return 0;
  } catch (const std::exception &e) {
    if (h)
      static_cast<Overlay *>(h)->valid = false;
    last_error = e.what();
    return -1;
  }
}
const char *sd_inspect(void *h, int detail) {
  try {
    require(h, "Invalid handle");
    return static_cast<Overlay *>(h)->inspect(detail != 0);
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
const char *sd_last_error() { return last_error.c_str(); }
int sd_abi_version() { return 1; }
}
