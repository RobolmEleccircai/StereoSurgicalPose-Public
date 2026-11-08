// Standalone native replay: no Python, model inference, or simulation.
#include "overlay.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "Usage: overlay_pair config.json font.ttc prediction.json "
                 "left.png right.png output.png\n";
    return 2;
  }
  try {
    std::ifstream file(argv[1]);
    if (!file)
      throw std::runtime_error("Cannot read configuration");
    std::string config((std::istreambuf_iterator<char>(file)), {});
    cv::FileStorage cfg(config, cv::FileStorage::READ |
                                    cv::FileStorage::MEMORY |
                                    cv::FileStorage::FORMAT_JSON);
    int w = (int)cfg["calibration"]["image"]["width"],
        h = (int)cfg["calibration"]["image"]["height"];
    cv::FileStorage prediction(argv[3], cv::FileStorage::READ);
    auto node = prediction["values17"];
    if (!node.isSeq() || node.size() != 17)
      throw std::runtime_error("Expected values17 array");
    double values[17];
    for (int i = 0; i < 17; ++i)
      values[i] = (double)node[i];
    cv::Mat left = cv::imread(argv[4]), right = cv::imread(argv[5]);
    if (left.empty() || right.empty() || left.size() != cv::Size(w, h) ||
        right.size() != left.size())
      throw std::runtime_error("Image dimensions mismatch");
    cv::cvtColor(left, left, cv::COLOR_BGR2RGB);
    cv::cvtColor(right, right, cv::COLOR_BGR2RGB);
    std::unique_ptr<void, decltype(&sd_destroy)> renderer(
        sd_create(config.c_str(), argv[2]), sd_destroy);
    if (!renderer)
      throw std::runtime_error(sd_last_error());
    int header=!cfg["display_mode"].empty() && (std::string)cfg["display_mode"]=="vectors"?0:50;
    cv::Mat output(h + header, 2 * w, CV_8UC3);
    int visibility[4] = {-1, -1, -1, -1};
    if (sd_render(renderer.get(), values, left.data, left.step, right.data,
                  right.step, output.data, output.step, visibility,
                  "native_pair", 0, 0))
      throw std::runtime_error(sd_last_error());
    cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(argv[6], output))
      throw std::runtime_error("Cannot write output");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
