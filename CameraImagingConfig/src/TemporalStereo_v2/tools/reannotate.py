"""Rebuild labeled previews from saved RGB/JSON only. Never imports Isaac Sim."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {"C": "#33d5ee", "T": "#ff68dd", "J1": "#66ee55", "J2": "#ffb43e", "d": "#ffe956"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(p, K, D):
    x, y = p[0] / p[2], p[1] / p[2]
    k1, k2, p1, p2, k3 = D
    r = x*x + y*y
    radial = 1 + k1*r + k2*r*r + k3*r*r*r
    return (K[0][0]*(x*radial + 2*p1*x*y + p2*(r+2*x*x)) + K[0][2],
            K[1][1]*(y*radial + p1*(r+2*y*y) + 2*p2*x*y) + K[1][2])


def dashed(draw, a, b, fill, width=2):
    length = math.dist(a, b)
    if length < 1:
        return
    for start in range(0, math.ceil(length), 14):
        ts = (start / length, min(start + 8, length) / length)
        draw.line([tuple(a[i] + t*(b[i]-a[i]) for i in range(2)) for t in ts], fill=fill, width=width)


def overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def panel(image, record, eye, font_path):
    w, h = image.size
    header, footer = 52, 106
    canvas = Image.new("RGB", (w, h+header+footer), "#101820")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), 25)
    small = ImageFont.truetype(str(font_path), 22)
    title = "左目 LEFT" if eye == "left" else "右目 RIGHT"
    draw.text((18, 12), f"{title}   {record['sequence_id']} / {record['job_id']}   t={record['time_s']:.4f}s", font=font, fill="white")
    e = record['eyes'][eye]
    g = e['tool']
    points = {"C": g['shaft_reference_pixel'], "T": g['tip_pixel'], "J1": g['jaw_tip_pixels'][0], "J2": g['jaw_tip_pixels'][1]}
    points = {k: (v[0], v[1]+header) for k, v in points.items()}
    C, direction = g['shaft_reference_position_mm'], g['shaft_direction_unit']
    # Start exactly at C; retain a 15 mm display length along positive d.
    # T is only a named midpoint marker. The arrow tip is not a landmark.
    curve = [project([C[j]+(15*i/60)*direction[j] for j in range(3)], e['K'], e['D']) for i in range(61)]
    curve = [(x, y+header) for x, y in curve]
    draw.line(curve, fill="#111111", width=8)
    draw.line(curve, fill=COLORS['d'], width=4)
    tip, previous = curve[-1], curve[-5]
    dx, dy = tip[0]-previous[0], tip[1]-previous[1]
    norm = math.hypot(dx, dy)
    if norm > 1e-8:
        dx, dy = dx/norm, dy/norm
        wing = [(tip[0]-19*dx+sign*9*dy, tip[1]-19*dy-sign*9*dx) for sign in (-1, 1)]
        draw.polygon([tip, *wing], fill=COLORS['d'])
    occupied = []
    targets = list(points.values())

    def callout(key, text, point, prefer, hidden=False, leader=True):
        color = COLORS[key]
        bbox = draw.textbbox((0, 0), text, font=font)
        bw, bh = bbox[2]-bbox[0]+22, 44
        offsets = [prefer, (-bw-75, -95), (-bw-70, 95), (65, -95), (65, 95), (-bw/2, -160), (-bw/2, 145)]
        options = []
        for i, (ox, oy) in enumerate(offsets):
            x = max(12, min(w-bw-12, point[0]+ox))
            y = max(header+12, min(header+h-bh-12, point[1]+oy))
            rect = (x, y, x+bw, y+bh)
            collision = sum(overlap(rect, (a-10, b-10, c+10, d+10)) for a, b, c, d in occupied)
            covers_point = sum(x-15 < px < x+bw+15 and y-15 < py < y+bh+15 for px, py in targets)
            options.append((collision*100 + covers_point*1e6 + i*100 + math.dist(point, (x+bw/2, y+bh/2)), rect))
        _, rect = min(options)
        occupied.append(rect)
        a, b, c, d = rect
        end = (max(a, min(c, point[0])), max(b, min(d, point[1])))
        if not leader:
            pass
        elif hidden:
            dashed(draw, point, end, color)
        else:
            draw.line((point, end), fill="#111111", width=4)
            draw.line((point, end), fill=color, width=2)
        draw.rounded_rectangle(rect, radius=7, fill="#111b25", outline=color, width=2)
        draw.text((a+11, b+7), text, fill=color, font=font)

    for key in ('C', 'T'):
        x, y = points[key]
        draw.polygon([(x,y-7),(x+7,y),(x,y+7),(x-7,y)], fill=COLORS[key], outline="#101820")
    for i, key in enumerate(('J1', 'J2')):
        x, y = points[key]
        v = e['jaw_visibility'][i]
        color = COLORS[key]
        if v['visible']:
            draw.ellipse((x-7, y-7, x+7, y+7), fill=color, outline="#111111", width=2)
        else:
            draw.ellipse((x-9, y-9, x+9, y+9), outline=color, width=3)
            if v['occluder_kind'] == 'tissue':
                draw.line((x-6,y-6,x+6,y+6), fill=color, width=2)
                draw.line((x-6,y+6,x+6,y-6), fill=color, width=2)
        status = ('出画' if not v['in_frame'] else '可见' if v['visible'] else '组织遮挡' if v['occluder_kind']=='tissue' else '自遮挡' if v['occluder_kind']=='instrument' else '遮挡')
        callout(key, f"{key} 钳口尖端{i+1} · {status}", points[key], (-390, 100 if i == 0 else -150), not v['visible'])
    callout('T', 'T 两尖端中点', points['T'], (-320, -55))
    callout('C', 'C 杆轴参考点', points['C'], (45, 110))
    callout('d', 'd 杆轴方向', curve[0], (15, -100), leader=False)
    y = header+h+12
    draw.text((18,y), '实心圆：可见尖端    空心圆＋虚线引线：自遮挡    空心叉圆＋虚线引线：组织遮挡', font=small, fill="#e6edf4")
    draw.text((18,y+38), '黄色箭头从 C 出发，沿 +d；T 只表示两尖端中点。C / T 不赋予表面可见性。', font=small, fill="#c4ced8")
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True, help='Chinese-capable TrueType/OpenType font')
    a = p.parse_args()
    labels = sorted(a.run.glob('sequences/*/labels/*.json'))
    if not labels:
        raise ValueError('No saved labels found')
    original = {str(f.relative_to(a.run)): sha(f) for f in (a.run/'sequences').rglob('*') if f.is_file()}
    a.output.mkdir(parents=True, exist_ok=True)
    files = []
    for path in labels:
        record = json.loads(path.read_text())
        panels = []
        for eye in ('left', 'right'):
            e = record['eyes'][eye]
            source = (path.parent/e['image']).resolve()
            if sha(source) != e['image_sha256']:
                raise ValueError(f'Original image hash mismatch: {source}')
            panels.append(panel(Image.open(source).convert('RGB'), record, eye, a.font))
        w, h = panels[0].size
        canvas = Image.new('RGB', (2*w, h))
        for i, im in enumerate(panels):
            canvas.paste(im, (i*w, 0))
        target = a.output/f"{record['sequence_id']}_{record['job_id']}.jpg"
        canvas.save(target, quality=95, subsampling=0)
        files.append({'file': target.name, 'sha256': sha(target), 'source_label': str(path.relative_to(a.run))})
    after = {str(f.relative_to(a.run)): sha(f) for f in (a.run/'sequences').rglob('*') if f.is_file()}
    if original != after:
        raise RuntimeError('Saved source data changed')
    report = {'renderer_invoked': False, 'source_files_unchanged': True, 'source_file_count': len(original),
              'preview_pairs': len(files), 'arrow_mm_relative_to_C': [0, 15], 'arrow_start': 'C', 'arrow_direction': '+d',
              'T_display': 'named midpoint marker only; no direction arrow',
              'annotation_code_sha256': sha(Path(__file__)), 'font_sha256': sha(a.font), 'files': files}
    (a.output/'annotation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"Annotated {len(files)} existing pairs; all {len(original)} source files unchanged; zero renders.")


if __name__ == '__main__':
    main()
