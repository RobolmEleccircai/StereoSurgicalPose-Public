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


def point_panel(image, record, eye, font_path):
    w, h = image.size
    header, footer = 52, 264
    canvas = Image.new("RGB", (w, h+header+footer), "#101820")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), 25)
    small = ImageFont.truetype(str(font_path), 22)
    title = "左目 LEFT" if eye == "left" else "右目 RIGHT"
    draw.text((18, 12), f"{title} · 真值投影   {record['sequence_id']} / {record['job_id']}   t={record['time_s']:.4f}s", font=font, fill="white")
    e = record['eyes'][eye]
    g = e['tool']
    positions = {'C': g['shaft_reference_position_mm'], 'T': g['tip_position_mm'],
                 'J1': g['jaw1_tip_position_mm'], 'J2': g['jaw2_tip_position_mm']}
    saved_pixels = {"C": g['shaft_reference_pixel'], "T": g['tip_pixel'], "J1": g['jaw_tip_pixels'][0], "J2": g['jaw_tip_pixels'][1]}
    # Recompute image projections from saved 3D truth, K and distortion, rather
    # than using an existing overlaid preview or moving markers for readability.
    projected = {key: project(p, e['K'], e['D']) for key, p in positions.items()}
    residual = max(abs(projected[k][i]-saved_pixels[k][i]) for k in positions for i in range(2))
    if residual >= 1e-6:
        raise ValueError('3D projection disagrees with saved labels')
    points = {k: (v[0], v[1]+header) for k, v in projected.items()}
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
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=5)
        bw, bh = bbox[2]-bbox[0]+22, max(44, bbox[3]-bbox[1]+20)
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
        draw.multiline_text((a+11, b+7), text, fill=color, font=font, spacing=5)

    for key in ('C',):
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
    # T is a projected position, never a vector arrow. Make it readable even
    # when closed-jaw endpoints nearly coincide with the midpoint.
    x, y = points['T']
    for line in [(x-20,y,x+20,y),(x,y-20,x,y+20)]:
        draw.line(line, fill='#101820', width=7)
        draw.line(line, fill=COLORS['T'], width=3)
    draw.ellipse((x-12,y-12,x+12,y+12), outline=COLORS['T'], width=3)
    draw.ellipse((x-3,y-3,x+3,y+3), fill=COLORS['T'])
    uv = projected['T']
    callout('T', f'T 两尖端中点 · 投影\n(u, v) = ({uv[0]:.2f}, {uv[1]:.2f}) px', points['T'], (-400, -65))
    callout('C', 'C 杆轴参考点', points['C'], (45, 110))
    callout('d', 'd 杆轴方向', curve[0], (15, -100), leader=False)
    y = header+h+10
    fmt = lambda v, digits=2: '(' + ', '.join(f'{x:.{digits}f}' for x in v) + ')'
    draw.text((18,y), '已有仿真真值 → 带畸变的图像投影（不是模型推理结果）', font=font, fill='white')
    draw.text((18,y+39), f"T 相机坐标 {fmt(positions['T'])} mm  →  像素 {fmt(projected['T'])} px", font=font, fill=COLORS['T'])
    draw.text((18,y+78), f"C 相机坐标 {fmt(C)} mm  →  像素 {fmt(projected['C'])} px", font=font, fill=COLORS['C'])
    draw.text((18,y+117), f"d = {fmt(direction, 4)}；黄色箭头：C → C + 15 mm × d", font=font, fill=COLORS['d'])
    draw.text((18,y+161), '实心圆：可见尖端    空心圆＋虚线：自遮挡    空心叉圆＋虚线：组织遮挡', font=small, fill='#e6edf4')
    draw.text((18,y+197), '像素坐标以每目原始 1280×720 图像左上角为原点；C / T 不赋予表面可见性。', font=small, fill='#c4ced8')
    projection = {'source': 'simulation_ground_truth', 'camera_positions_mm': positions,
                  'projected_pixels': projected, 'shaft_direction_unit': direction,
                  'arrow_end_pixel': project([C[i]+15*direction[i] for i in range(3)], e['K'], e['D']),
                  'maximum_difference_from_saved_pixels_px': residual}
    return canvas, projection


def vector_panel(image, record, eye, font_path, split_jaws=False):
    """Direction/tip arrows with per-camera XYZ; optional legacy midpoint view."""
    w,h=image.size; header,footer=50,370 if split_jaws else 100
    canvas=Image.new('RGB',(w,h+header+footer),'#101820');canvas.paste(image,(0,header))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(font_path),25);small=ImageFont.truetype(str(font_path),22)
    e=record['eyes'][eye];g=e['tool']
    C=g['shaft_reference_position_mm'];T=g['tip_position_mm'];direction=g['shaft_direction_unit']
    uvC=project(C,e['K'],e['D']);uvT=project(T,e['K'],e['D'])
    residual=max(abs(uvC[i]-g['shaft_reference_pixel'][i]) for i in range(2))
    residual=max(residual,max(abs(uvT[i]-g['tip_pixel'][i]) for i in range(2)))
    tips=[g['jaw1_tip_position_mm'],g['jaw2_tip_position_mm']]
    uv_tips=[project(p,e['K'],e['D']) for p in tips]
    residual=max(residual,max(abs(uv_tips[j][i]-g['jaw_tip_pixels'][j][i]) for j in range(2) for i in range(2)))
    if residual>=1e-6:raise ValueError('Saved C/T/J1/J2 projection mismatch')
    endpoint=[C[i]+15*direction[i] for i in range(3)]
    def curve(end):
        result=[project([C[j]+i/60*(end[j]-C[j]) for j in range(3)],e['K'],e['D']) for i in range(61)]
        return [(x,y+header) for x,y in result]
    axis,position=curve(endpoint),curve(T)
    blue,yellow,orange='#3d9dff','#ffe956','#ffa23a'
    tissue_hidden=all(v['in_frame'] and v['occluder_kind']=='tissue' for v in e['jaw_visibility'])
    def arrow(points,color,width=4,broken=False,head_size=19):
        length=sum(math.dist(a,b) for a,b in zip(points,points[1:]))
        if broken:
            distance=0.
            for a,b in zip(points,points[1:]):
                step=math.dist(a,b)
                if int(distance/9)%2==0:
                    draw.line((a,b),fill='#111111',width=width+3)
                    draw.line((a,b),fill=color,width=width)
                distance+=step
        else:
            draw.line(points,fill='#111111',width=width+3)
            draw.line(points,fill=color,width=width)
        tip,previous=points[-1],points[-5];dx,dy=tip[0]-previous[0],tip[1]-previous[1]
        norm=math.hypot(dx,dy)
        if norm>1e-8:
            dx,dy=dx/norm,dy/norm;size=min(head_size,length*.35)
            wings=[(tip[0]-size*dx+sign*size*.45*dy,tip[1]-size*dy-sign*size*.45*dx) for sign in (-1,1)]
            draw.polygon([tip,*wings],fill=color)
    jaw_curves=[curve(p) for p in tips]
    if split_jaws:
        # Different symbol widths keep nearly coincident closed-jaw arrows
        # distinguishable without changing either physical endpoint.
        arrow(jaw_curves[1],orange,width=6,broken=not e['jaw_visibility'][1]['visible'],head_size=21)
        arrow(jaw_curves[0],blue,width=3,broken=not e['jaw_visibility'][0]['visible'],head_size=14)
    else:
        arrow(position,blue,width=5,broken=tissue_hidden)
    arrow(axis,yellow,width=3)
    # Labels use thin neutral leaders, never additional vector arrows or markers.
    def label(text,point,offset,color,leader=True):
        x,y=point;box=draw.textbbox((0,0),text,font=font);bw=box[2]-box[0]+20;bh=44
        a=max(12,min(w-bw-12,x+offset[0]));b=max(header+12,min(header+h-bh-12,y+offset[1]))
        if leader:
            anchor=(max(a,min(a+bw,x)),max(b,min(b+bh,y)))
            draw.line((point,anchor),fill='#d3d9e0',width=1)
        draw.rounded_rectangle((a,b,a+bw,b+bh),radius=5,fill='#101820',outline=color,width=1)
        draw.text((a+10,b+7),text,font=font,fill=color)
    label('C 共同起点',axis[0],(55,85),'#e5edf4')
    label('d 杆轴方向',axis[30],(65,-110),yellow,leader=False)
    if split_jaws:
        for j,color in enumerate((blue,orange)):
            v=e['jaw_visibility'][j]
            state='出画' if not v['in_frame'] else '可见' if v['visible'] else '组织遮挡' if v['occluder_kind']=='tissue' else '自遮挡'
            upper=uv_tips[j][1]<=uv_tips[1-j][1]
            label(f'J{j+1} 钳口{j+1}尖端 · {state}',jaw_curves[j][-1],(-390,-130 if upper else 100),color)
    else:
        label('T 尖端位置',position[-1],(-270,-100),blue)
    title='左目 LEFT' if eye=='left' else '右目 RIGHT'
    kind='三箭头真值投影' if split_jaws else '双箭头真值投影'
    draw.text((16,10),f"{title} · {kind}   {record['sequence_id']} / {record['job_id']}",font=font,fill='white')
    if split_jaws:
        theta=record['jaw_half_angle_deg']
        y=header+h
        fmt=lambda v,digits=2: '('+', '.join(f'{x:+.{digits}f}' for x in v)+')'
        draw.text((16,y+10),'黄：杆轴方向 d    蓝：C → J1    橙：C → J2    |    实线：可见；虚线：遮挡，类型见标签',font=small,fill='white')
        draw.text((16,y+44),f'{title}光学坐标系：X 向右，Y 向下，Z 向前；位置原点为本目相机光心。',font=small,fill='#c4ced8')
        rows=[('C  杆轴参考点位置',C,'mm','#e5edf4',2),
              ('d  杆轴单位方向',direction,'无单位',yellow,4),
              ('J1 钳口1尖端位置',tips[0],'mm',blue,2),
              ('J2 钳口2尖端位置',tips[1],'mm',orange,2)]
        for i,(name,values,unit,color,digits) in enumerate(rows):
            draw.text((16,y+79+34*i),f'{name}    (X, Y, Z) = {fmt(values,digits)} {unit}',font=font,fill=color)
        # Camera-origin positions and C-origin arrow displacements are different
        # vectors. Display both explicitly; never relabel J1/J2 as J1-C/J2-C.
        for j,color in enumerate((blue,orange)):
            delta=[tips[j][i]-C[i] for i in range(3)]
            draw.text((16,y+222+31*j),f'C → J{j+1} 箭头位移  (ΔX, ΔY, ΔZ) = {fmt(delta)} mm',font=small,fill=color)
        draw.text((16,y+291),f'每片钳口半开合角 θ = {theta:.2f}°    |    对称开合角 2θ = {2*theta:.2f}°',font=font,fill='#e5edf4')
        note='闭合时两个尖端投影接近重合，未人为错开。' if theta<1e-6 else 'J1/J2 固定对应两片钳口，未用中点代替。'
        draw.text((16,y+333),note+'保存真值投影，非模型预测；原图未改动，未重新渲染。',font=small,fill='#c4ced8')
    else:
        draw.text((16,header+h+10),'黄色：C → C + 15 mm × d（杆轴方向）    蓝色：C → T（尖端位置指示，位移 T−C）',font=small,fill='white')
        note='蓝色虚线：两实际钳口尖端均被组织遮挡。' if tissue_hidden else 'T 为两实际钳口尖端的中点。'
        draw.text((16,header+h+47),note+'保存的仿真真值投影，非模型预测；原图和真值未改动，未重新渲染。',font=small,fill='#c4ced8')
    result={'source':'simulation_ground_truth','C_camera_mm':C,'T_camera_mm':T,'d_unit':direction,
            'position_indicator_vector_mm':[T[i]-C[i] for i in range(3)],
            'C_pixel':uvC,'T_pixel':uvT,'axis_arrow_end_pixel':project(endpoint,e['K'],e['D']),
            'both_physical_tips_tissue_occluded':tissue_hidden,'maximum_difference_from_saved_pixels_px':residual}
    if split_jaws:
        result.update(J1_camera_mm=tips[0],J2_camera_mm=tips[1],J1_pixel=uv_tips[0],J2_pixel=uv_tips[1],
                      jaw_position_indicator_vectors_mm=[[p[i]-C[i] for i in range(3)] for p in tips],
                      jaw_visibility=e['jaw_visibility'],jaw_half_angle_deg=record['jaw_half_angle_deg'],
                      jaw_symmetric_opening_deg=2*record['jaw_half_angle_deg'],T_displayed=False,
                      xyz_annotations={'frame':f'{eye}_optical','axes':'X right, Y down, Z forward',
                          'position_origin':'camera optical center','position_unit':'mm',
                          'direction_unit':'unit vector, dimensionless',
                          'positions_displayed':['C','J1','J2'],'direction_displayed':'d',
                          'arrow_displacements_displayed':['J1-C','J2-C'],
                          'display_decimal_places':{'position_and_displacement':2,'direction':4},
                          'rounded_for_display_only':True})
    return canvas,result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True, help='Chinese-capable TrueType/OpenType font')
    p.add_argument('--style', choices=('three-arrows','two-arrows','points'), default='three-arrows')
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
        projections = {}
        for eye in ('left', 'right'):
            e = record['eyes'][eye]
            source = (path.parent/e['image']).resolve()
            if sha(source) != e['image_sha256']:
                raise ValueError(f'Original image hash mismatch: {source}')
            rgb=Image.open(source).convert('RGB')
            if a.style=='points':preview,projections[eye]=point_panel(rgb,record,eye,a.font)
            else:preview,projections[eye]=vector_panel(rgb,record,eye,a.font,split_jaws=a.style=='three-arrows')
            panels.append(preview)
        w, h = panels[0].size
        canvas = Image.new('RGB', (2*w, h))
        for i, im in enumerate(panels):
            canvas.paste(im, (i*w, 0))
        target = a.output/f"{record['sequence_id']}_{record['job_id']}.jpg"
        canvas.save(target, quality=95, subsampling=0)
        files.append({'file': target.name, 'sha256': sha(target), 'source_label': str(path.relative_to(a.run)), 'projection_results': projections})
    after = {str(f.relative_to(a.run)): sha(f) for f in (a.run/'sequences').rglob('*') if f.is_file()}
    if original != after:
        raise RuntimeError('Saved source data changed')
    report = {'renderer_invoked': False, 'source_files_unchanged': True, 'source_file_count': len(original),
              'preview_pairs': len(files), 'arrow_mm_relative_to_C': [0, 15], 'arrow_start': 'C', 'arrow_direction': '+d',
              'style':a.style,
              'camera_xyz_annotations':a.style=='three-arrows',
              'T_display': 'not displayed; physical J1/J2 are shown separately' if a.style=='three-arrows' else 'blue position-indicator arrow C->T, vector T-C; model position remains camera-frame T' if a.style=='two-arrows' else 'midpoint projection crosshair with camera XYZ and image uv; no direction arrow',
              'projection_source': 'simulation_ground_truth_not_model_inference',
              'maximum_projection_difference_px': max(v['maximum_difference_from_saved_pixels_px'] for f in files for v in f['projection_results'].values()),
              'annotation_code_sha256': sha(Path(__file__)), 'font_sha256': sha(a.font), 'files': files}
    (a.output/'annotation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"Annotated {len(files)} existing pairs; all {len(original)} source files unchanged; zero renders.")


if __name__ == '__main__':
    main()
