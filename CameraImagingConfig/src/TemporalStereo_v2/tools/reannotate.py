"""Rebuild labeled previews from saved RGB/JSON only. Never imports Isaac Sim."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {"C": "#33d5ee", "T": "#ff68dd", "J1": "#66ee55", "J2": "#ffb43e", "d": "#ffe956"}
XYZ_COLORS = ('#ff554f', '#56ef77', '#8796ff')
XYZ_MAX_PROJECTED_RATIO = 0.55


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


def instrument_axis_geometry(record, pose_frame):
    """Project the saved roll-link XYZ frame at C, including axial rotation.

    Use full rigid-body truth, not camera basis vectors or an invented roll
    inferred from the shaft direction. One display length is used in both eyes.
    """
    import numpy as np
    body=pose_frame['roll_body']
    if not body.endswith('/psm_tool_roll_link'):
        raise ValueError('Expected the saved PSM roll-link frame at C')
    world_from_body=np.asarray(pose_frame['T_world_from_body_mm'][body],dtype=float)
    body_poses={}
    for eye,e in record['eyes'].items():
        world_from_camera=np.asarray(pose_frame['camera_truth']['T_world_from_optical_mm'][eye],dtype=float)
        if world_from_body.shape!=(4,4) or world_from_camera.shape!=(4,4):
            raise ValueError('Expected homogeneous 4x4 body and camera transforms')
        camera_from_body=np.linalg.inv(world_from_camera)@world_from_body
        R=camera_from_body[:3,:3];C=camera_from_body[:3,3]
        # USD-authored transforms retain about 1e-7 orthogonality roundoff.
        # Keep the saved rotation unchanged instead of silently refitting it.
        if not np.isfinite(camera_from_body).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0) or abs(np.linalg.det(R)-1)>1e-6:
            raise ValueError('Saved instrument frame is not a proper rigid rotation')
        if not np.allclose(C,e['tool']['shaft_reference_position_mm'],atol=1e-8,rtol=0):
            raise ValueError('Saved instrument origin disagrees with C')
        if not np.allclose(R[:,2],e['tool']['shaft_direction_unit'],atol=1e-7,rtol=0):
            raise ValueError('Saved instrument +Z disagrees with shaft direction d')
        body_poses[eye]=camera_from_body

    def path(C, end, e):
        return [project([C[j]+i/60*(end[j]-C[j]) for j in range(3)], e['K'], e['D']) for i in range(61)]

    def length(points):
        return sum(math.dist(a,b) for a,b in zip(points,points[1:]))

    references={}
    for eye,e in record['eyes'].items():
        g=e['tool'];C=g['shaft_reference_position_mm'];d=g['shaft_direction_unit']
        ends=[[C[i]+15*d[i] for i in range(3)], g['jaw1_tip_position_mm'], g['jaw2_tip_position_mm']]
        references[eye]={'mm':min(math.dist(C,p) for p in ends),
                         'px':min(length(path(C,p,e)) for p in ends)}
    axis_mm=0.5*min(r['mm'] for r in references.values())
    if axis_mm<=0 or any(r['px']<=1e-8 for r in references.values()):
        raise ValueError('Cannot size coordinate axes against a zero-length pose arrow')
    for _ in range(100):
        panels={};fits=True
        for eye,e in record['eyes'].items():
            C=e['tool']['shaft_reference_position_mm'];axes=[]
            for j,name in enumerate('XYZ'):
                direction=body_poses[eye][:3,j].tolist()
                end=[C[i]+axis_mm*direction[i] for i in range(3)]
                points=path(C,end,e);pixels=length(points)
                axes.append({'name':name,'start_camera_mm':C,'end_camera_mm':end,
                             'direction_camera_unit':direction,
                             'projected_polyline_pixels':points,'projected_length_px':pixels})
                fits &= pixels <= XYZ_MAX_PROJECTED_RATIO*references[eye]['px']
            panels[eye]={'frame':body,'expressed_in_frame':eye+'_optical','anchor':'C','length_mm':axis_mm,
                         'source':'saved_full_instrument_body_pose_truth',
                         'T_optical_from_body_mm':body_poses[eye].tolist(),
                         'Z_axis_matches_shaft_direction':True,
                         'reference_shortest_pose_arrow_mm':references[eye]['mm'],
                         'reference_shortest_pose_arrow_px':references[eye]['px'],
                         'max_projected_length_ratio':XYZ_MAX_PROJECTED_RATIO,
                         'display_only':True,'axes':axes}
        if fits:return panels
        axis_mm*=0.9
    raise ValueError('Unable to size projected coordinate axes')


def articulated_axis_geometry(record, pose_frame, hinges):
    """Shaft axes at C; native jaw axes at their authored USD hinge centers."""
    import numpy as np
    result=instrument_axis_geometry(record,pose_frame)
    world_poses=pose_frame['T_world_from_body_mm']
    jaws={eye:[] for eye in record['eyes']}
    for j in (1,2):
        definition=hinges['jaws'][f'jaw{j}']
        if not definition['jaw_body'].endswith(f'/psm_tool_gripper{j}_link'):
            raise ValueError('Jaw hinge definition has the wrong physical identity')
        world_body=np.array(world_poses[definition['jaw_body']])
        world_child_joint=world_body@np.array(definition['T_body1_from_joint1_mm'])
        world_parent_joint=np.array(world_poses[definition['parent_body']])@np.array(definition['T_body0_from_joint0_mm'])
        gap=float(np.linalg.norm(world_child_joint[:3,3]-world_parent_joint[:3,3]))
        if gap>1e-4:raise ValueError('Saved jaw body does not meet its physical USD hinge')
        world_hinge_frame=world_body.copy()
        # Use the jaw's native XYZ orientation, translated to the actual hinge.
        # The body origin is never assumed to be the joint center.
        world_hinge_frame[:3,3]=world_child_joint[:3,3]
        for eye,e in record['eyes'].items():
            optical=np.linalg.solve(np.array(pose_frame['camera_truth']['T_world_from_optical_mm'][eye]),world_hinge_frame)
            R=optical[:3,:3];H=optical[:3,3].tolist();tip=e['tool'][f'jaw{j}_tip_position_mm']
            if not np.isfinite(optical).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0) or abs(np.linalg.det(R)-1)>1e-6:
                raise ValueError('Saved jaw orientation is not a proper rotation')
            if H[2]<=0 or tip[2]<=0:raise ValueError('Jaw annotation is behind the optical center')
            path=[project([H[k]+i/60*(tip[k]-H[k]) for k in range(3)],e['K'],e['D']) for i in range(61)]
            jaws[eye].append({'frame':definition['jaw_body'],'expressed_in_frame':eye+'_optical',
                'anchor':f'H{j}','jaw_index':j,'hinge_camera_mm':H,'hinge_pixel':project(H,e['K'],e['D']),
                'tip_camera_mm':tip,'tip_pixel':project(tip,e['K'],e['D']),
                'hinge_to_tip_vector_mm':[tip[k]-H[k] for k in range(3)],
                'hinge_to_tip_polyline_pixels':path,
                'hinge_to_tip_length_mm':math.dist(H,tip),
                'hinge_to_tip_projected_length_px':sum(math.dist(a,b) for a,b in zip(path,path[1:])),
                'T_optical_from_hinge_frame_mm':optical.tolist(),
                'joint_path':definition['joint_path'],'origin_source':'USD physics:localPos1 on body1',
                'orientation_source':'saved native jaw-body rotation, translated to hinge',
                'max_parent_child_anchor_gap_mm':gap,'surface_visibility':'not_assigned_joint_reference'})
    # Equal short lengths for both jaws and both eyes; preserve overlaps.
    axis_mm=.5*min(v['hinge_to_tip_length_mm'] for values in jaws.values() for v in values)
    if axis_mm<=0:raise ValueError('Hinge-to-tip arrow has zero length')
    for _ in range(100):
        fits=True
        for eye,values in jaws.items():
            e=record['eyes'][eye]
            for v in values:
                H=v['hinge_camera_mm'];R=np.array(v['T_optical_from_hinge_frame_mm'])[:3,:3]
                axes=[]
                for j,name in enumerate('XYZ'):
                    direction=R[:,j].tolist();end=[H[k]+axis_mm*direction[k] for k in range(3)]
                    path=[project([H[k]+i/60*(end[k]-H[k]) for k in range(3)],e['K'],e['D']) for i in range(61)]
                    length=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
                    axes.append({'name':name,'start_camera_mm':H,'end_camera_mm':end,
                        'direction_camera_unit':direction,'projected_polyline_pixels':path,'projected_length_px':length})
                    fits &= length<=XYZ_MAX_PROJECTED_RATIO*v['hinge_to_tip_projected_length_px']
                v.update(axes=axes,length_mm=axis_mm,max_projected_length_ratio=XYZ_MAX_PROJECTED_RATIO,display_only=True)
        if fits:break
        axis_mm*=.9
    else:raise ValueError('Unable to size jaw coordinate axes')
    for eye,values in jaws.items():
        result[eye]['jaw_frames']=values
        result[eye]['H1_H2_separation_mm']=math.dist(values[0]['hinge_camera_mm'],values[1]['hinge_camera_mm'])
        result[eye]['shared_hinge_within_1e_4_mm']=result[eye]['H1_H2_separation_mm']<1e-4
    return result


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


def vector_panel(image, record, eye, font_path, split_jaws=False, coordinate_axes=None):
    """Shaft frame at C and articulated jaw frames/arrows at physical hinges."""
    w,h=image.size; header,footer=50,472 if split_jaws else 100
    canvas=Image.new('RGB',(w,h+header+footer),'#101820');canvas.paste(image,(0,header))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(font_path),25);small=ImageFont.truetype(str(font_path),22)
    e=record['eyes'][eye];g=e['tool']
    C=g['shaft_reference_position_mm'];T=g['tip_position_mm'];direction=g['shaft_direction_unit']
    uvC=project(C,e['K'],e['D']);uvT=project(T,e['K'],e['D'])
    residual=max(abs(uvC[i]-g['shaft_reference_pixel'][i]) for i in range(2))
    residual=max(residual,max(abs(uvT[i]-g['tip_pixel'][i]) for i in range(2)))
    tips=[g['jaw1_tip_position_mm'],g['jaw2_tip_position_mm']]
    if split_jaws:
        if coordinate_axes is None or 'jaw_frames' not in coordinate_axes[eye]:
            raise ValueError('Separate jaw annotations require real hinge definitions and full pose truth')
        triad=coordinate_axes[eye]
        hinge_points=[v['hinge_camera_mm'] for v in triad['jaw_frames']]
    uv_tips=[project(p,e['K'],e['D']) for p in tips]
    residual=max(residual,max(abs(uv_tips[j][i]-g['jaw_tip_pixels'][j][i]) for j in range(2) for i in range(2)))
    if residual>=1e-6:raise ValueError('Saved C/T/J1/J2 projection mismatch')
    endpoint=[C[i]+15*direction[i] for i in range(3)]
    def curve(end,start=None):
        start=C if start is None else start
        result=[project([start[j]+i/60*(end[j]-start[j]) for j in range(3)],e['K'],e['D']) for i in range(61)]
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
    jaw_curves=[curve(p,hinge_points[j] if split_jaws else C) for j,p in enumerate(tips)]
    if split_jaws:
        # Different symbol widths keep nearly coincident closed-jaw arrows
        # distinguishable without changing either physical endpoint.
        arrow(jaw_curves[1],orange,width=6,broken=not e['jaw_visibility'][1]['visible'],head_size=21)
        arrow(jaw_curves[0],blue,width=3,broken=not e['jaw_visibility'][0]['visible'],head_size=14)
    else:
        arrow(position,blue,width=5,broken=tissue_hidden)
    arrow(axis,yellow,width=3)
    # Jaw orientation diagrams move to separate corners in screen space only.
    # All physical hinge/tip positions and 3D geometry stay unchanged.
    inset_rectangles=[(16,12,296,252),(w-296,12,w-16,252)] if split_jaws else []
    protected_labels=[(a,b+header,c,d+header) for a,b,c,d in inset_rectangles]
    jaw_axis_insets=[]
    if split_jaws:
        axis_groups=[]
        for part,suffix in [(triad,'')]:
            for item,color in zip(part['axes'],XYZ_COLORS):
                points=[(x,y+header) for x,y in item['projected_polyline_pixels']]
                arrow(points,color,width=3 if suffix!='2' else 2,head_size=10)
                same=next((group for group in axis_groups if group['item']['name']==item['name']
                    and math.dist(group['item']['start_camera_mm'],item['start_camera_mm'])<1e-4
                    and math.dist(group['item']['end_camera_mm'],item['end_camera_mm'])<1e-4),None)
                if same:same['names'].append(item['name']+suffix)
                else:axis_groups.append({'item':item,'points':points,'color':color,'names':[item['name']+suffix]})
        for group in axis_groups:
            points,color=group['points'],group['color'];text='+'+'/'.join(group['names'])
            dx,dy=points[-1][0]-points[0][0],points[-1][1]-points[0][1]
            norm=math.hypot(dx,dy)
            ux,uy=(dx/norm,dy/norm) if norm>1e-8 else (0.,-1.)
            # Only text moves for readability. Real axis endpoints and shared
            # hinge positions are never offset to separate coincident frames.
            choices=[]
            for i,(along,side) in enumerate(((20,0),(32,0),(22,22),(22,-22),(35,35),(35,-35),(50,0))):
                x=max(60,min(w-60,points[-1][0]+along*ux-side*uy))
                y=max(header+18,min(header+h-18,points[-1][1]+along*uy+side*ux))
                box=draw.textbbox((x,y),text,font=small,anchor='mm',stroke_width=3)
                choices.append((sum(overlap(box,p) for p in protected_labels)*1000+i,x,y,box))
            _,x,y,box=min(choices)
            if math.dist((x,y),points[-1])>30:
                draw.line((points[-1],(x,y)),fill=color,width=1)
            draw.text((x,y),text,font=small,anchor='mm',fill=color,
                      stroke_width=3,stroke_fill='#101820')
            protected_labels.append((box[0]-8,box[1]-8,box[2]+8,box[3]+8))
    # Labels use thin neutral leaders, never additional vector arrows or markers.
    def label(text,point,offset,color,leader=True):
        x,y=point;box=draw.textbbox((0,0),text,font=font);bw=box[2]-box[0]+20;bh=44
        offsets=[offset,(offset[0]+80,offset[1]-55),(offset[0]+120,offset[1]+55),
                 (-bw-80,-140),(70,130),(-bw/2,-180)]
        choices=[]
        for i,(ox,oy) in enumerate(offsets):
            a=max(12,min(w-bw-12,x+ox));b=max(header+12,min(header+h-bh-12,y+oy))
            rect=(a,b,a+bw,b+bh)
            choices.append((sum(overlap(rect,p) for p in protected_labels)*1000+i,rect))
        _,(a,b,_,_)=min(choices)
        protected_labels.append((a-5,b-5,a+bw+5,b+bh+5))
        if leader:
            anchor=(max(a,min(a+bw,x)),max(b,min(b+bh,y)))
            draw.line((point,anchor),fill='#d3d9e0',width=1)
        draw.rounded_rectangle((a,b,a+bw,b+bh),radius=5,fill='#101820',outline=color,width=1)
        draw.text((a+10,b+7),text,font=font,fill=color)
    label('C 杆体坐标原点',axis[0],(55,85),'#e5edf4')
    label('d 杆轴方向',axis[30],(65,-110),yellow,leader=False)
    if split_jaws:
        if triad['shared_hinge_within_1e_4_mm']:
            label('H1 / H2 真实铰接中心',jaw_curves[0][0],(-270,155),'#e5edf4')
        else:
            for j,color in enumerate((blue,orange)):
                label(f'H{j+1} 钳口{j+1}铰接中心',jaw_curves[j][0],(-270,155+60*j),color)
        for j,color in enumerate((blue,orange)):
            v=e['jaw_visibility'][j]
            state='出画' if not v['in_frame'] else '可见' if v['visible'] else '组织遮挡' if v['occluder_kind']=='tissue' else '自遮挡'
            upper=uv_tips[j][1]<=uv_tips[1-j][1]
            label(f'J{j+1} 钳口{j+1}尖端 · {state}',jaw_curves[j][-1],(-390,-130 if upper else 100),color)
    else:
        label('T 尖端位置',position[-1],(-270,-100),blue)
    if split_jaws:
        inset_font=ImageFont.truetype(str(font_path),18)
        for v,rect,border in zip(triad['jaw_frames'],inset_rectangles,(blue,orange)):
            a,b,c,d=rect;j=v['jaw_index']
            origin=[(a+c)/2,b+126]
            offset=[origin[k]-v['hinge_pixel'][k] for k in range(2)]
            draw.rounded_rectangle((a,b+header,c,d+header),radius=9,fill='#111d2a',outline=border,width=2)
            draw.text((a+14,b+header+10),f'钳口 {j} · XYZ 姿态',font=small,fill=border)
            draw.text((a+12,d+header-29),f'仅移位显示；真实 H{j} 见器械',font=inset_font,fill='#d3dce7')
            inset_axes=[];label_boxes=[]
            for item,color in zip(v['axes'],XYZ_COLORS):
                translated=[[p[k]+offset[k] for k in range(2)] for p in item['projected_polyline_pixels']]
                points=[(x,y+header) for x,y in translated]
                arrow(points,color,width=3,head_size=10)
                dx,dy=translated[-1][0]-origin[0],translated[-1][1]-origin[1]
                norm=math.hypot(dx,dy);ux,uy=(dx/norm,dy/norm) if norm>1e-8 else (0.,-1.)
                text=f"+{item['name']}{j}";choices=[]
                for i,(along,side) in enumerate(((20,0),(32,0),(22,20),(22,-20),(34,30),(34,-30))):
                    x=max(a+32,min(c-32,translated[-1][0]+along*ux-side*uy))
                    y=max(b+51,min(d-48,translated[-1][1]+along*uy+side*ux))+header
                    box=draw.textbbox((x,y),text,font=small,anchor='mm',stroke_width=2)
                    choices.append((sum(overlap(box,q) for q in label_boxes)*1000+i,x,y,box))
                _,x,y,box=min(choices)
                if math.dist((x,y),points[-1])>30:draw.line((points[-1],(x,y)),fill=color,width=1)
                draw.text((x,y),text,font=small,anchor='mm',fill=color,stroke_width=2,stroke_fill='#111d2a')
                label_boxes.append((box[0]-5,box[1]-5,box[2]+5,box[3]+5))
                inset_axes.append({'name':item['name'],'display_polyline_pixels':translated})
            jaw_axis_insets.append({'jaw_index':j,'physical_anchor':v['anchor'],
                'placement':'top_left' if j==1 else 'top_right','bounds_image_pixels':list(rect),
                'physical_hinge_pixel':v['hinge_pixel'],'display_origin_pixel':origin,
                'screen_translation_px':offset,'screen_scale':1.,'display_position_is_truth':False,
                'projected_directions_and_lengths_unchanged':True,'axes':inset_axes})
    title='左目 LEFT' if eye=='left' else '右目 RIGHT'
    kind='杆体 / 双钳口局部坐标轴' if split_jaws else '双箭头真值投影'
    draw.text((16,10),f"{title} · {kind}   {record['sequence_id']} / {record['job_id']}",font=font,fill='white')
    if split_jaws:
        theta=record['jaw_half_angle_deg']
        y=header+h
        fmt=lambda v,digits=2: '('+', '.join(f'{x:+.{digits}f}' for x in v)+')'
        draw.text((16,y+10),'黄：C → C+15d    蓝：H1 → J1    橙：H2 → J2    |    尖端箭头：实线可见、虚线遮挡',font=small,fill='white')
        draw.text((16,y+44),'杆体 XYZ 在 C；钳口1姿态在左上角，钳口2在右上角（仅移位显示）。红 X、绿 Y、蓝紫 Z。',font=small,fill='#e5edf4')
        hinge_note='H1/H2 为模型共用的真实铰接位置，未人为错开。' if triad['shared_hinge_within_1e_4_mm'] else 'H1/H2 取自模型真实关节定义。'
        draw.text((16,y+78),hinge_note+f'下方 XYZ 数值仍属于{title}相机坐标系。',font=small,fill='#c4ced8')
        rows=[('C  杆轴参考点位置',C,'mm','#e5edf4',2),
              ('d  杆轴单位方向',direction,'无单位',yellow,4),
              ('J1 钳口1尖端位置',tips[0],'mm',blue,2),
              ('J2 钳口2尖端位置',tips[1],'mm',orange,2),
              ('H1 钳口1铰接中心',hinge_points[0],'mm',blue,2),
              ('H2 钳口2铰接中心',hinge_points[1],'mm',orange,2)]
        for i,(name,values,unit,color,digits) in enumerate(rows):
            draw.text((16,y+113+34*i),f'{name}    (X, Y, Z) = {fmt(values,digits)} {unit}',font=font,fill=color)
        # Tip positions are camera-relative; displayed jaw arrows start at H1/H2.
        for j,color in enumerate((blue,orange)):
            delta=[tips[j][i]-hinge_points[j][i] for i in range(3)]
            draw.text((16,y+324+31*j),f'H{j+1} → J{j+1} 箭头位移  (ΔX, ΔY, ΔZ) = {fmt(delta)} mm',font=small,fill=color)
        draw.text((16,y+393),f'每片钳口半开合角 θ = {theta:.2f}°    |    对称开合角 2θ = {2*theta:.2f}°',font=font,fill='#e5edf4')
        note='闭合时两个尖端投影接近重合，未人为错开。' if theta<1e-6 else 'J1/J2 固定对应两片钳口，未用中点代替。'
        draw.text((16,y+435),note+'保存真值投影，非模型预测；原图未改动，未重新渲染。',font=small,fill='#c4ced8')
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
                      jaw_position_indicator_vectors_mm=[[p[i]-hinge_points[j][i] for i in range(3)] for j,p in enumerate(tips)],
                      jaw_position_indicator_origins=['H1','H2'],
                      H1_camera_mm=hinge_points[0],H2_camera_mm=hinge_points[1],
                      H1_pixel=triad['jaw_frames'][0]['hinge_pixel'],H2_pixel=triad['jaw_frames'][1]['hinge_pixel'],
                      jaw_visibility=e['jaw_visibility'],jaw_half_angle_deg=record['jaw_half_angle_deg'],
                      jaw_symmetric_opening_deg=2*record['jaw_half_angle_deg'],T_displayed=False,
                      coordinate_axis_arrows=triad,
                      jaw_axis_insets=jaw_axis_insets,
                      xyz_annotations={'frame':f'{eye}_optical','axes':'X right, Y down, Z forward',
                          'position_origin':'camera optical center','position_unit':'mm',
                          'direction_unit':'unit vector, dimensionless',
                          'positions_displayed':['C','J1','J2','H1','H2'],'direction_displayed':'d',
                          'arrow_displacements_displayed':['J1-H1','J2-H2'],
                          'display_decimal_places':{'position_and_displacement':2,'direction':4},
                          'rounded_for_display_only':True})
    return canvas,result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True, help='Chinese-capable TrueType/OpenType font')
    p.add_argument('--style', choices=('three-arrows','two-arrows','points'), default='three-arrows')
    p.add_argument('--pose-geometry',type=Path,help='Saved full body transforms; default: RUN/pose_geometry/geometry.json')
    p.add_argument('--jaw-hinges',type=Path,help='Authored USD joint anchors; default: RUN/pose_geometry/jaw_hinges.json')
    a = p.parse_args()
    labels = sorted(a.run.glob('sequences/*/labels/*.json'))
    if not labels:
        raise ValueError('No saved labels found')
    pose_path=None;pose_frames={};pose_hash=None;hinge_path=None;hinge_hash=None;hinges=None
    if a.style=='three-arrows':
        pose_path=a.pose_geometry or a.run/'pose_geometry/geometry.json'
        if not pose_path.is_file():
            raise FileNotFoundError(f'Instrument XYZ axes require existing full body-pose truth: {pose_path}. No camera-axis fallback or new rendering is performed.')
        pose_hash=sha(pose_path)
        geometry=json.loads(pose_path.read_text())
        if geometry['unit']!='mm':raise ValueError('Saved body transforms must use millimeters')
        hinge_path=a.jaw_hinges or a.run/'pose_geometry/jaw_hinges.json'
        if not hinge_path.is_file():
            raise FileNotFoundError(f'Real jaw hinges require {hinge_path}. Export USD anchors with tools/export_jaw_hinges.py; no rendering is needed.')
        hinge_hash=sha(hinge_path);hinges=json.loads(hinge_path.read_text())
        if hinges['unit']!='mm' or hinges['pose_geometry_sha256']!=pose_hash or hinges['source_scene_sha256']!=geometry['source_scene_sha256']:
            raise ValueError('Jaw hinge definitions do not match the saved pose geometry')
        for frame in geometry['frames']:
            key=(frame['sequence_id'],frame['job_id'])
            if key in pose_frames:raise ValueError('Duplicate body-pose frame')
            pose_frames[key]=frame
    original = {str(f.relative_to(a.run)): sha(f) for f in (a.run/'sequences').rglob('*') if f.is_file()}
    a.output.mkdir(parents=True, exist_ok=True)
    files = []
    for path in labels:
        record = json.loads(path.read_text())
        panels = []
        projections = {}
        coordinate_axes=None
        if a.style=='three-arrows':
            pose_frame=pose_frames[(record['sequence_id'],record['job_id'])]
            if pose_frame['label_sha256']!=sha(path):
                raise ValueError('Saved full body-pose truth does not match the source label')
            coordinate_axes=articulated_axis_geometry(record,pose_frame,hinges)
        for eye in ('left', 'right'):
            e = record['eyes'][eye]
            source = (path.parent/e['image']).resolve()
            if sha(source) != e['image_sha256']:
                raise ValueError(f'Original image hash mismatch: {source}')
            rgb=Image.open(source).convert('RGB')
            if a.style=='points':preview,projections[eye]=point_panel(rgb,record,eye,a.font)
            else:preview,projections[eye]=vector_panel(rgb,record,eye,a.font,split_jaws=a.style=='three-arrows',coordinate_axes=coordinate_axes)
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
    if pose_path and sha(pose_path)!=pose_hash:
        raise RuntimeError('Saved full body-pose truth changed')
    if hinge_path and sha(hinge_path)!=hinge_hash:
        raise RuntimeError('USD jaw hinge definitions changed')
    report = {'renderer_invoked': False, 'source_files_unchanged': True, 'source_file_count': len(original),
              'preview_pairs': len(files), 'arrow_mm_relative_to_C': [0, 15], 'arrow_start': 'C', 'arrow_direction': '+d',
              'style':a.style,
              'camera_xyz_annotations':a.style=='three-arrows',
              'camera_xyz_axis_arrows':False,
              'instrument_xyz_axis_arrows':a.style=='three-arrows',
              'instrument_axis_body':'psm_tool_roll_link' if pose_path else None,
              'pose_geometry_sha256':pose_hash,
              'pose_geometry_unchanged':True if pose_path else None,
              'jaw_hinges_sha256':hinge_hash,
              'jaw_hinges_unchanged':True if hinge_path else None,
              'jaw_xyz_axis_arrows':a.style=='three-arrows',
              'jaw_xyz_display_layout':'top_left_jaw1_top_right_jaw2_per_eye' if a.style=='three-arrows' else None,
              'jaw_tip_arrow_origins':['H1','H2'] if a.style=='three-arrows' else None,
              'T_display': 'not displayed; physical J1/J2 are shown separately' if a.style=='three-arrows' else 'blue position-indicator arrow C->T, vector T-C; model position remains camera-frame T' if a.style=='two-arrows' else 'midpoint projection crosshair with camera XYZ and image uv; no direction arrow',
              'projection_source': 'simulation_ground_truth_not_model_inference',
              'maximum_projection_difference_px': max(v['maximum_difference_from_saved_pixels_px'] for f in files for v in f['projection_results'].values()),
              'annotation_code_sha256': sha(Path(__file__)), 'font_sha256': sha(a.font), 'files': files}
    (a.output/'annotation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"Annotated {len(files)} existing pairs; all {len(original)} source files unchanged; zero renders.")


if __name__ == '__main__':
    main()
