"""Shared approved drawing style for saved truth and camera-relative predictions."""
import math
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
from .projection import project_point as project
from .text_cache import CachedTextDraw

XYZ_COLORS = ('#ff554f', '#56ef77', '#8796ff')
XYZ_PANEL_BG_RGBA = (17, 29, 42, 102)

@lru_cache(maxsize=16)
def font_cached(path, size):
    return ImageFont.truetype(str(path), size)

def overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))

def vector_panel(image, record, eye, font_path, split_jaws=False, coordinate_axes=None, compact=False):
    """Shaft frame at C and articulated jaw frames/arrows at physical hinges."""
    w,h=image.size; header,footer=50,472 if split_jaws else 100
    source = record.get('display_source', 'simulation_ground_truth')
    if source not in ('simulation_ground_truth', 'ground_truth_replay', 'prediction'):
        raise ValueError('Unknown overlay source')
    has_quality=split_jaws and 'jaw_tip_region_visibility' in record['eyes'][eye]
    if has_quality:footer+=80
    if compact: footer=0
    canvas=Image.new('RGB',(w,h+header+footer),'#101820');canvas.paste(image,(0,header))
    draw=CachedTextDraw(canvas) if compact else ImageDraw.Draw(canvas)
    font=font_cached(str(font_path),25);small=font_cached(str(font_path),22)
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
    # Orientation diagrams move in screen space only: jaws above, shaft below.
    # All physical C/hinge/tip positions and 3D geometry stay unchanged.
    inset_rectangles=[(16,12,296,252),(w-296,12,w-16,252)] if split_jaws else []
    shaft_inset_rectangle=(16,h-252,296,h-12)
    protected_labels=[(a,b+header,c,d+header) for a,b,c,d in
                      (inset_rectangles+[shaft_inset_rectangle] if split_jaws else [])]
    jaw_axis_insets=[]
    shaft_axis_inset=None
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
            label('H1 / H2 铰接中心（预测）' if source=='prediction' else 'H1 / H2 真实铰接中心',jaw_curves[0][0],(-270,155),'#e5edf4')
        else:
            for j,color in enumerate((blue,orange)):
                label(f'H{j+1} 钳口{j+1}铰接中心',jaw_curves[j][0],(-270,155+60*j),color)
        for j,color in enumerate((blue,orange)):
            v=e['jaw_visibility'][j]
            state='出画' if not v['in_frame'] else '可见性未知' if v.get('visible') is None else '可见' if v['visible'] else '组织遮挡' if v['occluder_kind']=='tissue' else '自遮挡' if v['occluder_kind']=='instrument' else '遮挡'
            upper=uv_tips[j][1]<=uv_tips[1-j][1]
            name=f'J{j+1} 固定尖端点' if has_quality else f'J{j+1} 钳口{j+1}尖端'
            label(f'{name} · {state}',jaw_curves[j][-1],(-390,-130 if upper else 100),color)
    else:
        label('T 尖端位置',position[-1],(-270,-100),blue)
    if split_jaws:
        inset_font=font_cached(str(font_path),18)
        inset_specs=[(v,rect,border) for v,rect,border in
                     zip(triad['jaw_frames'],inset_rectangles,(blue,orange))]
        inset_specs.append((triad,shaft_inset_rectangle,yellow))
        for v,rect,border in inset_specs:
            a,b,c,d=rect;j=v.get('jaw_index')
            physical_origin=v['hinge_pixel'] if j is not None else uvC
            anchor=v['anchor']
            origin=[(a+c)/2,b+126]
            offset=[origin[k]-physical_origin[k] for k in range(2)]
            # Blend only the panel background; arrows, border and text stay opaque.
            ImageDraw.Draw(canvas,'RGBA').rounded_rectangle(
                (a,b+header,c,d+header),radius=9,fill=XYZ_PANEL_BG_RGBA,outline=border,width=2)
            heading=f'钳口 {j} · XYZ 姿态' if j is not None else '杆体 · XYZ 姿态'
            draw.text((a+14,b+header+10),heading,font=small,fill=border)
            anchor_note=f'仅移位显示；预测 {anchor} 见器械' if source=='prediction' else f'仅移位显示；真实 {anchor} 见器械'
            draw.text((a+12,d+header-29),anchor_note,font=inset_font,fill='#d3dce7')
            inset_axes=[];label_boxes=[]
            for item,color in zip(v['axes'],XYZ_COLORS):
                translated=[[p[k]+offset[k] for k in range(2)] for p in item['projected_polyline_pixels']]
                points=[(x,y+header) for x,y in translated]
                arrow(points,color,width=3,head_size=10)
                dx,dy=translated[-1][0]-origin[0],translated[-1][1]-origin[1]
                norm=math.hypot(dx,dy);ux,uy=(dx/norm,dy/norm) if norm>1e-8 else (0.,-1.)
                text=f"+{item['name']}"+(str(j) if j is not None else '');choices=[]
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
            inset={'physical_anchor':anchor,
                'placement':'bottom_left' if j is None else 'top_left' if j==1 else 'top_right',
                'bounds_image_pixels':list(rect),'display_origin_pixel':origin,
                'screen_translation_px':offset,'screen_scale':1.,'display_position_is_truth':False,
                'projected_directions_and_lengths_unchanged':True,'axes':inset_axes}
            if j is None:
                inset['physical_origin_pixel']=physical_origin
                shaft_axis_inset=inset
            else:
                inset.update(jaw_index=j,physical_hinge_pixel=physical_origin)
                jaw_axis_insets.append(inset)
    title='左目 LEFT' if eye=='left' else '右目 RIGHT'
    kind='杆体 / 双钳口局部坐标轴' if split_jaws else '双箭头真值投影'
    if source != 'simulation_ground_truth': kind = ('模型预测 PREDICTION' if source == 'prediction' else '真值输入回放 · 非模型预测')
    draw.text((16,10),f"{title} · {kind}   {record['sequence_id']} / {record['job_id']}",font=font,fill='white')
    if compact:
        pass
    elif split_jaws:
        theta=record['jaw_half_angle_deg']
        y=header+h
        fmt=lambda v,digits=2: '('+', '.join(f'{x:+.{digits}f}' for x in v)+')'
        legend='黄：C → C+15d    蓝：H1 → J1    橙：H2 → J2    |    尖端箭头：实线可见、虚线遮挡'
        if source!='simulation_ground_truth':legend='黄：C → C+15d    蓝：H1 → J1    橙：H2 → J2    |    实线：报告可见；虚线：遮挡或未知'
        draw.text((16,y+10),legend,font=small,fill='white')
        draw.text((16,y+44),'杆体 XYZ 在左下方；钳口1在左上角，钳口2在右上角（仅移位显示）。红 X、绿 Y、蓝紫 Z。',font=small,fill='#e5edf4')
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
        provenance = '保存真值投影，非模型预测；原图未改动，未重新渲染。' if source == 'simulation_ground_truth' else '模型预测值的运动学解码与投影。' if source == 'prediction' else '真值转换为17个数后的回放，不是训练模型预测。'
        draw.text((16,y+435),note+provenance,font=small,fill='#c4ced8')
        if has_quality:
            parts=[]
            for region in e['jaw_tip_region_visibility']:
                state='区域可见' if region['region_visible'] else '区域未见'
                parts.append(f"钳口{region['jaw_index']}：{state}，可见表面采样 {region['sample_counts']['visible']}/{region['surface_sample_count']}")
            draw.text((16,y+475),'；'.join(parts),font=small,fill='#c4f0da')
            quality=e['image_quality'];state='过曝标记' if quality['overexposed'] else '未触发过曝标记'
            draw.text((16,y+511),f"{state}：近白像素 {quality['near_white_fraction']*100:.2f}%（阈值 {quality['frame_fraction_threshold']*100:.1f}%）；与点/区域可见性分别记录。",
                      font=small,fill='#ffb460' if quality['overexposed'] else '#c4ced8')
    else:
        draw.text((16,header+h+10),'黄色：C → C + 15 mm × d（杆轴方向）    蓝色：C → T（尖端位置指示，位移 T−C）',font=small,fill='white')
        note='蓝色虚线：两实际钳口尖端均被组织遮挡。' if tissue_hidden else 'T 为两实际钳口尖端的中点。'
        draw.text((16,header+h+47),note+'保存的仿真真值投影，非模型预测；原图和真值未改动，未重新渲染。',font=small,fill='#c4ced8')
    result={'source':source,'C_camera_mm':C,'T_camera_mm':T,'d_unit':direction,
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
                      shaft_axis_inset=shaft_axis_inset,
                      xyz_annotations={'frame':f'{eye}_optical','axes':'X right, Y down, Z forward',
                          'position_origin':'camera optical center','position_unit':'mm',
                          'direction_unit':'unit vector, dimensionless',
                          'positions_displayed':['C','J1','J2','H1','H2'],'direction_displayed':'d',
                          'arrow_displacements_displayed':['J1-H1','J2-H2'],
                          'display_decimal_places':{'position_and_displacement':2,'direction':4},
                          'rounded_for_display_only':True})
    return canvas,result
