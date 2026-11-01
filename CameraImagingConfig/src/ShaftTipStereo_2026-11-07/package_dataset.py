"""Validate per-eye simulation labels, then draw overlays on final raw images."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parent
native=root/'render/native';final=root/'final';out=root/'dataset'
out.mkdir(exist_ok=False)
for d in ['images','ground_truth','overlays','pairs','config']:(out/d).mkdir()
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(native/'manifest.json');assert m['completed'] and m['samples']==['0000','0001','0002']
assert m['renderer']=='PathTracing' and m['render_sampling']['spp_per_frame']==64
assert m['source_files_unchanged']
intr=read(final/'config/camera_intrinsics.json')
definition=read(native/'config/landmark_definitions.json')
preflight=read(root/'preflight.json')
assert read(final/'effect_report.json')['all_checks_passed']
accepted=root.parent/'SingleInstrument_2026-11-07/preview_final'
assert intr==read(accepted/'config/camera_intrinsics.json')
assert read(final/'config/dark_border_parameters.json')==read(accepted/'config/dark_border_parameters.json')
for name in ['camera_intrinsics.json','applied_parameters.json','dark_border_parameters.json','initial_parameters.json']:
    shutil.copy2(final/'config'/name,out/'config'/name)
shutil.copy2(native/'config/landmark_definitions.json',out/'config/landmark_definitions.json')
asset=Path('/home/leorobot/orbit-surgical/source/extensions/orbit.surgical.assets/data/Robots/dVRK/PSM/psm_col.usd')
write(out/'config/source_manifest.json',{'instrument_asset':{'path':str(asset),'sha256':sha(asset),'repository':'https://github.com/orbit-surgical/orbit-surgical'},
    'intrinsics_source':intr['source'],'source_scene_and_profile_hashes':m['source_files_sha256_before'],
    'accepted_imaging_reference':str(accepted/'config'),'landmark_definition_authority':'Dataset conventions defined from the actual USD geometry; not a manufacturer-supplied landmark annotation.'})
font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(font_path,25);small=ImageFont.truetype(font_path,22)
cyan=(25,235,255);yellow=(255,225,35);pink=(255,75,170)
def label(draw,xy,text,color):
    draw.text(xy,text,font=small,fill=color,stroke_width=3,stroke_fill=(8,13,20))
def draw_overlay(image,tool):
    image=image.copy().convert('RGB');draw=ImageDraw.Draw(image)
    line=[tuple(x) for x in tool['shaft_polyline_pixels_raw']]
    draw.line(line,fill=(0,0,0),width=8);draw.line(line,fill=yellow,width=4)
    end=np.asarray(line[-1]);back=np.asarray(line[-11]);v=end-back;v=v/np.linalg.norm(v);n=np.array([-v[1],v[0]])
    arrow=[tuple(end),tuple(end-17*v+7*n),tuple(end-17*v-7*n)]
    draw.polygon(arrow,fill=yellow)
    x,y=tool['shaft_reference_pixel_raw']
    draw.ellipse((x-7,y-7,x+7,y+7),outline=(0,0,0),width=5)
    draw.ellipse((x-6,y-6,x+6,y+6),outline=cyan,width=3)
    label(draw,(x-35,y+19),'C: shaft reference',cyan)
    x,y=tool['tip_pixel_raw']
    for width,color in [(7,(0,0,0)),(3,pink)]:
        draw.line((x-11,y,x+11,y),fill=color,width=width);draw.line((x,y-11,x,y+11),fill=color,width=width)
    label(draw,(x-30,y-43),'T: jaw tip',pink)
    sx,sy=line[len(line)//2];label(draw,(sx-25,sy-39),'shaft axis',yellow)
    return image

def pair(images,sid,kind):
    canvas=Image.new('RGB',(3840,1140),'#142033');draw=ImageDraw.Draw(canvas)
    for i,eye in enumerate(['left','right']):
        canvas.paste(images[eye],(1920*i,60));draw.text((1920*i+24,16),f'{sid} / {eye.upper()} / {kind}',font=font,fill='white')
    return canvas

checks=[];detail_rows=[];overview_rows=[];summary=[]
for idx,sid in enumerate(m['samples']):
    record=read(native/f'ground_truth/{sid}.json');tools=record['tools']['right_tool']
    output={'sample_id':sid,'source':'Isaac Sim frozen-stage geometry; not a model prediction',
        'position_unit':'mm','direction_unit':'unit vector','optical_axes':definition['optical_axes'],
        'pixel_domain':'Original 1920x1080 distorted image; no stereo rectification',
        'landmark_definitions':'../config/landmark_definitions.json','eyes':{}}
    images={};overlays={};crops=[]
    for eye in ['left','right']:
        tool=tools[eye]
        q=np.array([tool['shaft_reference_position_mm'],tool['tip_position_mm']])
        d=np.array(tool['shaft_direction_unit']);assert abs(np.linalg.norm(d)-1)<1e-10
        assert q[:,2].min()>0
        uv=cv2.projectPoints(q,np.zeros(3),np.zeros(3),np.array(intr[eye]['K']),np.array(intr[eye]['D']))[0].reshape(-1,2)
        target=np.array([tool['shaft_reference_pixel_raw'],tool['tip_pixel_raw']])
        error=float(abs(uv-target).max());assert error<1e-6
        line=q[0]+np.array(tool['shaft_polyline_s_mm'])[:,None]*d
        uv_line=cv2.projectPoints(line,np.zeros(3),np.zeros(3),np.array(intr[eye]['K']),np.array(intr[eye]['D']))[0].reshape(-1,2)
        line_error=float(abs(uv_line-np.array(tool['shaft_polyline_pixels_raw'])).max());assert line_error<1e-6
        reference=preflight['samples'][idx]['labels'][eye]
        for field in ['shaft_reference_position_mm','tip_position_mm','shaft_direction_unit']:
            assert np.allclose(tool[field],reference[field],rtol=0,atol=1e-6),(sid,eye,field)
        src=final/f'images/{sid}_{eye}.png';dst=out/f'images/{sid}_{eye}.png';shutil.copy2(src,dst)
        im=Image.open(dst).convert('RGB');assert im.size==(1920,1080)
        images[eye]=im;overlays[eye]=draw_overlay(im,tool)
        overlays[eye].save(out/f'overlays/{sid}_{eye}.png')
        output['eyes'][eye]={'image':f'../images/{sid}_{eye}.png','image_sha256':sha(dst),'K':intr[eye]['K'],'D':intr[eye]['D'],'tool':tool}
        alluv=np.concatenate([uv,uv_line]);cx,cy=(alluv.min(0)+alluv.max(0))/2
        # Fixed 800x400 pixel crops, only for enlarged visual review.
        x0=int(np.clip(round(cx)-400,0,1120));y0=int(np.clip(round(cy)-200,0,680))
        crops.append(overlays[eye].crop((x0,y0,x0+800,y0+400)))
        checks.append({'sample_id':sid,'eye':eye,'point_projection_max_error_px':error,'axis_projection_max_error_px':line_error,
                       'preflight_world_transform_agreement':True,'image_sha256':sha(dst),'crop_xywh':[x0,y0,800,400]})
    write(out/f'ground_truth/{sid}.json',output)
    pair(images,sid,'RAW STEREO').save(out/f'pairs/{sid}_raw.jpg',quality=95,subsampling=0)
    p=pair(overlays,sid,'SIMULATION GROUND TRUTH')
    p.save(out/f'pairs/{sid}_overlay.jpg',quality=95,subsampling=0)
    overview_rows.append(p.resize((1600,475),Image.Resampling.LANCZOS))
    row=Image.new('RGB',(1600,448),'#142033');dr=ImageDraw.Draw(row)
    for i,eye in enumerate(['left','right']):
        row.paste(crops[i],(800*i,48));dr.text((800*i+18,11),f'{sid} / {eye.upper()} / DETAIL',font=small,fill='white')
    row.save(out/f'pairs/{sid}_detail.jpg',quality=96,subsampling=0);detail_rows.append(row)
    summary.append({'sample_id':sid,'eyes':{eye:{k:tools[eye][k] for k in ['shaft_reference_position_mm','shaft_direction_unit','tip_position_mm']} for eye in tools}})

for name,rows in [('overview',overview_rows),('details',detail_rows)]:
    montage=Image.new('RGB',(rows[0].width,sum(x.height for x in rows)))
    y=0
    for row in rows:montage.paste(row,(0,y));y+=row.height
    montage.save(out/f'{name}.jpg',quality=96,subsampling=0)
write(out/'ground_truth_summary.json',summary)
write(out/'checks.json',{'passed':True,'new_stereo_pairs':3,'new_images':6,'renderer':'PathTracing','spp_per_frame':64,'total_spp_setting':256,
    'intrinsics_distortion_and_accepted_vignette_unchanged':True,'positions_unit':'mm','opencv_projection_independent_check':checks,
    'new_render_manifest':'../render/native/manifest.json','scene_source_sha256':m['source_files_sha256_before'],
    'scope':'Checks validate simulation-coordinate/projection implementation; overlays are exact geometric labels, not learned predictions or real-world accuracy evidence.'})
write(final/'manifest.json',{**read(final/'manifest.json'),'new_scene_render_performed':True,'source_run':'../render','native_scene':'../render/native/scene',
     'instrument_configuration':m['instrument_configuration'],'native_render_sampling':m['render_sampling']})
print('DATASET_COMPLETE',out,flush=True)
