"""Four controlled diagnostic combinations, with the original scale solver.

True R and normalized translation direction are deliberately used in controls;
metric translation/baseline is never supplied to reconstruct(). These checks
isolate the stages and are not claims of successful image-only calibration.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parent
out=root/'separation_check'
old=Path('/home/leorobot/capstone-1/Grasp5-Stereo-SurgPose/pixel_extraction')
sys.path.insert(0,str(old))
from geometry import reconstruct

prior=json.loads((root/'result.json').read_text())
truth=json.loads((out/'truth_from_scene.json').read_text())
known=json.loads((root/'known_AB.json').read_text())
cams=json.loads((root/'intrinsics.json').read_text())['cameras']
detected=prior['points']
perfect=truth['points']
estimated=prior['estimates']['blue_filter_removed']
R_est=np.array(estimated['R']);t_est=np.array(estimated['unit_t'])
R_true=np.array(truth['relative_pose']['R_right_from_left'])
t_true=np.array(truth['relative_pose']['unit_translation_right_from_left'])
baseline_true=truth['relative_pose']['baseline_mm']
length=known['known_length_mm']
assert abs(np.linalg.norm(t_true)-1)<1e-12
assert abs(np.linalg.norm(t_est)-1)<1e-12
assert hashlib.sha256((old/'geometry.py').read_bytes()).hexdigest()==prior['algorithm_sha256']

def run(name,points,R,t,point_source,pose_source):
    status,epi,baseline=reconstruct(points,cams,R,t,length)
    return {'name':name,'point_source':point_source,'pose_source':pose_source,
        'status':status,'epipolar_residual_px':epi,'baseline_estimated_mm':baseline,
        'baseline_true_mm':baseline_true,
        'signed_error_mm':None if baseline is None else baseline-baseline_true,
        'absolute_error_mm':None if baseline is None else abs(baseline-baseline_true),
        'relative_error_percent':None if baseline is None else 100*abs(baseline/baseline_true-1)}

cases=[
    run('both_true',perfect,R_true,t_true,'simulation_model_projection','simulation_R_and_unit_t'),
    run('pose_error_only',perfect,R_est,t_est,'simulation_model_projection','frozen_image_estimate'),
    run('endpoint_error_only',detected,R_true,t_true,'image_extracted','simulation_R_and_unit_t'),
    run('both_estimated',detected,R_est,t_est,'image_extracted','frozen_image_estimate'),
]
assert cases[0]['status']=='paired'
assert cases[0]['absolute_error_mm']<1e-5
assert abs(cases[3]['baseline_estimated_mm']-estimated['baseline_estimated_mm'])<1e-9

individual=[]
for idx,label in enumerate(['A','B']):
    mixed={eye:[list(v) for v in perfect[eye]] for eye in perfect}
    for eye in mixed:mixed[eye][idx]=detected[eye][idx]
    individual.append(run('only_'+label+'_detected',mixed,R_true,t_true,
        f'{label} from image, other endpoint from model','simulation_R_and_unit_t'))

point_errors={}
for eye in ['left','right']:
    error=np.array(detected[eye])-np.array(perfect[eye])
    point_errors[eye]={'truth':perfect[eye],'detected':detected[eye],
        'delta_uv_px':error.tolist(),'euclidean_error_px':np.linalg.norm(error,axis=1).tolist(),
        'true_AB_pixel_distance':float(np.linalg.norm(np.diff(np.array(perfect[eye]),axis=0))),
        'detected_AB_pixel_distance':float(np.linalg.norm(np.diff(np.array(detected[eye]),axis=0)))}

report={'purpose':'Separate endpoint extraction and relative-pose errors for the same single-instrument stereo pair.',
    'no_new_render':True,'no_refitting_to_truth':True,'old_solver_unchanged':True,
    'algorithm_sha256':prior['algorithm_sha256'],'known_length_mm':length,
    'true_baseline_used_only_for_reporting':True,
    'controls_use_true_R_and_unit_t_not_metric_translation':True,
    'cases':cases,'individual_endpoint_controls':individual,'point_errors':point_errors,
    'relative_pose_errors':{
        'rotation_deg':float(np.degrees(np.arccos(np.clip((np.trace(R_est@R_true.T)-1)/2,-1,1)))),
        'translation_direction_deg':float(np.degrees(np.arccos(np.clip(t_est@t_true,-1,1))))},
    'scope_notes':[
        'Ground-truth endpoints are projections of the previously chosen fixed model A/B definitions. This does not prove that a visible silhouette midpoint or ellipse center denotes exactly that physical point.',
        'The image-defined tip midpoint is not guaranteed to coincide with the projection of the 3D midpoint of closed-jaw ends.',
        'The four-case errors are nonlinear and must not be added as independent percentage contributions.',
        'Both-true is an implementation/units/projection consistency check, not evidence of image-only calibration accuracy.']}
(out/'result.json').write_text(json.dumps(report,indent=2))
with (out/'four_cases.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=cases[0].keys());writer.writeheader();writer.writerows(cases)

canvas=Image.new('RGB',(1600,560),'#142033')
draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
for ei,eye in enumerate(['left','right']):
    cropbox=(1160,590,1450,730) if eye=='left' else (1255,580,1545,720)
    im=Image.open(root/'input_images'/eye/'00.png').convert('RGB')
    crop=im.crop(cropbox).resize((783,378))
    d=ImageDraw.Draw(crop);scale=783/290
    def uv(p):return ((p[0]-cropbox[0])*scale,(p[1]-cropbox[1])*scale)
    d.line([uv(p) for p in perfect[eye]],fill='#00eeff',width=2)
    d.line([uv(p) for p in detected[eye]],fill='#ffe600',width=2)
    for label,gt,det in zip(['A','B'],perfect[eye],detected[eye]):
        x,y=uv(gt);u,v=uv(det)
        d.line([(x,y),(u,v)],fill='white',width=1)
        d.line([(x-6,y),(x+6,y)],fill='#00eeff',width=2)
        d.line([(x,y-6),(x,y+6)],fill='#00eeff',width=2)
        d.ellipse((u-5,v-5,u+5,v+5),outline='#ffe600',width=2)
        d.text((u+8,v+10),label,font=font,fill='white')
    canvas.paste(crop,(ei*800+8,90))
    draw.text((ei*800+12,12),eye.upper(),font=font,fill='white')
    draw.text((ei*800+12,45),'CYAN +: model truth     YELLOW o: extracted',font=small,fill='white')
    errors=point_errors[eye]['euclidean_error_px']
    draw.text((ei*800+12,488),f'Position error: A {errors[0]:.2f} px; B {errors[1]:.2f} px',font=font,fill='white')
canvas.save(out/'truth_vs_detection.png')
print(json.dumps({'cases':cases,'individual_endpoint_controls':individual,
    'relative_pose_errors':report['relative_pose_errors'],'point_errors':point_errors},indent=2))
