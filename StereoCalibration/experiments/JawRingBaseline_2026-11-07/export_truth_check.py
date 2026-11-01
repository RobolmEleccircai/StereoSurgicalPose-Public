"""Export fresh A/B projections directly from the saved Isaac Sim USD scene.

This is explicitly a ground-truth diagnostic, not an image detector. No renderer
is launched and no existing scene or test result is modified.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from pxr import Usd, UsdGeom

root=Path(__file__).resolve().parent
out=root/'separation_check'
out.mkdir(exist_ok=True)
capture=root.parent/'SingleInstrument_2026-11-07/preview'
scene=capture/'native/scene/sample_0000.usda'
stage=Usd.Stage.Open(str(scene),Usd.Stage.LoadNone)
unit=float(UsdGeom.GetStageMetersPerUnit(stage))
known=json.loads((root/'known_AB.json').read_text())
intrinsics=json.loads((capture/'config/camera_intrinsics.json').read_text())
frame='/SurgicalTools/PSM_Right/Robot/psm_tool_pitch_link'

def world(path):
    prim=stage.GetPrimAtPath(path)
    assert prim.IsValid(),path
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

# Authored robot mesh coordinates use meters; the robot root scales them into
# stage centimeters. Use the actual composed transform, then stage meters/unit.
model=np.array([known['A_model_mm'],known['B_model_mm']])/1000
xyz_world=(world(frame)@np.c_[model,np.ones(2)].T).T[:,:3]*unit
length_mm=float(np.linalg.norm(xyz_world[1]-xyz_world[0])*1000)
assert abs(length_mm-known['known_length_mm'])<1e-7

def project(xyz,cam):
    x,y=xyz[:,0]/xyz[:,2],xyz[:,1]/xyz[:,2]
    k1,k2,p1,p2,k3=cam['D']
    r=x*x+y*y
    radial=1+k1*r+k2*r*r+k3*r*r*r
    K=np.array(cam['K'])
    return np.c_[K[0,0]*(x*radial+2*p1*x*y+p2*(r+2*x*x))+K[0,2],
                 K[1,1]*(y*radial+p1*(r+2*y*y)+2*p2*x*y)+K[1,2]]

record={'source_scene':str(scene),'source_scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),
    'point_source':'Fresh USD transform of the same fixed model A/B points; distorted image projections, not image detections.',
    'model_definition':known,'stage_meters_per_unit':unit,'model_length_mm':length_mm,
    'world_points_m':xyz_world.tolist(),'points':{},'cameras':{}}
Tw={}
for eye in ['left','right']:
    camera='/Endoscope/ECM/Robot/ecm_tip_link/StereoPreviewRig/'+eye.capitalize()
    T=world(camera)@np.diag([1.,-1.,-1.,1.])
    assert np.allclose(np.linalg.norm(T[:3,:3],axis=0),1,atol=1e-8)
    T[:3,3]*=unit
    Tw[eye]=T
    xyz=(np.linalg.inv(T)@np.c_[xyz_world,np.ones(2)].T).T[:,:3]
    assert np.all(xyz[:,2]>0)
    record['points'][eye]=project(xyz,intrinsics[eye]).tolist()
    record['cameras'][eye]={'T_world_from_optical_m':T.tolist(),'points_optical_m':xyz.tolist()}

relative=np.linalg.inv(Tw['right'])@Tw['left']
translation=relative[:3,3]
baseline=float(np.linalg.norm(translation))
record['relative_pose']={'R_right_from_left':relative[:3,:3].tolist(),
    'unit_translation_right_from_left':(translation/baseline).tolist(),
    'translation_right_from_left_m':translation.tolist(),'baseline_mm':baseline*1000}
saved=json.loads((capture/'ground_truth/stereo_extrinsics.json').read_text())
record['consistency_with_saved_truth']={
    'rotation_max_abs_difference':float(np.max(abs(relative[:3,:3]-np.array(saved['R_right_from_left'])))),
    'translation_max_abs_difference_m':float(np.max(abs(translation-np.array(saved['t_right_from_left_m'])))),
    'baseline_difference_mm':float(baseline*1000-saved['baseline_m']*1000)}
assert record['consistency_with_saved_truth']['rotation_max_abs_difference']<1e-8
assert record['consistency_with_saved_truth']['translation_max_abs_difference_m']<1e-8
(out/'truth_from_scene.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'length_mm':length_mm,'points':record['points'],
    'relative_pose':record['relative_pose'],'consistency':record['consistency_with_saved_truth']},indent=2))
