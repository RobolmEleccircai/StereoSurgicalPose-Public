"""Derive the candidate closed-jaw tip-to-rim reference length from USD mesh data.

The NPZ was exported from the current USD asset. These are CAD-defined reference
points, not a claim that an image silhouette midpoint is exactly their projection.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parent
d=np.load(root/'model_geometry.npz')
T=d['psm_tool_pitch_link_T']
tips=[]
for name in ['psm_tool_gripper1_link__gripper_right','psm_tool_gripper2_link__gripper_left']:
    q=np.unique(d[name+'_vertices'].astype(float),axis=0)
    distal=q[abs(q[:,1]-q[:,1].max())<1e-9]
    relative=np.linalg.inv(T)@d[name+'_T']
    tips.append((relative@np.r_[distal.mean(0),1])[:3]*1000)
A=np.mean(tips,axis=0)
q=np.unique(d['psm_tool_pitch_link__tool_pitch_link_vertices'].astype(float)*1000,axis=0)
ring=q[(abs(q[:,0]-9.1)<1.3)&(abs(q[:,2])<1.3)&(abs(q[:,1]+2.591)<.0005)]
assert len(ring)==24
xy=ring[:,[0,2]]
c=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],(xy**2).sum(1),rcond=None)[0]
radius=np.sqrt(c[2]+sum(c[:2]**2))
B=np.array([c[0],ring[:,1].mean(),c[1]])
record={'units':'mm','coordinate_frame':'psm_tool_pitch_link, canonical metric mesh coordinates',
    'A_definition':'Midpoint of distal-most visible-mesh end centers of the two closed jaw meshes; not the nominal tool control frame. Image estimate uses end-edge midpoint.',
    'B_definition':'Center of 24-vertex circular rim in the visible negative-Y side of tool_pitch_link, plane Y=-2.591 mm, radius derived from mesh.',
    'A_model_mm':A.tolist(),'B_model_mm':B.tolist(),'known_length_mm':float(np.linalg.norm(B-A)),
    'rim_radius_mm':float(radius),'rim_max_circle_residual_mm':float(abs(np.linalg.norm(xy-c[:2],axis=1)-radius).max()),
    'mesh_sample_count':len(ring),'source_file':'/home/leorobot/orbit-surgical/source/extensions/orbit.surgical.assets/data/Robots/dVRK/PSM/psm_col.usd',
    'geometry_export_sha256':hashlib.sha256((root/'model_geometry.npz').read_bytes()).hexdigest(),
    'scope':'Metric length computed only from model geometry and current closed-jaw relative transforms. No baseline is used.'}
(root/'known_AB.json').write_text(json.dumps(record,indent=2))
print(record['known_length_mm'])
