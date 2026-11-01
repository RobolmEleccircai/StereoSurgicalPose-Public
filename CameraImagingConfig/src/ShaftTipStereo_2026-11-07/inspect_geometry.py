from pathlib import Path
import json
import numpy as np
from pxr import Usd, UsdGeom

root=Path(__file__).resolve().parent
scene=root.parent/'SingleInstrument_2026-11-07/preview/native/scene/sample_0000.usda'
stage=Usd.Stage.Open(str(scene),Usd.Stage.LoadNone)
robot='/SurgicalTools/PSM_Right/Robot'
def world(p):
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
report={}
arrays={}
for link in ['psm_tool_roll_link','psm_tool_pitch_link','psm_tool_yaw_link','psm_tool_gripper1_link','psm_tool_gripper2_link','psm_tool_tip_link']:
    prim=stage.GetPrimAtPath(robot+'/'+link)
    report[link]={'world':world(str(prim.GetPath())).tolist(),'meshes':[]}
    for p in Usd.PrimRange(prim):
        if not p.IsA(UsdGeom.Mesh): continue
        mesh=UsdGeom.Mesh(p);q=np.unique(np.asarray(mesh.GetPointsAttr().Get(),float),axis=0)
        rel=np.linalg.inv(world(str(prim.GetPath())))@world(str(p.GetPath()))
        coords=(rel@np.c_[q,np.ones(len(q))].T).T[:,:3]
        rec={'path':str(p.GetPath()),'visibility':str(UsdGeom.Imageable(p).ComputeVisibility()),'count':len(q),'bbox_link':np.array([coords.min(0),coords.max(0)]).tolist()}
        report[link]['meshes'].append(rec)
        arrays[link+'__'+p.GetName()]=coords
        if link=='psm_tool_roll_link' and rec['visibility']!='invisible':
            zvals,cnt=np.unique(np.round(coords[:,2],6),return_counts=True)
            rec['z_slices']=[[float(z),int(n)] for z,n in zip(zvals,cnt) if n>=12]
(root/'geometry_inspection.json').write_text(json.dumps(report,indent=2))
np.savez(root/'geometry_inspection.npz',**arrays)
print(json.dumps(report,indent=2))
