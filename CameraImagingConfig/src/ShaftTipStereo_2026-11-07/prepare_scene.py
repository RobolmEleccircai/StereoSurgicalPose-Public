from pathlib import Path
import json
import numpy as np
from pxr import Usd, UsdGeom, Sdf, Gf
import landmarks

root=Path(__file__).resolve().parent
source=root.parent/'SingleInstrument_2026-11-07/preview/native/scene/sample_0000.usda'
scene=root/'scene.usda'
assert not scene.exists()
stage=Usd.Stage.CreateNew(str(scene));stage.GetRootLayer().subLayerPaths=[str(source)]
UsdGeom.SetStageMetersPerUnit(stage,.01);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.y)
definition=landmarks.define(stage)
profile=json.loads((root.parent/'SingleInstrument_2026-11-07/preview_final/config/camera_intrinsics.json').read_text())
C=np.diag([1.,-1.,-1.,1.]);rig='/Endoscope/ECM/Robot/ecm_tip_link/StereoPreviewRig/'
Tw={eye:landmarks.world(stage,rig+eye.capitalize())@C for eye in ['left','right']}
base=landmarks.world(stage,'/SurgicalTools/PSM_Right');pivot=landmarks.physical_tip(stage,definition)
parent=landmarks.world(stage,'/SurgicalTools');results=[]
for v in landmarks.VARIANTS:
    T=np.linalg.inv(parent)@landmarks.variant_transform(base,pivot,Tw['left'][:3,:3],v)
    xf=UsdGeom.Xformable(stage.GetPrimAtPath('/SurgicalTools/PSM_Right'));xf.ClearXformOpOrder()
    attr=xf.GetPrim().GetAttribute('xformOp:transform')
    op=UsdGeom.XformOp(attr) if attr.IsValid() else xf.AddTransformOp()
    op.Set(Gf.Matrix4d(T.T.tolist()));xf.SetXformOpOrder([op])
    labels=landmarks.labels(stage,definition,Tw,profile)
    results.append({'variation':v,'labels':labels})
    stage.GetRootLayer().Export(str(root/f'scene_{v["id"]}.usda'))
    print(v['id'],{eye:{key:labels[eye][key] for key in ['shaft_reference_pixel_raw','tip_pixel_raw','shaft_direction_unit']} for eye in labels},flush=True)
# Saved entry scene uses the original placement, with the new non-rendering frames.
stage.GetRootLayer().Clear()
stage.GetRootLayer().subLayerPaths=[str(root/'scene_0000.usda')]
stage.GetRootLayer().Save()
(root/'landmark_definitions.json').write_text(json.dumps(definition,indent=2))
(root/'preflight.json').write_text(json.dumps({'passed':True,'samples':results},indent=2))
