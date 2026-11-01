from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False})
from pxr import Usd,UsdGeom
from pathlib import Path
import json
p=Path('/home/leorobot/orbit-surgical/source/extensions/orbit.surgical.assets/data/Robots/dVRK')
result={}
for model in ['PSM/psm.usd','PSM/psm_col.usd','ECM/ecm.usd']:
 s=Usd.Stage.Open(str(p/model));meshes=[x for x in s.Traverse() if x.IsA(UsdGeom.Mesh)]
 result[model]={'meshes':[{'path':str(x.GetPath()),'vertices':len(UsdGeom.Mesh(x).GetPointsAttr().Get() or []),'purpose':str(UsdGeom.Imageable(x).ComputePurpose()),'visibility':str(UsdGeom.Imageable(x).ComputeVisibility())} for x in meshes], 'cameras':[str(x.GetPath()) for x in s.Traverse() if x.IsA(UsdGeom.Camera)]}
s=Usd.Stage.Open('/home/leorobot/sufia-bc/scenes/OR_scene_CTLiver-Prostate-Bladder/organs_psm_ecm.usda')
result['lights']=[str(x.GetPath()) for x in s.Traverse() if 'Light' in x.GetTypeName()]
result['scene_roots']=[str(x.GetPath()) for x in s.Traverse() if str(x.GetPath()).count('/')<=3]
Path(__file__).with_name('asset_audit.json').write_text(json.dumps(result,indent=2))
print('AUDIT_COMPLETE',flush=True)
app.close()
