"""Save and validate the user-requested single-instrument USD composition."""
import ast
import hashlib
import json
import shutil
from pathlib import Path
from pxr import Usd, UsdGeom

home = Path('/home/leorobot')
project = home / 'PoseProject/CameraImagingConfig'
refinement = project / 'CameraRefinement_2026-11-06'
out = project / 'SingleInstrument_2026-11-07'
out.mkdir(exist_ok=True)
(out / 'backups').mkdir(exist_ok=True)
base = refinement / 'run05_initial/native/scene/sample_0000.usda'
source = home / 'sufia-bc/scenes/OR_scene_CTLiver-Prostate-Bladder/organs_psm_ecm.usda'
scene = out / 'scene.usda'
left_tool = '/SurgicalTools/PSM_Left'
right_tool = '/SurgicalTools/PSM_Right'
ecm = '/Endoscope/ECM'
rig = ecm + '/Robot/ecm_tip_link/StereoPreviewRig'
camera_paths = [rig + '/Left', rig + '/Right']
protected = [base, source, refinement / 'CONFIRMED_BASELINE.json',
             project / 'calibration/profiles/drends_exvivo_1080.json']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def transform(stage, path):
    matrix = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return [list(row) for row in matrix]

def camera_state(stage):
    state = {}
    for path in camera_paths:
        prim = stage.GetPrimAtPath(path)
        assert prim and prim.IsA(UsdGeom.Camera)
        state[path] = {'transform': transform(stage, path),
                       'attributes': {a.GetName(): str(a.Get()) for a in prim.GetAttributes()}}
    return state

before_hashes = {str(p): sha(p) for p in protected}
original = Usd.Stage.Open(str(base))
assert original
before_cameras = camera_state(original)
before_right = transform(original, right_tool)
before_ecm = transform(original, ecm)
assert not scene.exists(), 'Do not overwrite a previously saved scene'
stage = Usd.Stage.CreateNew(str(scene))
stage.GetRootLayer().subLayerPaths = [str(base)]
UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
stage.OverridePrim(left_tool).SetActive(False)
stage.GetRootLayer().customLayerData = {
    'configuration': 'single_psm_instrument_stereo_ecm',
    'activeInstrument': right_tool,
    'instrumentCount': 1,
    'stereoCameraCount': 2,
    'sourceSample': str(base),
}
stage.GetRootLayer().Save()
del stage
stage = Usd.Stage.Open(str(scene))
assert not stage.GetPrimAtPath(left_tool).IsActive()
assert stage.GetPrimAtPath(right_tool).IsActive()
assert stage.GetPrimAtPath(ecm).IsActive()
assert camera_state(stage) == before_cameras
assert transform(stage, right_tool) == before_right
assert transform(stage, ecm) == before_ecm
active_tools = [str(p.GetPath()) for p in stage.GetPrimAtPath('/SurgicalTools').GetChildren()
                if p.IsActive() and p.GetName().startswith('PSM_')]
assert active_tools == [right_tool], active_tools
assert not any(str(p.GetPath()).startswith(left_tool + '/') for p in stage.Traverse())

# Apply the reviewed capture update, preserving the previous source for replay.
capture = refinement / 'capture_initial.py'
assert sha(capture) == 'c52c818d68f5892702947420eb216300207a3ea830884bcaaa35d459b5d7f8aa'
new_capture = out / 'capture_initial.updated.py'
assert sha(new_capture) == 'ea95d3672439b099486f1e7e0cbf71003a8c1534872007bf75bac25b366f2042'
ast.parse(new_capture.read_text())
shutil.copy2(capture, out / 'backups/capture_initial.py')
shutil.copy2(new_capture, capture)

# The existing scene viewer now opens the saved single-instrument composition.
viewer = home / 'sufia-bc/view_ecm_scene.py'
old = "path=Path.home()/'sufia-bc/scenes/OR_scene_CTLiver-Prostate-Bladder/organs_psm_ecm.usda'"
new = "path=Path.home()/'PoseProject/CameraImagingConfig/SingleInstrument_2026-11-07/scene.usda'"
text = viewer.read_text()
assert text.count(old) == 1
text = text.replace(old, new)
ast.parse(text)
shutil.copy2(viewer, out / 'backups/view_ecm_scene.py')
viewer.write_text(text)

assert before_hashes == {str(p): sha(p) for p in protected}
result = {
    'scene': str(scene),
    'active_instrument_count': len(active_tools),
    'active_instrument_paths': active_tools,
    'inactive_instrument_path': left_tool,
    'ecm_retained': True,
    'stereo_camera_count': len(camera_paths),
    'camera_transforms_and_attributes_unchanged': True,
    'right_tool_pose_unchanged': True,
    'ecm_pose_unchanged': True,
    'capture_script': str(capture),
    'capture_labels': ['right_tool'],
    'scene_viewer': str(viewer),
    'accepted_appearance_profile': str(refinement / 'CONFIRMED_BASELINE.json'),
    'protected_file_hashes': before_hashes,
    'new_images_rendered': False,
    'viewer_launched': False,
    'verified': True,
}
(out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
