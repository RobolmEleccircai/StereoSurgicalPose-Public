"""Per-frame articulated truth, independent of rendering and preview placement."""
import numpy as np


def jaw_definitions(stage, psm_root):
    from pxr import UsdPhysics
    from kinematics import joint_frame, rigid_mm, usd_world
    definitions = {}
    for index in (1, 2):
        name = f'psm_tool_gripper{index}_joint'
        matches = [p for p in stage.Traverse() if p.GetName() == name
                   and str(p.GetPath()).startswith(psm_root+'/')]
        if len(matches) != 1 or not matches[0].IsA(UsdPhysics.RevoluteJoint):
            raise ValueError('Expected one authored jaw joint: '+name)
        prim = matches[0]
        item = {'joint_path': str(prim.GetPath()), 'joint_name': name,
                'axis': prim.GetAttribute('physics:axis').Get()}
        for side in (0, 1):
            targets = prim.GetRelationship(f'physics:body{side}').GetTargets()
            if len(targets) != 1:
                raise ValueError('Missing or ambiguous physical jaw body')
            body = str(targets[0]); raw = usd_world(stage, body)
            local = np.linalg.solve(rigid_mm(raw), rigid_mm(raw@joint_frame(prim, side)))
            item[f'T_body{side}_from_joint{side}_mm'] = local.tolist()
            item['jaw_body' if side else 'parent_body'] = body
            item[f'authored_localPos{side}'] = list(prim.GetAttribute(f'physics:localPos{side}').Get())
        if not item['jaw_body'].endswith(f'/psm_tool_gripper{index}_link'):
            raise ValueError('Incorrect physical jaw identity')
        definitions[f'jaw{index}'] = item
    return {'unit': 'mm', 'jaws': definitions}


def annotation_inputs(record, camera_truth):
    truth = record['body_pose_truth']
    if truth['unit'] != 'mm':
        raise ValueError('Body pose truth must use millimeters')
    return {'roll_body': truth['roll_body'],
            'T_world_from_body_mm': truth['T_world_from_body_mm'],
            'camera_truth': camera_truth}, truth['jaw_hinge_definitions']


def attach_body_truth(record, world_poses, roll_body, hinges, camera_truth, source_scene_sha256):
    """Save full rotations so roll is never guessed from a single shaft vector."""
    from tools.reannotate import articulated_axis_geometry
    record['body_pose_truth'] = {
        'unit': 'mm', 'source': 'authored_USD_joint_frames_and_frame_joint_state',
        'source_scene_sha256': source_scene_sha256, 'roll_body': roll_body,
        'T_world_from_body_mm': {k: np.asarray(v).tolist() for k, v in world_poses.items()},
        'jaw_hinge_definitions': hinges}
    pose_frame, definitions = annotation_inputs(record, camera_truth)
    geometry = articulated_axis_geometry(record, pose_frame, definitions)
    for eye, part in geometry.items():
        frames = [('shaft', 'C', part['frame'], part['T_optical_from_body_mm'])]
        frames += [(f"jaw{j['jaw_index']}", j['anchor'], j['frame'],
                    j['T_optical_from_hinge_frame_mm']) for j in part['jaw_frames']]
        record['eyes'][eye]['local_frames'] = {
            name: {'physical_origin': anchor, 'body': body,
                   'expressed_in': eye+'_optical', 'position_unit': 'mm',
                   'rotation_convention': 'columns are local +X,+Y,+Z in this optical frame',
                   'T_optical_from_local_mm': transform}
            for name, anchor, body, transform in frames}
    return geometry


def validate_body_truth(record, camera_truth):
    from tools.reannotate import articulated_axis_geometry
    pose_frame, hinges = annotation_inputs(record, camera_truth)
    geometry = articulated_axis_geometry(record, pose_frame, hinges)
    relative = np.asarray(camera_truth['T_right_from_left_mm'])
    for eye, part in geometry.items():
        expected = {'shaft': part['T_optical_from_body_mm'],
                    **{f"jaw{j['jaw_index']}": j['T_optical_from_hinge_frame_mm'] for j in part['jaw_frames']}}
        for name, transform in expected.items():
            saved = record['eyes'][eye]['local_frames'][name]['T_optical_from_local_mm']
            if not np.allclose(saved, transform, atol=1e-8, rtol=0):
                raise ValueError('Local frame disagrees with body pose: '+eye+'/'+name)
    for name in ('shaft', 'jaw1', 'jaw2'):
        left = np.asarray(record['eyes']['left']['local_frames'][name]['T_optical_from_local_mm'])
        right = np.asarray(record['eyes']['right']['local_frames'][name]['T_optical_from_local_mm'])
        if not np.allclose(relative@left, right, atol=1e-6, rtol=0):
            raise ValueError('Local frame stereo truth is inconsistent: '+name)
    return geometry
