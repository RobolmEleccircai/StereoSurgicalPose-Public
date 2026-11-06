"""Decode a 17-value articulated pose using a fixed USD-derived instrument model.

No image labels, world poses, camera extrinsics or baseline enter this function.
Quaternion order is x,y,z,w; translations are mm; joint angles are radians.
"""
import numpy as np
from scipy.spatial.transform import Rotation

OUTPUT_LAYOUT = [
    ('C_left_mm', 3), ('shaft_quaternion_left_xyzw', 4),
    ('C_right_mm', 3), ('shaft_quaternion_right_xyzw', 4),
    ('wrist_pitch_rad', 1), ('wrist_yaw_rad', 1), ('jaw_half_angle_rad', 1),
]


def decode(values, model):
    values = np.asarray(values, dtype=float)
    if values.shape != (17,) or not np.isfinite(values).all():
        raise ValueError('Expected 17 finite pose values')
    pitch, yaw, half = values[14:17]
    q = {'psm_tool_pitch_joint': pitch, 'psm_tool_yaw_joint': yaw,
         'psm_tool_gripper1_joint': -half, 'psm_tool_gripper2_joint': half}
    for joint in model['joints']:
        if not joint['lower_rad'] - 1e-8 <= q[joint['name']] <= joint['upper_rad'] + 1e-8:
            raise ValueError('Predicted joint exceeds model limits: ' + joint['name'])
    result = {}
    for eye, offset in [('left', 0), ('right', 7)]:
        quaternion = values[offset+3:offset+7]
        if np.linalg.norm(quaternion) < 1e-12:
            raise ValueError('A zero quaternion does not define a rotation')
        shaft = np.eye(4)
        shaft[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        shaft[:3, 3] = values[offset:offset+3]
        poses = {model['shaft_body']: shaft}
        for joint in model['joints']:
            anchor = np.asarray(joint['A_parent_joint_mm'])
            bind = np.asarray(joint['B_parent_child_bind_mm'])
            motion = np.eye(4)
            axis = np.eye(3)['XYZ'.index(joint['axis'])]
            motion[:3, :3] = Rotation.from_rotvec(axis * (q[joint['name']] - joint['bind_angle_rad'])).as_matrix()
            poses[joint['child']] = poses[joint['parent']] @ anchor @ motion @ np.linalg.inv(anchor) @ bind
        out = {'C': shaft[:3, 3], 'd': shaft[:3, 2], 'R_shaft': shaft[:3, :3]}
        for index in (1, 2):
            jaw = model['jaws'][f'jaw{index}']; pose = poses[jaw['body']]
            out[f'H{index}'] = (pose @ np.r_[jaw['hinge_local_mm'], 1.])[:3]
            out[f'J{index}'] = (pose @ np.r_[jaw['tip_local_mm'], 1.])[:3]
            out[f'R_jaw{index}'] = pose[:3, :3]
        out['T'] = (out['J1'] + out['J2']) / 2
        result[eye] = out
    return result
