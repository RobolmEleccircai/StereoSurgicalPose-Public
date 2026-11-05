"""Explicit saved-truth replay adapter, separate from the prediction API."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .pipeline import StereoFrame, PosePrediction


def values_from_label(label):
    result = []
    for eye in ('left', 'right'):
        e = label['eyes'][eye]
        result.extend(e['tool']['shaft_reference_position_mm'])
        rotation = np.asarray(e['local_frames']['shaft']['T_optical_from_local_mm'])[:3, :3]
        quaternion = Rotation.from_matrix(rotation).as_quat()
        if quaternion[3] < 0: quaternion = -quaternion
        result.extend(quaternion.tolist())
    q = label['joint_positions']['psm']
    half = q['psm_tool_gripper2_joint']
    if abs(q['psm_tool_gripper1_joint']+half) > 1e-8: raise ValueError('Unsupported asymmetric jaws')
    return result+[q['psm_tool_pitch_joint'], q['psm_tool_yaw_joint'], half]


def load_pair(label_path):
    label_path = Path(label_path); label = json.loads(label_path.read_text()); images = {}
    for eye, data in label['eyes'].items():
        path = label_path.parent/data['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != data['image_sha256']:
            raise ValueError('Source RGB hash mismatch')
        with Image.open(path) as image: images[eye] = np.asarray(image.convert('RGB'))
    frame = StereoFrame(label['sequence_id']+'/'+label['job_id'], label['timestamp_ns'], images['left'], images['right'])
    prediction = PosePrediction(frame.frame_id, frame.timestamp_ns, values_from_label(label), source='ground_truth_replay')
    calibration = {'image': {'width': images['left'].shape[1], 'height': images['left'].shape[0], 'space': 'raw_distorted'},
                   **{eye: {key: e[key] for key in ('K', 'D')} for eye, e in label['eyes'].items()}}
    return frame, prediction, calibration, label
