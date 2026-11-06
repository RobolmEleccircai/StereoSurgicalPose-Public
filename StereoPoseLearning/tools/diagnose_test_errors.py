"""Audit saved test predictions using independent NumPy FK and oracle substitutions.

No inference, training or rendering. Oracle differences are not additive attributions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stereo_pose.data import _pose_truth
from reference.pose_display.decoding import decode


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    report = json.loads(Path(args.report).read_text())
    model = json.loads((ROOT / 'assets/instrument_model.json').read_text())
    records = {(r['sequence_id'], r['frame_index']): r for r in manifest['records']}
    variants = {'prediction': [], 'C_truth': [0, 1, 2, 7, 8, 9],
                'shaft_rotation_truth': [3, 4, 5, 6, 10, 11, 12, 13],
                'wrist_pitch_truth': [14], 'wrist_yaw_truth': [15],
                'jaw_half_angle_truth': [16], 'both_wrist_angles_truth': [14, 15],
                'all_joint_angles_truth': [14, 15, 16],
                'shaft_pose_truth': list(range(14)), 'all_truth': list(range(17))}
    rows = []
    for row in report['frames_detail']:
        record = records[(row['sequence_id'], row['frame_index'])]
        assert record['split'] == 'test'
        truth = _pose_truth(json.loads(Path(record['label_path']).read_text()))
        gt = np.r_[truth['C'][0], Rotation.from_matrix(truth['R'][0]).as_quat(),
                   truth['C'][1], Rotation.from_matrix(truth['R'][1]).as_quat(), truth['joints']]
        pred = np.asarray(row['prediction_values17'])
        errors = {}
        for name, indices in variants.items():
            values = pred.copy()
            values[indices] = gt[indices]
            pose = decode(values, model)
            tips = np.array([[pose[e]['J1'], pose[e]['J2']] for e in ('left', 'right')])
            errors[name] = float(np.linalg.norm(tips - truth['points'][:, 3:5], axis=-1).mean())
        assert abs(errors['prediction'] - row['tip_mm']) < 1e-3, 'Saved metric/FK mismatch'
        assert errors['all_truth'] < 1e-4, 'Ground truth FK mismatch'
        rows.append({'sequence_id': row['sequence_id'], 'frame_index': row['frame_index'],
                     'tip_mm_with_oracle_replacement': errors,
                     'predicted_angles_deg': np.rad2deg(pred[14:]).tolist(),
                     'truth_angles_deg': np.rad2deg(gt[14:]).tolist(),
                     'C_error_xyz_mm_left_right': (pred[[0,1,2,7,8,9]].reshape(2,3)-truth['C']).tolist(),
                     'position_mm': row['position_mm'], 'shaft_rotation_deg': row['rotation_deg'],
                     'shaft_axis_deg': row['axis_deg']})
    summary = {}
    for seq in sorted({r['sequence_id'] for r in rows}):
        subset = [r for r in rows if r['sequence_id'] == seq]
        pred = np.array([r['predicted_angles_deg'] for r in subset])
        gt = np.array([r['truth_angles_deg'] for r in subset])
        summary[seq] = {'frames': len(subset),
            'angle_order': ['wrist_pitch', 'wrist_yaw', 'jaw_half_angle'],
            'angles_mae_deg': np.abs(pred-gt).mean(0).tolist(),
            'angles_bias_deg': (pred-gt).mean(0).tolist(),
            'truth_angles_range_deg': np.stack([gt.min(0), gt.max(0)]).tolist(),
            'prediction_angles_range_deg': np.stack([pred.min(0), pred.max(0)]).tolist(),
            'C_bias_xyz_mm_left_right': np.mean([r['C_error_xyz_mm_left_right'] for r in subset],0).tolist(),
            **{k: float(np.mean([r[k] for r in subset])) for k in ['position_mm','shaft_rotation_deg','shaft_axis_deg']},
            'tip_mm_with_oracle_replacement': {k: float(np.mean([r['tip_mm_with_oracle_replacement'][k] for r in subset])) for k in variants}}
    train = np.rad2deg([r['joint_angles_rad'] for r in manifest['records'] if r['split'] == 'train'])
    result = {'method': 'Saved epoch predictions; independent NumPy FK; replace only named components with ground truth. Effects are coupled and not additive.',
              'report_diagnostic_checkpoint_flag': report.get('diagnostic_checkpoint'),
              'train_angles_range_deg': np.stack([train.min(0),train.max(0)]).tolist(),
              'summary': summary, 'frames': rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
