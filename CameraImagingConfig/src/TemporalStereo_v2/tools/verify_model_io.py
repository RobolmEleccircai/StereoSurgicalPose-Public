"""Export fixed instrument geometry and verify 17-value decoding; never render."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import load_machine, read, write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('machine', 'settings', 'run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    a = parser.parse_args(); machine = load_machine(a.machine)
    if a.output.resolve().is_relative_to(a.run.resolve()):
        raise ValueError('Write evidence outside the source run')
    if not a.worker:
        isaac = Path(machine['isaac_root']); libraries = list((isaac/'extscache').glob('omni.usd.libs-*'))
        if len(libraries) != 1: raise ValueError('Expected one USD runtime')
        env = os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): env[name] = '1'
        env['PYTHONPATH'] = os.pathsep.join([str(libraries[0]), machine['deps_root'], str(ROOT), env.get('PYTHONPATH', '')])
        env['LD_LIBRARY_PATH'] = os.pathsep.join([str(libraries[0]/'bin'), env.get('LD_LIBRARY_PATH', '')])
        subprocess.run([str(isaac/'python.sh'), str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], env=env, check=True)
        return

    import numpy as np
    from scipy.spatial.transform import Rotation
    from runtime import Scene
    from kinematics import usd_world, rigid_mm
    from pose_truth import jaw_definitions
    from kinematic_decoder import decode, project, OUTPUT_LAYOUT

    before = {str(p): sha(p) for p in (a.run/'sequences').rglob('*') if p.is_file()}
    scene = Scene(machine, read(a.settings))
    model = {'source': 'fixed_USD_model_only_no_frame_label_fitting',
             'source_scene_sha256': sha(scene.paths['source_scene']), 'unit': 'mm',
             'quaternion_order': 'xyzw', 'shaft_body': scene.roll, 'joints': [], 'jaws': {}}
    names = ['psm_tool_pitch_joint', 'psm_tool_yaw_joint', 'psm_tool_gripper1_joint', 'psm_tool_gripper2_joint']
    for name in names:
        j = scene.psm.by_name[name]
        parent_raw = usd_world(scene.stage, j['parent'])
        parent = rigid_mm(parent_raw); child = rigid_mm(usd_world(scene.stage, j['child']))
        model['joints'].append({'name': name, 'parent': j['parent'], 'child': j['child'], 'axis': j['axis'],
            'bind_angle_rad': j['q0'], 'lower_rad': float(j['lower']), 'upper_rad': float(j['upper']),
            'A_parent_joint_mm': np.linalg.solve(parent, rigid_mm(parent_raw @ j['J0'])).tolist(),
            'B_parent_child_bind_mm': np.linalg.solve(parent, child).tolist()})
    hinges = jaw_definitions(scene.stage, scene.psm.root)
    tips = dict(scene.refs)
    for name, j in hinges['jaws'].items():
        model['jaws'][name] = {'body': j['jaw_body'],
            'hinge_local_mm': np.asarray(j['T_body1_from_joint1_mm'])[:3, 3].tolist(),
            'tip_local_mm': tips[j['jaw_body']].tolist(), 'joint_path': j['joint_path']}
    # Freeze and reload constants before seeing any per-frame target values.
    write(a.output/'instrument_model.json', model); model = read(a.output/'instrument_model.json')
    maxima = {'position_mm': {k: 0. for k in ('C', 'H1', 'H2', 'J1', 'J2', 'T')},
              'direction_l2': 0., 'rotation_element_abs': 0., 'landmark_projection_px': 0.,
              'arrow_and_axis_projection_px': 0.}
    targets = []; records = []
    for path in sorted(a.run.glob('sequences/*/labels/*.json')):
        label = read(path)
        if 'eyes' not in label: continue
        if label['body_pose_truth']['source_scene_sha256'] != model['source_scene_sha256']:
            raise ValueError('Source model differs from the captured asset')
        values = []
        for eye in ('left', 'right'):
            e = label['eyes'][eye]; shaft = np.asarray(e['local_frames']['shaft']['T_optical_from_local_mm'])
            values.extend(e['tool']['shaft_reference_position_mm'])
            quaternion = Rotation.from_matrix(shaft[:3, :3]).as_quat()
            if quaternion[3] < 0: quaternion = -quaternion
            values.extend(quaternion.tolist())
        q = label['joint_positions']['psm']; half = q['psm_tool_gripper2_joint']
        if abs(q['psm_tool_gripper1_joint'] + half) > 1e-8:
            raise ValueError('17-value contract assumes symmetric coupled jaws')
        values.extend([q[names[0]], q[names[1]], half])
        decoded = decode(values, model)
        targets.append({'sequence_id': label['sequence_id'], 'job_id': label['job_id'], 'values': values})
        frame_max = 0.
        for eye, e in label['eyes'].items():
            out = decoded[eye]; tool = e['tool']
            matrices = {k: np.asarray(v['T_optical_from_local_mm']) for k, v in e['local_frames'].items()}
            gold = {'C': np.asarray(tool['shaft_reference_position_mm']), 'd': np.asarray(tool['shaft_direction_unit']),
                    'J1': np.asarray(tool['jaw1_tip_position_mm']), 'J2': np.asarray(tool['jaw2_tip_position_mm']),
                    'T': np.asarray(tool['tip_position_mm']), 'H1': matrices['jaw1'][:3, 3], 'H2': matrices['jaw2'][:3, 3],
                    'R_shaft': matrices['shaft'][:3, :3], 'R_jaw1': matrices['jaw1'][:3, :3], 'R_jaw2': matrices['jaw2'][:3, :3]}
            for key in maxima['position_mm']:
                error = float(np.linalg.norm(out[key] - gold[key])); frame_max = max(frame_max, error)
                maxima['position_mm'][key] = max(maxima['position_mm'][key], error)
            maxima['direction_l2'] = max(maxima['direction_l2'], float(np.linalg.norm(out['d']-gold['d'])))
            for key in ('R_shaft', 'R_jaw1', 'R_jaw2'):
                maxima['rotation_element_abs'] = max(maxima['rotation_element_abs'], float(abs(out[key]-gold[key]).max()))
            points = [out[k] for k in ('C', 'T', 'J1', 'J2')]
            pixels = np.asarray([tool['shaft_reference_pixel'], tool['tip_pixel'], *tool['jaw_tip_pixels']])
            maxima['landmark_projection_px'] = max(maxima['landmark_projection_px'], float(abs(project(points, e['K'], e['D'])-pixels).max()))
            def paths(g):
                lines = [(g['C'], g['C']+15*g['d']), (g['H1'], g['J1']), (g['H2'], g['J2'])]
                for origin, rotation in [('C', 'R_shaft'), ('H1', 'R_jaw1'), ('H2', 'R_jaw2')]:
                    lines.extend((g[origin], g[origin]+3*g[rotation][:, i]) for i in range(3))
                return np.vstack([np.linspace(x, y, 61) for x, y in lines])
            error = float(abs(project(paths(out), e['K'], e['D'])-project(paths(gold), e['K'], e['D'])).max())
            maxima['arrow_and_axis_projection_px'] = max(maxima['arrow_and_axis_projection_px'], error)
        records.append({'sequence_id': label['sequence_id'], 'job_id': label['job_id'], 'max_position_error_mm': frame_max})
    if not records: raise ValueError('No complete labeled frames found')
    if max(maxima['position_mm'].values()) >= 1e-4 or maxima['direction_l2'] >= 1e-6 or maxima['rotation_element_abs'] >= 1e-6 or max(maxima['landmark_projection_px'], maxima['arrow_and_axis_projection_px']) >= 1e-3:
        raise ValueError('Decoding differs from stored truth: '+json.dumps(maxima))
    if before != {str(p): sha(p) for p in (a.run/'sequences').rglob('*') if p.is_file()}:
        raise ValueError('Source dataset was modified')
    report = {'passed': True, 'pairs': len(records), 'eyes': 2*len(records), 'output_values_per_pair': 17,
              'output_layout': OUTPUT_LAYOUT, 'fixed_model_sha256': sha(a.output/'instrument_model.json'),
              'decoder_sha256': sha(Path(__file__).with_name('kinematic_decoder.py')),
              'input_to_decoder': '17 values plus fixed USD-derived instrument constants',
              'additional_projection_inputs': 'per-eye K and D only',
              'camera_extrinsics_or_baseline_used_by_decoder': False,
              'per_frame_jaw_world_or_camera_poses_used_by_decoder': False,
              'source_files_unchanged': len(before), 'new_rgb_renders': 0,
              'maximum_errors': maxima, 'records': records,
              'scope': 'Numerical sufficiency using ground-truth inputs; not a trained-network accuracy test.'}
    write(a.output/'report.json', report)
    write(a.output/'targets_17.json', {'layout': OUTPUT_LAYOUT, 'records': targets})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
