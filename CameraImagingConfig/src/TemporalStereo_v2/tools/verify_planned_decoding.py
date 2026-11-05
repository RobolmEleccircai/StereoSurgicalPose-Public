"""Check decoded articulated poses over every planned state, without rendering."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import load_machine, read, write, sha, require_seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('machine', 'settings', 'run', 'instrument-model', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    machine = load_machine(args.machine)
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Write evidence outside the source run')
    if not args.worker:
        isaac = Path(machine['isaac_root'])
        libraries = list((isaac/'extscache').glob('omni.usd.libs-*'))
        if len(libraries) != 1:
            raise ValueError('Expected one USD runtime')
        env = os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[name] = '1'
        env['PYTHONPATH'] = os.pathsep.join([str(libraries[0]), machine['deps_root'], str(ROOT), env.get('PYTHONPATH', '')])
        env['LD_LIBRARY_PATH'] = os.pathsep.join([str(libraries[0]/'bin'), env.get('LD_LIBRARY_PATH', '')])
        subprocess.run([str(isaac/'python.sh'), str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], env=env, check=True)
        return

    import numpy as np
    from scipy.spatial.transform import Rotation
    from runtime import Scene
    from pose_display.decoding import decode, OUTPUT_LAYOUT
    from pose_display.projection import project

    settings = read(args.settings)
    scene = Scene(machine, settings)
    model = read(args.instrument_model)
    if model['source_scene_sha256'] != sha(scene.paths['source_scene']):
        raise ValueError('Fixed instrument model differs from planned asset')
    manifest = read(args.run/'config/manifest.json')
    index = read(args.run/'trajectories/index.json')
    require_seal(args.run/'trajectories/index.json', manifest['fingerprint'])
    if settings != manifest['settings']:
        raise ValueError('Settings differ from the checked plan')
    before = {str(p): sha(p) for p in (args.run/'trajectories').rglob('*') if p.is_file()}
    maxima = {'position_mm': 0., 'direction_l2': 0., 'rotation_element_abs': 0.,
              'projection_px': 0., 'stereo_position_consistency_mm': 0.}
    counts = {}; per_sequence = []; all_angles = []
    for sid in index['sequences']:
        path = args.run/'trajectories'/f'{sid}.json'
        if sha(path) != index['trajectory_hashes'][sid]:
            raise ValueError('Changed planned trajectory')
        require_seal(path, manifest['fingerprint'])
        seq = read(path)
        for frame in seq['frames']:
            qp = np.asarray(frame['q_psm']); qe = np.asarray(frame['q_ecm'])
            world = scene.psm.fk(qp)
            cameras = scene.cameras(qe, seq)
            ground = scene.labels(qp, qe, seq)
            joints = scene.psm.positions(qp)
            half = joints['psm_tool_gripper2_joint']
            if abs(joints['psm_tool_gripper1_joint']+half) > 1e-10:
                raise ValueError('Asymmetric jaws violate the 17-value contract')
            values = []
            for eye in ('left', 'right'):
                shaft = np.linalg.solve(cameras[eye], world[scene.roll])
                values.extend(shaft[:3, 3])
                values.extend(Rotation.from_matrix(shaft[:3, :3]).as_quat())
            values.extend([joints['psm_tool_pitch_joint'], joints['psm_tool_yaw_joint'], half])
            decoded = decode(values, model)
            all_angles.append(float(np.rad2deg(half)))
            for eye, actual in decoded.items():
                tc = np.linalg.inv(cameras[eye]); g = ground[eye]
                gold = {'C': g['shaft_reference_position_mm'], 'J1': g['jaw1_tip_position_mm'],
                        'J2': g['jaw2_tip_position_mm'], 'T': g['tip_position_mm']}
                rotations = {'R_shaft': (tc@world[scene.roll])[:3, :3]}
                for number in (1, 2):
                    jaw = model['jaws'][f'jaw{number}']
                    pose = tc@world[jaw['body']]
                    gold[f'H{number}'] = (pose@np.r_[jaw['hinge_local_mm'], 1])[:3]
                    rotations[f'R_jaw{number}'] = pose[:3, :3]
                maxima['position_mm'] = max(maxima['position_mm'], *(float(np.linalg.norm(actual[k]-v)) for k,v in gold.items()))
                maxima['direction_l2'] = max(maxima['direction_l2'], float(np.linalg.norm(actual['d']-g['shaft_direction_unit'])))
                maxima['rotation_element_abs'] = max(maxima['rotation_element_abs'], *(float(abs(actual[k]-v).max()) for k,v in rotations.items()))
                pixels = [g['shaft_reference_pixel'], g['tip_pixel'], *g['jaw_tip_pixels']]
                result = project([actual[k] for k in ('C','T','J1','J2')], scene.profile[eye]['K'], scene.profile[eye]['D'])
                maxima['projection_px'] = max(maxima['projection_px'], float(abs(result-pixels).max()))
            # Camera truth is used only to check the separately decoded results.
            relative = np.linalg.solve(cameras['right'], cameras['left'])
            for k in ('C','H1','H2','J1','J2','T'):
                expected = (relative@np.r_[decoded['left'][k], 1])[:3]
                maxima['stereo_position_consistency_mm'] = max(maxima['stereo_position_consistency_mm'], float(np.linalg.norm(expected-decoded['right'][k])))
        counts[seq['split']] = counts.get(seq['split'], 0)+len(seq['frames'])
        per_sequence.append({'sequence_id':sid, 'split':seq['split'], 'category':seq['category'], 'checked_states':len(seq['frames'])})
    if not per_sequence:
        raise ValueError('No planned states')
    if any(maxima[k] >= limit for k,limit in {'position_mm':1e-4, 'direction_l2':1e-6,
            'rotation_element_abs':1e-6, 'projection_px':1e-3, 'stereo_position_consistency_mm':1e-6}.items()):
        raise ValueError('Decoded planned state differs from full-chain truth: '+str(maxima))
    if before != {str(p):sha(p) for p in (args.run/'trajectories').rglob('*') if p.is_file()}:
        raise ValueError('Planned paths changed during verification')
    if any(name.startswith(('isaacsim','omni')) for name in sys.modules):
        raise ValueError('Verification unexpectedly loaded a simulator application')
    write(args.output, {'passed':True, 'planned_states':sum(counts.values()), 'counts':counts,
        'sequences':per_sequence, 'eyes':2*sum(counts.values()), 'maximum_errors':maxima,
        'jaw_half_angle_range_deg':[min(all_angles),max(all_angles)], 'output_layout':OUTPUT_LAYOUT,
        'model_sha256':sha(args.instrument_model), 'verification_code_sha256':sha(Path(__file__)),
        'source_trajectory_files_unchanged':len(before), 'new_rgb_renders':0,
        'scope':'Decoder vs full USD kinematic chain over all planned states. No image, visibility or trained-model accuracy claim.',
        'camera_extrinsics_or_baseline_used_by_decoder':False,
        'camera_truth_used_only_for_independent_stereo_consistency_check':True})
    print('PLANNED_DECODING_PASSED',sum(counts.values()),maxima,flush=True)


if __name__ == '__main__':
    main()
