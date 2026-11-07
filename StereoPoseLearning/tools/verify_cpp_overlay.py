"""CPU-only native/Python geometry regression and warmed stereo overlay timing.

Uses saved predictions and original RGB, never a model or simulator. Samples JSON
must contain samples[{sequence_id,frame_index,prediction_values17}]. Manifest paths
must already resolve on this machine. Run outside the training process.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import subprocess
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def build_replay_fixture(config):
    """Saved model-output fixture for the video transport check, not inference."""
    from pose_display.pipeline import PosePrediction
    rows = json.loads((Path(config['_config_dir']) / config['rows']).read_text())
    def predict(frame, calibration):
        row = rows[int(frame.frame_id)]
        return PosePrediction(frame.frame_id, row['timestamp_ns'], row['values17'], source='prediction')
    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--font', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=140)
    a = p.parse_args()
    os.nice(19)
    sys.path.insert(0, str(a.reference or a.project / 'reference'))
    from pose_display.cpp_backend import CppStereoOverlay
    from pose_display.pipeline import StereoOverlay, StereoFrame, PosePrediction
    from pose_display.decoding import decode
    from pose_display.geometry import build_geometry
    from pose_display.projection import project
    a.output.mkdir(parents=True, exist_ok=True)
    model = json.loads((a.project / 'assets/instrument_model.json').read_text())
    manifest = json.loads(a.manifest.read_text())
    assert model['source_scene_sha256'] == manifest['source_scene_sha256']
    records = {(r['sequence_id'], r['frame_index']): r for r in manifest['records'] if r['split'] == 'test'}
    cases = []
    for row in json.loads(a.samples.read_text())['samples']:
        r = records[(row['sequence_id'], row['frame_index'])]
        label = json.loads(Path(r['label_path']).read_text())
        images = [np.asarray(Image.open(r['image_paths'][e]).convert('RGB')) for e in ('left', 'right')]
        cameras = {'image': {'width': images[0].shape[1], 'height': images[0].shape[0], 'space': 'raw_distorted'}}
        for e in ('left', 'right'): cameras[e] = {k: label['eyes'][e][k] for k in ('K', 'D')}
        native = CppStereoOverlay(model, cameras, a.font)
        legacy = StereoOverlay(model, cameras, a.font, compact=True)
        cases.append((row, images, cameras, native, legacy))
    digests = [hashlib.sha256(im.tobytes()).hexdigest() for _, images, *_ in cases for im in images]
    maximum = {'pose_mm_or_rotation': 0., 'projection_px': 0., 'display_length_mm': 0.}
    def compare(values, cameras, native):
        gt = decode(values, model); geometry = build_geometry(gt, cameras, model)
        actual = native.compute_geometry(values)
        for eye in ('left', 'right'):
            for key, expected in gt[eye].items():
                err = np.max(np.abs(expected - np.asarray(actual[eye][key])))
                maximum['pose_mm_or_rotation'] = max(maximum['pose_mm_or_rotation'], float(err))
                assert err < 1e-9, (eye, key, err)
            entries = [geometry[eye], *geometry[eye]['jaw_frames']]
            for old, new in zip(entries, actual[eye]['frames']):
                err = abs(old['length_mm'] - new['length_mm'])
                maximum['display_length_mm'] = max(maximum['display_length_mm'], err)
                assert err < 1e-9
                for ax, bx in zip(old['axes'], new['axes']):
                    err = np.max(np.abs(np.asarray(ax['projected_polyline_pixels']) - bx['projected_polyline_pixels']))
                    maximum['projection_px'] = max(maximum['projection_px'], float(err))
                    assert err < 1e-6
            g = gt[eye]
            for i, (start, end) in enumerate(((g['C'], g['C'] + 15*g['d']), (g['H1'], g['J1']), (g['H2'], g['J2']))):
                expected = project(np.linspace(start, end, 61), cameras[eye]['K'], cameras[eye]['D'])
                err = np.max(np.abs(expected - actual[eye]['main_curves'][i]))
                maximum['projection_px'] = max(maximum['projection_px'], float(err))
                assert err < 1e-6
    rng = np.random.default_rng(20261111)
    for i in range(140):
        row, _, cam, native, _ = cases[i % len(cases)]
        v = np.array(row['prediction_values17'])
        if i >= len(cases):
            for off in (0, 7):
                v[off:off+3] += rng.uniform(-1., 1., 3)
                v[off+3:off+7] = (Rotation.from_rotvec(rng.uniform(-.08, .08, 3)) * Rotation.from_quat(v[off+3:off+7])).as_quat()
            v[14:17] = [rng.uniform(-1.3, 1.3), rng.uniform(-.5, .5), rng.uniform(0, np.pi/6)]
        compare(v, cam, native)
    checks = ['140 saved/random pose and distorted XYZ geometry comparisons']
    row, images, cam, native, legacy = cases[0]
    values = np.array(row['prediction_values17'])
    for kind in ('nan', 'quaternion', 'limit', 'behind'):
        bad = values.copy()
        if kind == 'nan': bad[0] = np.nan
        if kind == 'quaternion': bad[3:7] = 0
        if kind == 'limit': bad[16] = 1.
        if kind == 'behind': bad[2] = -100.
        try: native.compute_geometry(bad)
        except ValueError: checks.append('rejected_' + kind)
        else: raise AssertionError(kind)
    frame = StereoFrame('ownership', 0, *images)
    pred = PosePrediction('ownership', 0, values)
    first, _ = native.render(frame, pred); before = first.copy()
    buffer = np.empty_like(first)
    second, _ = native.render(frame, pred, out=buffer)
    assert np.array_equal(first, before) and np.array_equal(first, second) and second is buffer
    try: native.render(frame, PosePrediction('stale', 0, values))
    except ValueError: checks.append('rejected_stale_frame')
    else: raise AssertionError('stale frame')
    # Distinguish visible, tissue occlusion, self occlusion and unknown.
    native.geometry_log = True
    states = [None, {'visible': True, 'occluder_kind': 'none'},
              {'visible': False, 'occluder_kind': 'tissue'}, {'visible': False, 'occluder_kind': 'instrument'}]
    for state in states:
        vis = None if state is None else {e: [state, state] for e in ('left', 'right')}
        p = PosePrediction('ownership', 0, values, visibility=vis, visibility_source='unknown' if vis is None else 'test')
        _, meta = native.render(frame, p)
        expected = -1 if state is None else 1 if state['visible'] else 2 if state['occluder_kind'] == 'tissue' else 3
        assert all(c in (0, expected) for e in meta['projection_results'].values() for c in e['visibility_codes'])
    native.geometry_log = False
    checks.extend(['independent output ownership', 'reusable output matches', 'visibility semantics', 'original RGB unchanged'])
    # A standalone C++ executable and the actual video entry point use this backend.
    # These tiny repeated-image fixtures check transport, never dataset quality/FPS.
    fixture = a.output / 'integration'; fixture.mkdir(exist_ok=True)
    (fixture / 'calibration.json').write_text(json.dumps(cam))
    (fixture / 'configuration.json').write_text(json.dumps({'model': model, 'calibration': cam}))
    (fixture / 'prediction.json').write_text(json.dumps({'values17': values.tolist()}))
    rec = records[(row['sequence_id'], row['frame_index'])]
    reference = a.reference or a.project / 'reference'
    subprocess.run([str(reference / 'pose_display/native/build/overlay_pair'),
        str(fixture / 'configuration.json'), a.font, str(fixture / 'prediction.json'),
        rec['image_paths']['left'], rec['image_paths']['right'], str(fixture / 'standalone.png')], check=True)
    import cv2
    cv2.setNumThreads(1)
    for eye, im in zip(('left', 'right'), images):
        writer = cv2.VideoWriter(str(fixture / (eye + '.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), 60, (im.shape[1], im.shape[0]))
        assert writer.isOpened()
        for _ in range(4): writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        writer.release()
    rows = [{'timestamp_ns': round(i*1e9/60), 'values17': values.tolist()} for i in range(4)]
    (fixture / 'rows.json').write_text(json.dumps(rows))
    (fixture / 'predictor.json').write_text(json.dumps({'rows': 'rows.json'}))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(reference), str(Path(__file__).parent)]))
    command = [sys.executable, '-m', 'pose_display.video', '--backend', 'cpp',
        '--left', str(fixture / 'left.mp4'), '--right', str(fixture / 'right.mp4'),
        '--calibration', str(fixture / 'calibration.json'), '--instrument-model', str(a.project / 'assets/instrument_model.json'),
        '--font', a.font, '--predictor', 'verify_cpp_overlay:build_replay_fixture',
        '--predictor-config', str(fixture / 'predictor.json')]
    subprocess.run(command + ['--output', str(fixture / 'success')], check=True, env=env, stdout=subprocess.DEVNULL)
    report = json.loads((fixture / 'success/report.json').read_text()); assert report['passed'] and report['frames'] == 4
    rows[-1]['timestamp_ns'] += 1
    (fixture / 'rows.json').write_text(json.dumps(rows))
    failed = subprocess.run(command + ['--output', str(fixture / 'stale_frame')], env=env, stdout=subprocess.DEVNULL)
    assert failed.returncode == 2
    report = json.loads((fixture / 'stale_frame/report.json').read_text()); assert not report['passed'] and len(report['failed_frames']) == 1
    checks.extend(['standalone C++ executable', 'four-frame native video callback and encode', 'stale video frame displays error and reports failure'])
    def run(i, cpp, reusable=True):
        row, images, _, native, legacy = cases[i % len(cases)]
        fid = row['sequence_id'] + f'/{i:06d}'; stamp = round(i/60*1e9)
        frame = StereoFrame(fid, stamp, *images); prediction = PosePrediction(fid, stamp, row['prediction_values17'])
        return native.render(frame, prediction, out=buffer if reusable else None) if cpp else legacy.render(frame, prediction)
    # Fresh IDs/timestamps prevent benchmarking a single cached title.
    for i in range(35): run(i, True); run(i, False)
    times = {'cpp_reused_buffer': [], 'python_compact': [], 'cpp_owned_output': []}
    for i in range(a.iterations):
        order = [('cpp_reused_buffer', True, True), ('python_compact', False, False), ('cpp_owned_output', True, False)]
        if i % 2: order.reverse()
        for name, cpp, reuse in order:
            t = time.perf_counter(); run(i+1000, cpp, reuse); times[name].append((time.perf_counter()-t)*1000)
    for i, (row, *_) in enumerate(cases):
        canvas, _ = run(i, True)
        Image.fromarray(canvas).save(a.output / (row['sequence_id'] + '_cpp.jpg'), quality=95)
        if i == 0:
            old, _ = run(i, False)
            Image.fromarray(old).save(a.output / 'python_reference.jpg', quality=95)
    assert digests == [hashlib.sha256(im.tobytes()).hexdigest() for _, ims, *_ in cases for im in ims]
    result = {'passed': True, 'max_errors': maximum, 'checks': checks,
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                    'numpy': np.__version__, 'cpp_abi': 1},
        'source_sha256': {str(p.relative_to(reference)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [reference/'pose_display/native/overlay.cpp', reference/'pose_display/native/overlay.h',
                      reference/'pose_display/cpp_backend.py', reference/'pose_display/video.py']},
        'samples': len(cases), 'iterations_per_backend': a.iterations,
        'scope': '1280x720 per eye; paired CPU FK, distorted projection and compact drawing; warm cache, changing timestamps. No inference, I/O, video encoding or display. GPU training continues.',
        'rendering_difference': 'OpenCV antialiasing, FreeType glyph rasterization and label text bounds differ from Pillow; geometric projections and panel dimensions are preserved.',
        'timings': {k: {'mean_ms': float(np.mean(v)), 'median_ms': float(np.median(v)), 'p95_ms': float(np.percentile(v,95))} for k,v in times.items()}}
    (a.output / 'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    for *_, native, _ in cases: native.close()


if __name__ == '__main__':
    main()
