"""Check shared overlays and video callback using existing RGB only; no simulator."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import PIL
import scipy
from PIL import Image
from io_utils import read, write, sha, annotation_code_hash
from live_preview import LivePreview
from pose_display.adapters import load_pair
from pose_display.decoding import decode
from pose_display.pipeline import StereoOverlay, PosePrediction
from pose_display.projection import project
from pose_display.renderer import XYZ_PANEL_BG_RGBA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'instrument-model', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--font'); args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()): raise ValueError('Do not write inside the source run')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()): raise ValueError('Use a new empty output directory')
    try:
        before = {str(p.relative_to(args.run)): sha(p) for p in args.run.rglob('*') if p.is_file()}
        paths = sorted(args.run.glob('sequences/*/labels/*.json')); saved_frames = []
        model = read(args.instrument_model); maxima = {'position_mm': 0., 'projection_px': 0., 'rotation_element_abs': 0.}
        unchanged_previews = 0; frame = prediction = calibration = None
        for path in paths:
            frame, prediction, calibration, label = load_pair(path)
            saved_frames.append((frame, prediction))
            if label['body_pose_truth']['source_scene_sha256'] != model['source_scene_sha256']:
                raise ValueError('Model asset differs from source labels')
            overlay = StereoOverlay(model, calibration, args.font)
            preview = LivePreview(args.output/'legacy_previews', overlay.font)
            actual = preview.publish_saved(path)
            expected = args.run/'previews'/actual.name
            if sha(actual) != sha(expected): raise ValueError('Approved legacy preview changed: '+actual.name)
            unchanged_previews += 1
            raw_before = [a.tobytes() for a in (frame.left_rgb, frame.right_rgb)]
            canvas, metadata = overlay.render(frame, prediction)
            assert canvas.shape == (770, 2560, 3)
            assert raw_before == [a.tobytes() for a in (frame.left_rgb, frame.right_rgb)]
            decoded = decode(prediction.values17, model)
            for eye, e in label['eyes'].items():
                g = decoded[eye]; tool = e['tool']; result = metadata['projection_results'][eye]
                gold = {'C': tool['shaft_reference_position_mm'], 'J1': tool['jaw1_tip_position_mm'],
                        'J2': tool['jaw2_tip_position_mm'], 'T': tool['tip_position_mm'],
                        **{f'H{i}': np.asarray(e['local_frames'][f'jaw{i}']['T_optical_from_local_mm'])[:3, 3] for i in (1, 2)}}
                for name, value in gold.items(): maxima['position_mm'] = max(maxima['position_mm'], float(np.linalg.norm(g[name]-value)))
                saved = [tool['shaft_reference_pixel'], tool['tip_pixel'], *tool['jaw_tip_pixels']]
                error = abs(project([g[k] for k in ('C', 'T', 'J1', 'J2')], e['K'], e['D'])-saved).max()
                maxima['projection_px'] = max(maxima['projection_px'], float(error))
                assert result['jaw_position_indicator_origins'] == ['H1', 'H2']
                assert result['shaft_axis_inset']['placement'] == 'bottom_left'
                assert [v['placement'] for v in result['jaw_axis_insets']] == ['top_left', 'top_right']
                assert all(v['visible'] is None for v in result['jaw_visibility'])
                assert np.allclose(result['H1_camera_mm'], g['H1'])
                geometry = result['coordinate_axis_arrows']
                for name, entry in [('shaft', geometry), ('jaw1', geometry['jaw_frames'][0]), ('jaw2', geometry['jaw_frames'][1])]:
                    rotation = np.asarray(e['local_frames'][name]['T_optical_from_local_mm'])[:3, :3]
                    drawn = np.asarray([axis['direction_camera_unit'] for axis in entry['axes']]).T
                    maxima['rotation_element_abs'] = max(maxima['rotation_element_abs'], float(abs(drawn-rotation).max()))
            Image.fromarray(canvas).save(args.output/'last_replay.jpg', quality=95, subsampling=0)
        if not paths: raise ValueError('No frames to verify')
        assert XYZ_PANEL_BG_RGBA == (17, 29, 42, 102)
        assert maxima['position_mm'] < 1e-4 and maxima['projection_px'] < 1e-3 and maxima['rotation_element_abs'] < 1e-6
        rejected = []
        def rejects(name, callback):
            try: callback()
            except (ValueError, KeyError, TypeError): rejected.append(name)
            else: raise AssertionError('Not rejected: '+name)
        for name, index, value in [('nonfinite', 0, float('nan')), ('joint_limit', 14, 100.), ('negative_opening', 16, -.1)]:
            bad = copy.deepcopy(prediction); bad.values17[index] = value
            rejects(name, lambda: overlay.render(frame, bad))
        bad = copy.deepcopy(prediction); bad.values17[3:7] = [0.]*4
        rejects('zero_quaternion', lambda: overlay.render(frame, bad))
        bad = copy.deepcopy(prediction); bad.values17[2] = -1000.
        rejects('behind_camera', lambda: overlay.render(frame, bad))
        bad = copy.deepcopy(prediction); bad.frame_id = 'old_frame'
        rejects('stale_frame_id', lambda: overlay.render(frame, bad))
        bad = copy.deepcopy(prediction); bad.timestamp_ns -= 1
        rejects('stale_timestamp', lambda: overlay.render(frame, bad))
        bad_frame = copy.copy(frame); bad_frame.left_rgb = frame.left_rgb[:100]
        rejects('wrong_resolution', lambda: overlay.render(bad_frame, prediction))
        wrong_camera = copy.deepcopy(calibration); wrong_camera['image']['space'] = 'rectified'
        rejects('mismatched_image_geometry', lambda: StereoOverlay(model, wrong_camera, args.font))
        invalid = overlay.unavailable(frame, 'verification: invalid output')
        assert np.array_equal(invalid[50:, :1280], frame.left_rgb)
        assert np.array_equal(invalid[50:, 1280:], frame.right_rgb)
        # The explicit prediction branch must never claim truth or visible tips.
        synthetic = PosePrediction(frame.frame_id, frame.timestamp_ns, prediction.values17)
        _, meta = overlay.render(frame, synthetic)
        assert meta['source'] == 'prediction' and meta['visibility_source'] == 'unknown'
        for _ in range(5): overlay.render(frame, prediction)
        timings = []
        for i in range(50):
            frame, prediction = copy.copy(saved_frames[i % len(saved_frames)][0]), copy.copy(saved_frames[i % len(saved_frames)][1])
            frame.frame_id = prediction.frame_id = f'benchmark_{i:06d}'
            frame.timestamp_ns = prediction.timestamp_ns = round(i*1e9/60)
            started = time.perf_counter(); overlay.render(frame, prediction)
            timings.append((time.perf_counter()-started)*1000)
        # Two real consecutive source frames exercise the video adapter and codec.
        fixture = args.output/'video_fixture'; fixture.mkdir()
        pair = [load_pair(p) for p in paths[:2]]
        if pair[0][3]['sequence_id'] != pair[1][3]['sequence_id']: raise ValueError('Fixture needs a consecutive pair')
        for eye in ('left', 'right'):
            writer = cv2.VideoWriter(str(fixture/(eye+'.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), 60, (1280, 720))
            if not writer.isOpened(): raise RuntimeError('Fixture writer unavailable')
            for f, _, _, _ in pair: writer.write(cv2.cvtColor(getattr(f, eye+'_rgb'), cv2.COLOR_RGB2BGR))
            writer.release()
        write(fixture/'calibration.json', pair[0][2])
        write(fixture/'predictions.json', {'records': [{'frame_id': f'{i:06d}', 'timestamp_ns': round(i*1e9/60),
              'values17': p.values17} for i, (_, p, _, _) in enumerate(pair)]})
        write(fixture/'predictor.json', {'predictions_json': 'predictions.json'})
        command = [sys.executable, '-m', 'pose_display.video', '--left', str((fixture/'left.mp4').resolve()),
            '--right', str((fixture/'right.mp4').resolve()), '--calibration', str((fixture/'calibration.json').resolve()),
            '--instrument-model', str(args.instrument_model.resolve()), '--output', str((args.output/'video_test').resolve()),
            '--predictor', 'pose_display.recorded:build_predictor', '--predictor-config', str((fixture/'predictor.json').resolve())]
        if args.font: command += ['--font', args.font]
        subprocess.run(command, cwd=ROOT, check=True)
        video = read(args.output/'video_test/report.json')
        assert video['passed'] and video['frames'] == 2 and video['sources'] == ['ground_truth_replay']
        after = {str(p.relative_to(args.run)): sha(p) for p in args.run.rglob('*') if p.is_file()}
        assert before == after
        assert not any(name.split('.')[0] in ('isaacsim', 'omni', 'pxr') for name in sys.modules)
        report = {'passed': True, 'pairs': len(paths), 'pixel_identical_legacy_previews': unchanged_previews,
            'source_files_unchanged': len(before), 'new_rgb_renders': 0, 'trained_model_used': False,
            'maximum_errors': maxima, 'rejected_invalid_inputs': rejected, 'unknown_visibility_retained': True,
            'invalid_prediction_does_not_reuse_arrows': True, 'video_test': video,
            'postprocess_benchmark': {'samples': len(timings), 'median_ms': float(np.median(timings)),
                'p95_ms': float(np.percentile(timings, 95)), 'excludes': ['model_inference', 'image_IO', 'video_encoding', 'display_window'],
                'canvas_width_height': [2560, 770], 'scope': 'CPU decode, projection, arrows, labels and RGB canvas'},
            'annotation_code_sha256': annotation_code_hash(), 'instrument_model_sha256': sha(args.instrument_model),
            'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__, 'Pillow': PIL.__version__, 'opencv': cv2.__version__}}
        write(args.output/'report.json', report); print(json.dumps(report, indent=2))
    except BaseException as error:
        write(args.output/'report.json', {'passed': False, 'error': str(error)})
        raise


if __name__ == '__main__': main()
