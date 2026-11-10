"""Measure actual single-pair streaming inference plus native stereo overlay.

Inputs are preloaded original RGB and per-camera K/D. No label poses, masks,
baseline or inter-camera extrinsics enter the predictor. Batch/time are 1/1;
the model's three recurrent states persist within each consecutive sequence.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def statistics(values):
    a = np.asarray(values, dtype=float)
    return {'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
            'p95_ms': float(np.percentile(a, 95)), 'p99_ms': float(np.percentile(a, 99)),
            'min_ms': float(a.min()), 'max_ms': float(a.max())}


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem',
        '--format=csv'], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', default='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    parser.add_argument('--expected-gpu', default='4060')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--modes', nargs='+', choices=('fp32', 'bf16'), default=['fp32', 'bf16'])
    parser.add_argument('--inference-backend', choices=('python', 'libtorch'), default='python')
    parser.add_argument('--preprocessing', choices=('cpu','gpu'), default=None)
    parser.add_argument('--display-mode',choices=('vectors','full'),default='vectors')
    parser.add_argument('--visual-engine',type=Path,help='Optional TensorRT visual frontend; downstream remains LibTorch')
    parser.add_argument('--overlay-library',type=Path,help='Optional separately built native renderer')
    parser.add_argument('--predictor-config',type=Path,help='Native runtime configuration; overrides checkpoint/engine/execution selection')
    args = parser.parse_args()
    if args.visual_engine is not None and args.inference_backend!='libtorch':
        parser.error('--visual-engine requires --inference-backend libtorch')
    if args.preprocessing is None:
        args.preprocessing='gpu' if args.inference_backend=='libtorch' else 'cpu'
    if args.preprocessing=='gpu' and args.inference_backend!='libtorch':
        parser.error('GPU preprocessing uses the native LibTorch input path')
    if args.inference_backend == 'libtorch' and args.modes != ['fp32']:
        parser.error('LibTorch comparison keeps FP32: specify --modes fp32')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new benchmark output directory')
    if args.rounds < 1:
        raise ValueError('rounds must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.project)); sys.path.insert(0, str(args.project/'reference'))
    from stereo_pose.predict import StereoPredictor
    from pose_display import StereoFrame, CppStereoOverlay
    from pose_display.decoding import decode

    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    gpu_before = gpu_snapshot()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; CPU substitution is not permitted')
    name = torch.cuda.get_device_name(0)
    if args.expected_gpu not in name:
        raise RuntimeError(f'Unexpected GPU: {name}')
    free, total = torch.cuda.mem_get_info()
    if free < 3 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory for an isolated inference test')
    torch.cuda.set_per_process_memory_fraction(.75, 0)
    # Keep downstream FP32 / cuDNN / TF32 behavior. An optional visual engine
    # changes only the frontend; input size and single-pair scope stay fixed.
    modes = args.modes
    if 'bf16' in modes and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requested BF16 comparison is unsupported')
    bundle = json.loads((args.inputs/'samples.json').read_text())
    checkpoint = args.checkpoint or args.inputs/'best_val.pt'
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != bundle['checkpoint_sha256']:
        raise ValueError('Transferred checkpoint hash mismatch')
    if args.inference_backend == 'libtorch':
        from stereo_pose.libtorch_predict import LibTorchPredictor
        if args.predictor_config:
            from stereo_pose.libtorch_predict import build_predictor
            config=json.loads(args.predictor_config.read_text());config['_config_dir']=str(args.predictor_config.resolve().parent)
            predictor=build_predictor(config)
            if digest(predictor.checkpoint_path)!=checkpoint_hash:raise ValueError('Configuration checkpoint mismatch')
        else:
            predictor = LibTorchPredictor(checkpoint, preprocessing=args.preprocessing,visual_engine=args.visual_engine)
    else:
        predictor = StereoPredictor(checkpoint, device='cuda')
    if predictor.instrument_model['source_scene_sha256'] != bundle['source_scene_sha256']:
        raise ValueError('Checkpoint geometry and image assets differ')
    groups = defaultdict(list)
    overlays = {}
    for row in bundle['samples']:
        calibration = row['calibration']
        fingerprint = json.dumps(calibration, sort_keys=True)
        if fingerprint not in overlays:
            overlays[fingerprint] = CppStereoOverlay(predictor.instrument_model, calibration, args.font,display_mode=args.display_mode,library=args.overlay_library)
        images = [np.asarray(Image.open(args.inputs/row['images'][e]).convert('RGB'))
                  for e in ('left', 'right')]
        frame = StereoFrame(f"{row['sequence_id']}/{row['frame_index']:06d}", row['timestamp_ns'], *images)
        groups[row['sequence_id']].append((row, frame, overlays[fingerprint]))
    for rows in groups.values():
        rows.sort(key=lambda x: x[0]['frame_index'])
        for a, b in zip(rows, rows[1:]):
            assert b[0]['frame_index'] == a[0]['frame_index']+1
            assert abs(b[0]['timestamp_ns']-a[0]['timestamp_ns']-1e9/60) <= 1
    canvas = np.empty(next(iter(overlays.values())).canvas_shape, dtype=np.uint8)
    events = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
    hooks = [predictor.model.register_forward_pre_hook(lambda *_: events[0].record()),
             predictor.model.register_forward_hook(lambda *_: events[1].record())]
    stage = {}
    original_preprocess = predictor._prepare_inputs
    def measured_preprocess(*a, **kw):
        start = time.perf_counter_ns(); result = original_preprocess(*a, **kw)
        stage['input_preparation_host_wall_ms'] = (time.perf_counter_ns()-start)/1e6
        return result
    predictor._prepare_inputs = measured_preprocess
    telemetry_file = (args.output/'gpu_telemetry.csv').open('w')
    telemetry = subprocess.Popen(['nvidia-smi',
        '--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm',
        '--format=csv', '-lms', '1000'], stdout=telemetry_file, stderr=subprocess.DEVNULL)
    details, summaries, predictions = [], {}, {}
    try:
        for mode in modes:
            def context():
                return nullcontext() if mode == 'fp32' else torch.autocast('cuda', dtype=torch.bfloat16)
            cold_ms = None
            # Warm all input categories, C++ label tiles, CUDA kernels and GRUs.
            for rows in groups.values():
                predictor.reset()
                for row, frame, overlay in rows:
                    start = time.perf_counter_ns()
                    with context(): prediction = predictor(frame, row['calibration'])
                    overlay.render(frame, prediction, out=canvas)
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    if cold_ms is None: cold_ms = elapsed
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            measured = []
            measured_wall=time.perf_counter();measured_cpu=time.process_time()
            example = None
            for repeat in range(args.rounds):
                for sid, rows in groups.items():
                    predictor.reset(); reset_count = predictor.reset_count
                    for row, frame, overlay in rows:
                        start = time.perf_counter_ns()
                        with context(): prediction = predictor(frame, row['calibration'])
                        after_prediction = time.perf_counter_ns()
                        image, metadata = overlay.render(frame, prediction, out=canvas)
                        end = time.perf_counter_ns()
                        # Returning a CPU numpy prediction synchronizes the GPU
                        # result; this is real completion, not launch timing.
                        events[1].synchronize()
                        assert metadata['valid'] and metadata['backend'] == 'cpp'
                        assert predictor.reset_count == reset_count, 'Unexpected mid-sequence state reset'
                        record = {'mode': mode, 'repeat': repeat, 'sequence_id': sid,
                            'category': row['category'], 'frame_index': row['frame_index'],
                            'predictor_including_preprocess_transfer_export_ms': (after_prediction-start)/1e6,
                            'cpp_projection_and_draw_ms': (end-after_prediction)/1e6,
                            'end_to_end_ms': (end-start)/1e6,
                            'network_cuda_interval_ms': events[0].elapsed_time(events[1]), **stage}
                        measured.append(record); details.append(record)
                        if repeat == 0:
                            predictions[(mode, sid, row['frame_index'])] = prediction.values17.copy()
                        if example is None and row is rows[-1][0]:
                            example = image.copy()
                print(json.dumps({'mode': mode, 'round_complete': repeat+1,
                    'pairs_so_far': len(measured), 'mean_total_ms': np.mean([r['end_to_end_ms'] for r in measured])}), flush=True)
            fields = ('predictor_including_preprocess_transfer_export_ms', 'cpp_projection_and_draw_ms',
                      'end_to_end_ms', 'network_cuda_interval_ms', 'input_preparation_host_wall_ms')
            summary = {field: statistics([r[field] for r in measured]) for field in fields}
            total_times = [r['end_to_end_ms'] for r in measured]
            cpu_seconds=time.process_time()-measured_cpu
            process_wall=time.perf_counter()-measured_wall
            summary.update(process_cpu_seconds=cpu_seconds,process_cpu_percent=100*cpu_seconds/process_wall,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                measured_pairs=len(measured), warmup_pairs=len(bundle['samples']),
                first_warmup_pair_ms=cold_ms, hz=1000/np.mean(total_times),
                inverse_p95_hz=1000/np.percentile(total_times,95),
                fraction_within_33_33_ms=float(np.mean(np.asarray(total_times)<=1000/30)),
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                per_category={category: statistics([r['end_to_end_ms'] for r in measured if r['category']==category])
                              for category in sorted({r['category'] for r in measured})})
            summaries[mode] = summary
            if example is not None: Image.fromarray(example).save(args.output/(mode+'_overlay.jpg'), quality=95)
        drift = {}
        if 'bf16' in modes and 'fp32' in modes:
            positions, tips, joints = [], [], []
            for row in bundle['samples']:
                key = row['sequence_id'], row['frame_index']
                a, b = predictions[('fp32', *key)], predictions[('bf16', *key)]
                aa, bb = decode(a, predictor.instrument_model), decode(b, predictor.instrument_model)
                joints.append(np.rad2deg(abs(a[14:]-b[14:])))
                for eye in ('left', 'right'):
                    positions.append(float(np.linalg.norm(aa[eye]['C']-bb[eye]['C'])))
                    for point in ('J1','J2'): tips.append(float(np.linalg.norm(aa[eye][point]-bb[eye][point])))
            drift = {'meaning': 'BF16 disagreement with FP32, not error against ground truth',
                     'C_mm_mean': float(np.mean(positions)), 'C_mm_max': max(positions),
                     'tip_mm_mean': float(np.mean(tips)), 'tip_mm_max': max(tips),
                     'joint_deg_mean_pitch_yaw_half': np.mean(joints,0).tolist(),
                     'joint_deg_max_pitch_yaw_half': np.max(joints,0).tolist()}
        result = {'completed': True, 'gpu': name, 'gpu_before': gpu_before, 'gpu_after': gpu_snapshot(),
            'inference_backend': args.inference_backend,
            'execution_mode':getattr(predictor,'execution_mode','reference'),
            'actual_execution':getattr(predictor.model,'actual_execution','reference'),
            'graph_error':getattr(predictor.model,'graph_error',None),
            'preprocessing': args.preprocessing,
            'display_mode': args.display_mode,
            'overlay_library':str(args.overlay_library) if args.overlay_library else 'reference',
            'overlay_library_sha256':digest(args.overlay_library or args.project/'reference/pose_display/native/build/libsurgical_overlay.so'),
            'visual_engine':getattr(predictor,'visual_engine',None),
            'checkpoint_epoch': predictor.epoch+1, 'checkpoint_sha256': checkpoint_hash,
            'raw_resolution_per_eye': [1280,720], 'network_resolution_hw': list(predictor.image_size),
            'batch_size': 1, 'time_steps_per_call': 1, 'causal_states': 'body/wrist/jaw, retained within each eight-frame sequence',
            'input_pairs': len(bundle['samples']), 'sequences': len(groups), 'results': summaries,
            'mixed_precision_drift': drift,
            'timing_scope': 'RAM RGB -> resize/normalize -> all H2D transfers -> real model -> 17 values D2H -> C++ FK/distortion projection/Chinese arrows and XYZ panels -> completed RGB canvas. No disk/image decoding, capture latency, video encoding, imshow or refresh.',
            'cuda_timing_note': 'Network CUDA interval excludes preprocessing and CPU drawing; end-to-end wall time is the primary result. GPU input_preparation_host_wall_ms is submission time, not completed GPU time, and cannot be added to the model interval to obtain latency. Records and metrics are written after timing.',
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'cpu_threads': torch.get_num_threads(), 'cudnn_benchmark': torch.backends.cudnn.benchmark,
                        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32},
            'source_sha256': {str(p.relative_to(args.project)): digest(p) for p in
                [args.project/'stereo_pose/predict.py', args.project/'stereo_pose/model.py',
                 args.project/'reference/pose_display/native/overlay.cpp', Path(__file__).resolve()]}}
        if args.inference_backend == 'libtorch':
            import importlib
            module=importlib.import_module(predictor.model.native_module)
            result['native_library_sha256']=digest(module.__file__)
            result['preprocess_library_sha256']=digest(importlib.import_module('stereo_libtorch').__file__)
            result['source_sha256'].update({str(p.relative_to(args.project)): digest(p) for p in
                [args.project/'native/libtorch/model.cpp', args.project/'native/libtorch/model.h',
                 args.project/'native/libtorch/bindings.cpp', args.project/'native/libtorch/preprocess.cpp',
                 args.project/'native/libtorch/preprocess.h', args.project/'stereo_pose/libtorch_predict.py']})
        (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
        (args.output/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        for h in hooks: h.remove()
        telemetry.terminate(); telemetry.wait(timeout=10); telemetry_file.close()
        for overlay in overlays.values(): overlay.close()


if __name__ == '__main__':
    main()
