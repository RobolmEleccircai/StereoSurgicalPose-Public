"""Resource/latency audit and separate intrusive operator profiling.

Production libraries remain in build/. Diagnostic libraries live in
build/profile/. No training, model conversion or precision changes are made.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from collections import defaultdict
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
from PIL import Image
import torch


def summary(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.percentile(a,95)),
                min=float(a.min()),max=float(a.max()))


def process_status():
    lines={s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/self/status').read_text().splitlines() if ':' in s}
    return {'rss_mib':int(lines['VmRSS'].split()[0])/1024,'threads':int(lines['Threads'])}


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('measure','stages','operators'),required=True)
    p.add_argument('--rounds',type=int,default=10)
    p.add_argument('--display-mode',choices=('vectors','full'),default='vectors')
    p.add_argument('--predictor-config',type=Path)
    p.add_argument('--overlay-library',type=Path)
    a=p.parse_args();a.inputs=a.inputs.resolve()
    if a.predictor_config and a.mode!='operators':raise ValueError('Configured runtime profiling uses operators mode; run benchmark_realtime --timing-detail stages for timing')
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a new output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(project));sys.path.insert(0,str(project/'reference'))
    native=project/'native/libtorch/build'
    if a.mode!='measure':native=native/'profile'
    sys.path.insert(0,str(native))
    if not a.predictor_config:
        import stereo_libtorch  # legacy diagnostic build
    from stereo_pose.libtorch_predict import LibTorchPredictor
    from pose_display import StereoFrame,CppStereoOverlay
    assert '4060' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if a.predictor_config:
        import importlib
        from stereo_pose.libtorch_predict import build_predictor
        c=json.loads(a.predictor_config.read_text());c['_config_dir']=str(a.predictor_config.resolve().parent)
        predictor=build_predictor(c);stereo_libtorch=importlib.import_module(predictor.model.native_module)
    else:predictor=LibTorchPredictor(a.inputs/'best_val.pt',preprocessing='gpu')
    bundle=json.loads((a.inputs/'samples.json').read_text())
    if digest(a.inputs/'best_val.pt')!=bundle['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
    loaded_model_status=process_status()
    rows=[];overlays={}
    overlay_library=a.overlay_library or project/'reference/pose_display/native/build/profile/libsurgical_overlay.so'
    for r in bundle['samples']:
        images=[np.asarray(Image.open(a.inputs/r['images'][e]).convert('RGB')) for e in ('left','right')]
        frame=StereoFrame(f"{r['sequence_id']}/{r['frame_index']:06d}",r['timestamp_ns'],*images)
        key=json.dumps(r['calibration'],sort_keys=True)
        if key not in overlays:
            renderer=CppStereoOverlay(predictor.instrument_model,r['calibration'],library=overlay_library,display_mode=a.display_mode)
            if a.mode!='operators':
                renderer._lib.sd_profile_last.argtypes=[ct.c_void_p,ct.c_void_p]
                renderer._lib.sd_profile_last.restype=ct.c_int
            overlays[key]=renderer
        rows.append((r,frame,overlays[key]))
    canvas=np.empty(next(iter(overlays.values())).canvas_shape,np.uint8)
    overlay_times=np.empty(5,np.float64)
    raw_bytes=sum(f.left_rgb.nbytes+f.right_rgb.nbytes for _,f,_ in rows)
    for r,f,o in rows:
        prediction=predictor(f,r['calibration']);o.render(f,prediction,out=canvas)
    # Verify that the diagnostic renderer has not changed the canvas.
    original_overlay=CppStereoOverlay(predictor.instrument_model,r['calibration'],display_mode=a.display_mode,library=overlay_library if a.predictor_config else None)
    expected,_=original_overlay.render(f,prediction)
    np.testing.assert_array_equal(expected,canvas);original_overlay.close()
    torch.cuda.synchronize()
    initial_status=process_status()
    common={'mode':a.mode,'display_mode':a.display_mode,'gpu':torch.cuda.get_device_name(0),'checkpoint_epoch':predictor.epoch+1,
        'checkpoint_sha256':bundle['checkpoint_sha256'],'pairs_available':len(rows),
        'batch_size':1,'precision':'Unchanged configured engine + FP32 downstream + FP64 C unprojection',
        'cuda_fusions':list(getattr(predictor,'cuda_fusions',())),
        'actual_execution':getattr(predictor.model,'actual_execution','reference'),
        'visual_engine':getattr(predictor,'visual_engine',None),
        'library_sha256':digest(stereo_libtorch.__file__),'overlay_library_sha256':digest(overlay_library),
        'diagnostic_overlay_pixels_equal':True,'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,
        'logical_cpus':os.cpu_count(),'allowed_cpus':len(os.sched_getaffinity(0)),'torch_cpu_threads':torch.get_num_threads()},
        'loaded_model_process_status':loaded_model_status,'preloaded_image_bytes':raw_bytes,
        'after_warmup_process_status':initial_status}
    if a.mode in ('measure','stages'):
        stage_records=defaultdict(list);stage_names=[]
        stage_events=[torch.cuda.Event(enable_timing=True) for _ in range(64)]
        if a.mode=='stages':
            def stage_hook(name):
                stage_events[len(stage_names)].record();stage_names.append(name)
            stereo_libtorch.set_stage_hook(stage_hook)
        current={};events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        original_preprocessor=predictor.preprocessor
        class TimedPreprocessor:
            def run(self):
                current['gpu_submit_host_ns']=time.perf_counter_ns()
                events[0].record();result=original_preprocessor.run();events[1].record()
                current['native_preprocess_host_ms']=(time.perf_counter_ns()-current['gpu_submit_host_ns'])/1e6
                return result
        predictor.preprocessor=TimedPreprocessor()
        def before(*_):
            current['model_host_start_ns']=time.perf_counter_ns();events[2].record()
        def after(*_):
            events[3].record();current['model_host_end_ns']=time.perf_counter_ns()
        hooks=[predictor.model.register_forward_pre_hook(before),predictor.model.register_forward_hook(after)]
        telemetry=(a.output/'gpu_telemetry.csv').open('w')
        sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,clocks.mem',
                                  '--format=csv,nounits','-lms','200'],stdout=telemetry,stderr=subprocess.DEVNULL)
        stop=threading.Event();resources=[]
        def monitor():
            while not stop.is_set():
                resources.append(process_status());stop.wait(.2)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        torch.cuda.reset_peak_memory_stats()
        records=[];cpu_before=os.times();wall_start=time.perf_counter()
        try:
            for repeat in range(a.rounds):
                predictor.reset()
                for r,f,o in rows:
                    stage_names.clear()
                    start=time.perf_counter_ns();prediction=predictor(f,r['calibration']);predicted=time.perf_counter_ns()
                    o.render(f,prediction,out=canvas);end=time.perf_counter_ns()
                    assert o._lib.sd_profile_last(o._handle,overlay_times.ctypes.data)==0
                    records.append({'category':r['category'],'total_ms':(end-start)/1e6,
                        'predictor_ms':(predicted-start)/1e6,'overlay_ms':(end-predicted)/1e6,
                        'host_before_gpu_submit_ms':(current['gpu_submit_host_ns']-start)/1e6,
                        'gpu_preprocess_interval_ms':events[0].elapsed_time(events[1]),
                        'gpu_preprocess_to_model_gap_ms':events[1].elapsed_time(events[2]),
                        'network_cuda_interval_ms':events[2].elapsed_time(events[3]),
                        'network_host_call_ms':(current['model_host_end_ns']-current['model_host_start_ns'])/1e6,
                        'return_wait_and_validation_ms':(predicted-current['model_host_end_ns'])/1e6,
                        **{name:float(t) for name,t in zip(('fk_project_ms','visibility_ms','canvas_copy_ms','left_panel_ms','right_panel_ms'),overlay_times)}})
                    for i,name in enumerate(stage_names[:-1]):
                        if name!='stage::__end__':stage_records[name].append(stage_events[i].elapsed_time(stage_events[i+1]))
            elapsed=time.perf_counter()-wall_start;cpu_after=os.times()
        finally:
            stop.set();thread.join();sampler.terminate();sampler.wait(timeout=10);telemetry.close()
            for hook in hooks:hook.remove()
            if a.mode=='stages':stereo_libtorch.set_stage_hook(None)
        means={k:summary([r[k] for r in records]) for k in records[0] if k.endswith('_ms')}
        cpu_seconds=cpu_after.user+cpu_after.system-cpu_before.user-cpu_before.system
        gpu_samples=[]
        for line in (a.output/'gpu_telemetry.csv').read_text().splitlines()[1:]:
            cols=line.split(',')
            if len(cols)==8:
                try:gpu_samples.append(list(map(float,cols[1:])))
                except ValueError:pass
        common.update(measured_pairs=len(records),warmup_pairs=len(rows),timing_ms=means,
            stage_cuda_intervals_ms={k:summary(v) for k,v in stage_records.items()},
            stereo_hz=1000/means['total_ms']['mean'],fraction_within_33_33_ms=float(np.mean([r['total_ms']<=1000/30 for r in records])),
            cpu={'process_core_equivalents':cpu_seconds/elapsed,'percent_of_one_core':100*cpu_seconds/elapsed,
                 'percent_of_available_cpus':100*cpu_seconds/elapsed/len(os.sched_getaffinity(0)),
                 'process_cpu_ms_per_pair':1000*cpu_seconds/len(records)},
            process_rss_mib=summary([r['rss_mib'] for r in resources]),process_thread_count=summary([r['threads'] for r in resources]),
            cuda_allocated_mib=torch.cuda.memory_allocated()/1024**2,cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            cuda_reserved_mib=torch.cuda.memory_reserved()/1024**2,cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            gpu_samples=len(gpu_samples),gpu_resource={k:summary(np.asarray(gpu_samples)[:,i]) for i,k in enumerate(
                ('device_memory_mib','gpu_utilization_percent','memory_controller_utilization_percent','power_w','temperature_c','sm_clock_mhz','memory_clock_mhz'))},
            notes=['Wall latency is measured without torch.profiler; the diagnostic overlay adds five CPU timer reads.',
                   'CPU submission, GPU intervals and return wait overlap: do not add their columns.',
                   'return_wait includes waiting for pending model kernels, not just copying 68 bytes.',
                   'Input RGB is preloaded in RAM; disk decode, acquisition, encoding and GUI refresh are excluded.',
                   'GPU utilization measures busy time, not fraction of peak FLOPs; device memory includes context/desktop.'])
        (a.output/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        Image.fromarray(canvas).save(a.output/'preview.jpg',quality=95)
    else:
        selected=[];counts=defaultdict(int)
        for row in rows:
            if counts[row[0]['sequence_id']]<2:selected.append(row)
            counts[row[0]['sequence_id']]+=1
        from torch.profiler import profile,ProfilerActivity,record_function
        preprocessing_events=[]
        original_prepare=predictor._prepare_inputs
        def traced_prepare(*args,**kwargs):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();value=original_prepare(*args,**kwargs);end.record()
            preprocessing_events.append((begin,end));return value
        predictor._prepare_inputs=traced_prepare
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,
                     with_flops=True,profile_memory=True) as prof:
            predictor.reset()
            for r,f,o in selected:
                with record_function('pipeline::predict'):prediction=predictor(f,r['calibration'])
                with record_function('pipeline::draw'):o.render(f,prediction,out=canvas)
        predictor._prepare_inputs=original_prepare
        torch.cuda.synchronize()
        common['profiled_preprocess_cuda_ms']=summary([a.elapsed_time(b) for a,b in preprocessing_events])
        number=len(selected)
        stages=[]
        for e in prof.key_averages():
            if e.key.startswith('stage::') and str(e.device_type)=='DeviceType.CPU':
                stages.append({'stage':e.key,'count':e.count,'cpu_inclusive_ms_per_pair':e.cpu_time_total/1000/number,
                    'gpu_active_ms_per_pair':e.device_time_total/1000/number})
        ops=[];flops=0;by_stage=defaultdict(float);flops_by_op=defaultdict(float)
        for e in prof.events():
            if e.flops:
                flops+=e.flops
                flops_by_op[e.name]+=e.flops
                parent=e.cpu_parent
                while parent and not parent.name.startswith('stage::'):parent=parent.cpu_parent
                by_stage[parent.name if parent else 'other']+=e.flops
        averages=[e for e in prof.key_averages() if str(e.device_type)=='DeviceType.CPU' and e.key.startswith('aten::')]
        for e in sorted(averages,key=lambda e:e.self_device_time_total,reverse=True)[:35]:
            ops.append({'name':e.key,'count_per_pair':e.count/number,
                'self_device_ms_per_pair':e.self_device_time_total/1000/number,
                'self_cpu_ms_per_pair':e.self_cpu_time_total/1000/number})
        with tempfile.NamedTemporaryFile(suffix='.json',prefix='stereo-profile-',delete=False) as f:trace_path=Path(f.name)
        try:
            prof.export_chrome_trace(str(trace_path));trace=json.loads(trace_path.read_text())['traceEvents']
        finally:trace_path.unlink(missing_ok=True)
        kernels=[e for e in trace if e.get('cat')=='kernel' and e.get('ph')=='X']
        copies=[e for e in trace if e.get('cat')=='gpu_memcpy' and e.get('ph')=='X']
        runtime=[e for e in trace if e.get('cat')=='cuda_runtime' and e.get('ph')=='X']
        grouped=defaultdict(lambda:{'count':0,'us':0})
        for e in runtime:
            grouped[e['name']]['count']+=1;grouped[e['name']]['us']+=e['dur']
        # Compact trace-derived evidence; discard the large raw trace.
        transfer_groups=defaultdict(lambda:{'count':0,'us':0,'bytes':0})
        for e in copies:
            g=transfer_groups[e['name']];g['count']+=1;g['us']+=e['dur'];g['bytes']+=e.get('args',{}).get('bytes',0)
        common.update(profiled_pairs=number,stages=stages,top_operators=ops,
            supported_ops_gflops_per_pair=flops/number/1e9,
            supported_ops_gflops_by_stage={k:v/number/1e9 for k,v in by_stage.items()},
            supported_ops_gflops_by_operator={k:v/number/1e9 for k,v in flops_by_op.items()},
            kernels_per_pair=len(kernels)/number,kernel_active_ms_per_pair=sum(e['dur'] for e in kernels)/number/1000,
            memcpy_active_ms_per_pair=sum(e['dur'] for e in copies)/number/1000,
            kernel_names={k:{'calls_per_pair':sum(1 for e in kernels if e['name']==k)/number,
                'gpu_ms_per_pair':sum(e['dur'] for e in kernels if e['name']==k)/number/1000} for k in sorted({e['name'] for e in kernels})},
            cuda_runtime={k:{'calls_per_pair':v['count']/number,'cpu_ms_per_pair':v['us']/number/1000} for k,v in grouped.items()},
            transfers={k:{'calls_per_pair':v['count']/number,'gpu_ms_per_pair':v['us']/number/1000,'bytes_per_pair':v['bytes']/number} for k,v in transfer_groups.items()},
            notes=['Intrusive operator profiling is separate from real latency measurement.',
                   'GPU active times sum attributed kernel/memcpy durations; they exclude launch gaps and CPU drawing.',
                   'FLOPs estimate profiler-supported operations; unsupported operations are not counted. See per-operator breakdown.',
                   'No large Chrome trace is retained.'])
    for renderer in overlays.values():renderer.close()
    (a.output/'report.json').write_text(json.dumps(common,indent=2)+'\n')
    print(json.dumps(common,indent=2),flush=True)


if __name__=='__main__':main()
