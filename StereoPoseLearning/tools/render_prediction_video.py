"""Saved stereo RGB sequences -> real model predictions -> approved C++ overlays.

Reads only image paths, frame timing and K/D for inference. No saved pose, mask,
visibility, baseline or camera-extrinsic truth is supplied to the predictor.
Sequence cuts are explicit; no interpolation, smoothing or frame dropping.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Encoder:
    def __init__(self, path, width, height, fps):
        self.path=path
        self.log=path.with_suffix('.ffmpeg.log').open('wb')
        self.proc=subprocess.Popen([
            'ffmpeg','-hide_banner','-loglevel','error','-n',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
            '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
            '-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2',
            '-movflags','+faststart',str(path)],stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,stderr=self.log)
        self.count=0

    def write(self, rgb):
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes());self.count+=1

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:self.proc.stdin.close()
        code=self.proc.wait();self.log.close()
        if code:raise RuntimeError(f'Video encoder failed ({code}): {self.path}')
        info=json.loads(subprocess.check_output([
            'ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=nb_frames,width,height,r_frame_rate','-of','json',str(self.path)],text=True))['streams'][0]
        if int(info['nb_frames'])!=self.count:raise RuntimeError('Video frame count differs')
        return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--sequences',nargs='+',help='Default: every sequence, longest first')
    parser.add_argument('--playback-fps',type=int,default=60,help='Output playback rate; source timing is unchanged')
    parser.add_argument('--path-prefix',nargs=2,action='append',default=[],metavar=('ORIGINAL','LOCAL'),
                        help='Resolve archived input paths without modifying the source manifest')
    parser.add_argument('--predictor-config',type=Path,required=True)
    parser.add_argument('--overlay-library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    if args.playback_fps<=0:raise ValueError('Playback fps must be positive')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use a new empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(args.project),str(args.project/'reference')]
    from stereo_pose.libtorch_predict import build_predictor
    from pose_display import StereoFrame,CppStereoOverlay
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    config=json.loads(args.predictor_config.read_text());config['_config_dir']=str(args.predictor_config.resolve().parent)
    predictor=build_predictor(config)
    manifest=json.loads(args.manifest.read_text())
    if args.sequences is None:
        counts={}
        for row in manifest['records']:counts[row['sequence_id']]=counts.get(row['sequence_id'],0)+1
        args.sequences=sorted(counts,key=lambda sid:(-counts[sid],sid))
    if len(set(args.sequences))!=len(args.sequences):raise ValueError('Duplicate requested sequences')
    combined=None;encoder=None;sequence_reports=[];total=0;processing=[]
    prediction_file=(args.output/'predictions.jsonl').open('w')
    started=time.time()
    def resolve_input(value):
        path=Path(value)
        for old,new in args.path_prefix:
            if path.is_relative_to(Path(old)):return Path(new)/path.relative_to(Path(old))
        return path
    def load(row):
        label=json.loads(resolve_input(row['label_path']).read_text())
        calibration={'image':{'width':1280,'height':720,'space':'raw_distorted'}}
        for eye in ['left','right']:
            calibration[eye]={'K':label['eyes'][eye]['K'],'D':label['eyes'][eye]['D']}
        images=[np.array(Image.open(resolve_input(row['image_paths'][eye])).convert('RGB')) for eye in ['left','right']]
        if any(x.shape!=(720,1280,3) for x in images):raise ValueError('Unexpected input resolution')
        return row,StereoFrame(f"{row['sequence_id']}/{row['frame_index']:06d}",int(row['timestamp_ns']),*images),calibration
    try:
        for sid in args.sequences:
            rows=sorted([x for x in manifest['records'] if x['sequence_id']==sid],key=lambda x:x['frame_index'])
            if not rows:raise ValueError('Sequence not found: '+sid)
            if len({x['trajectory_group_id'] for x in rows})!=1:raise ValueError('Sequence ID is ambiguous')
            if any(abs(float(x['fps'])-60)>1e-9 for x in rows):raise ValueError('This review expects 60 Hz source data')
            for a,b in zip(rows,rows[1:]):
                if b['frame_index']!=a['frame_index']+1 or abs(b['timestamp_ns']-a['timestamp_ns']-1e9/60)>1:
                    raise ValueError('Missing frame or discontinuous timestamps')
            if any(x['source_scene_sha256']!=predictor.instrument_model['source_scene_sha256'] for x in rows):
                raise ValueError('Source scene differs from checkpoint geometry')
            with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(load,rows))
            overlays={}
            for _,_,cal in frames:
                key=json.dumps(cal,sort_keys=True)
                if key not in overlays:overlays[key]=CppStereoOverlay(predictor.instrument_model,cal,library=args.overlay_library,display_mode='vectors')
            canvas=np.empty(next(iter(overlays.values())).canvas_shape,np.uint8)
            # Warm up then reset all three states before processing real video frames.
            predictor.reset()
            for row,frame,cal in frames[:8]:
                pred=predictor(frame,cal);overlays[json.dumps(cal,sort_keys=True)].render(frame,pred,out=canvas)
            predictor.reset();reset_count=predictor.reset_count
            name=f"{rows[0]['category']}_{sid}.mp4";encoder=Encoder(args.output/name,2560,720,args.playback_fps)
            if combined is None:combined=Encoder(args.output/f'stereo_pose_predictions_{args.playback_fps}fps.mp4',2560,720,args.playback_fps)
            first_output=total;times=[];values=[]
            for i,(row,frame,cal) in enumerate(frames):
                t=time.perf_counter_ns();pred=predictor(frame,cal)
                if pred.source!='prediction':raise AssertionError('Only actual model predictions are permitted')
                image,meta=overlays[json.dumps(cal,sort_keys=True)].render(frame,pred,out=canvas)
                elapsed=(time.perf_counter_ns()-t)/1e6;times.append(elapsed);processing.append(elapsed)
                if not meta['valid'] or predictor.reset_count!=reset_count:raise AssertionError('Unexpected state reset or invalid overlay')
                encoder.write(image);combined.write(image)
                record={'output_frame_index':total,'sequence_id':sid,'source_frame_index':row['frame_index'],
                        'source_timestamp_ns':frame.timestamp_ns,'output_time_s':total/args.playback_fps,
                        'category':row['category'],'source':'prediction','values17':pred.values17.tolist(),
                        'predict_and_draw_ms':elapsed,'calibration':cal}
                prediction_file.write(json.dumps(record,allow_nan=False)+'\n')
                values.append(pred.values17.copy());total+=1
                if i==len(frames)//2:Image.fromarray(image).save(args.output/f'{sid}_preview.jpg',quality=95)
            info=encoder.close();encoder=None;v=np.asarray(values)
            sequence_reports.append({'sequence_id':sid,'category':rows[0]['category'],'video':name,
                                     'frames':len(rows),'source_first_frame':rows[0]['frame_index'],
                                     'source_last_frame':rows[-1]['frame_index'],'combined_start_frame':first_output,
                                     'duration_s':len(rows)/args.playback_fps,'source_duration_s':len(rows)/60,
                                     'model_state_resets_before_sequence':True,
                                     'unexpected_mid_sequence_resets':0,'video_info':info,
                                     'mean_predict_and_draw_ms':float(np.mean(times)),
                                     'values17_range':(v.max(0)-v.min(0)).tolist()})
            for overlay in overlays.values():overlay.close()
            print(json.dumps(sequence_reports[-1]),flush=True)
            del frames,overlays
        combined_info=combined.close();combined=None;prediction_file.close()
        report={'completed':True,'frames':total,'fps':args.playback_fps,'source_fps':60,
                'duration_s':total/args.playback_fps,'source_duration_s':total/60,
                'playback_speed_ratio':args.playback_fps/60,'source':'prediction',
                'source_manifest_sha256':sha(args.manifest),'path_prefix_remapping':args.path_prefix,
                'model_epoch':predictor.epoch+1,'checkpoint_sha256':sha(predictor.checkpoint_path),
                'overlay_library_sha256':sha(args.overlay_library),'predictor_configuration':config,
                'actual_execution':getattr(predictor.model,'actual_execution',None),
                'three_vector_layout':'shaft plus jaw1/jaw2, own XYZ axes, values and transparent panels',
                'smoothing':False,'frame_interpolation':False,'dropped_source_frames':0,
                'ground_truth_pose_supplied_to_model':False,'model_input':'RGB left/right + per-eye K,D + causal states',
                'sequence_cuts':[x['combined_start_frame'] for x in sequence_reports[1:]],
                'sequence_cut_note':'Independent sequences are joined by hard cuts; states reset only at sequence boundaries.',
                'sequences':sequence_reports,'video_info':combined_info,
                'mean_predict_and_draw_ms':float(np.mean(processing)),
                'render_wall_seconds':time.time()-started,
                'timing_note':'Playback rate is separate from source timestamps. Timing is observational during video encoding, not an isolated latency benchmark.'}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'completed':True,'frames':total,'duration_s':total/args.playback_fps}),flush=True)
    except BaseException as error:
        (args.output/'report.json').write_text(json.dumps({'completed':False,'frames':total,'error':repr(error)},indent=2))
        raise
    finally:
        if not prediction_file.closed:prediction_file.close()
        for active in (encoder,combined):
            if active is not None and active.proc.poll() is None:
                if active.proc.stdin and not active.proc.stdin.closed:active.proc.stdin.close()
                active.proc.wait();active.log.close()


if __name__=='__main__':main()
