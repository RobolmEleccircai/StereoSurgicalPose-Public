"""Compare GPU input preparation with the original Pillow/NumPy CPU path."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-config',type=Path)
    parser.add_argument('--candidate-config',type=Path)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(project));sys.path.insert(0,str(project/'reference'))
    from stereo_pose.libtorch_predict import LibTorchPredictor
    from pose_display import StereoFrame,CppStereoOverlay
    from pose_display.decoding import decode
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    from stereo_pose.libtorch_predict import build_predictor
    def configured(path, fallback):
        if not path:return fallback()
        c=json.loads(path.read_text());c['_config_dir']=str(path.resolve().parent);return build_predictor(c)
    cpu=configured(args.baseline_config,lambda:LibTorchPredictor(args.inputs/'best_val.pt',preprocessing='cpu'))
    gpu=configured(args.candidate_config,lambda:LibTorchPredictor(args.inputs/'best_val.pt',preprocessing='gpu'))
    bundle=json.loads((args.inputs/'samples.json').read_text())
    differences={'resized_uint8':0,'normalized_fp32':0,'K':0,'D':0,'hires':0,
                 'values17':0,'hidden':0,'derived_point_mm':0,'joint_deg':0,'overlay_rgb':0}
    renderers={};previous=None

    def compare_inputs(frame,cal):
        a,ak=cpu._prepare_inputs(frame,cal);b,bk=gpu._prepare_inputs(frame,cal)
        for name,x,y in zip(('normalized_fp32','K','D'),a,b):
            differences[name]=max(differences[name],float((x-y).abs().max()))
            assert x.stride()==y.stride(),(name,x.stride(),y.stride())
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        x,y=ak['hires_images'],bk['hires_images']
        differences['hires']=max(differences['hires'],int((x.to(torch.int16)-y.to(torch.int16)).abs().max()))
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        expected=np.stack([np.asarray(Image.fromarray(getattr(frame,e+'_rgb')).resize((640,360),Image.Resampling.BILINEAR))
                           for e in ('left','right')])
        actual=gpu.resized_rgb.cpu().numpy()
        differences['resized_uint8']=max(differences['resized_uint8'],int(np.abs(expected.astype(np.int16)-actual).max()))
        np.testing.assert_array_equal(expected,actual)

    for row in bundle['samples']:
        if row['sequence_id']!=previous: cpu.reset();gpu.reset()
        images=[np.asarray(Image.open(args.inputs/row['images'][e]).convert('RGB')) for e in ('left','right')]
        frame=StereoFrame(f"{row['sequence_id']}/{row['frame_index']:06d}",row['timestamp_ns'],*images)
        compare_inputs(frame,row['calibration'])
        a=cpu(frame,row['calibration']);b=gpu(frame,row['calibration'])
        differences['values17']=max(differences['values17'],float(np.abs(a.values17-b.values17).max()))
        differences['hidden']=max(differences['hidden'],float((cpu.hidden-gpu.hidden).abs().max()))
        np.testing.assert_array_equal(a.values17,b.values17)
        torch.testing.assert_close(cpu.hidden,gpu.hidden,rtol=0,atol=0)
        aa,bb=decode(a.values17,cpu.instrument_model),decode(b.values17,cpu.instrument_model)
        differences['derived_point_mm']=max(differences['derived_point_mm'],
            max(float(np.linalg.norm(aa[e][p]-bb[e][p])) for e in ('left','right') for p in ('C','H1','H2','J1','J2','T')))
        differences['joint_deg']=max(differences['joint_deg'],float(np.rad2deg(abs(a.values17[14:]-b.values17[14:])).max()))
        key=json.dumps(row['calibration'],sort_keys=True)
        if key not in renderers:renderers[key]=CppStereoOverlay(cpu.instrument_model,row['calibration'])
        ia,_=renderers[key].render(frame,a);ib,_=renderers[key].render(frame,b)
        differences['overlay_rgb']=max(differences['overlay_rgb'],int(np.abs(ia.astype(np.int16)-ib).max()))
        np.testing.assert_array_equal(ia,ib)
        previous=row['sequence_id']

    rng=np.random.default_rng(174)
    patterns=[np.zeros((720,1280,3),np.uint8),np.full((720,1280,3),255,np.uint8),
              rng.integers(0,256,(720,1280,3),dtype=np.uint8),
              np.broadcast_to((np.indices((720,1280)).sum(0)%2*255).astype(np.uint8)[...,None],(720,1280,3)),
              np.broadcast_to((np.arange(1280)%256).astype(np.uint8)[None,:,None],(720,1280,3))]
    edge=np.zeros((720,1280,3),np.uint8);edge[[0,-1],:]=255;edge[:,[0,-1]]=255;patterns.append(edge)
    for pattern in patterns:
        compare_inputs(StereoFrame('pattern',0,pattern,pattern[:,::-1]),row['calibration'])

    # Changed per-eye intrinsics/distortion must invalidate cached calibration.
    cal=copy.deepcopy(row['calibration']);cal['left']['K'][0][0]*=1.01;cal['right']['D'][0]+=.001
    compare_inputs(frame,cal)
    for predictor in (cpu,gpu):
        count=predictor.reset_count;predictor(frame,row['calibration']);assert predictor.reset_count==count+1
    np.testing.assert_array_equal(cpu(frame,cal).values17,gpu(frame,cal).values17)

    # Failed frames reset history; pinned buffers must remain safe to reuse.
    bad=StereoFrame('bad',0,images[0][:100],images[1])
    for predictor in (cpu,gpu):
        try:predictor(bad,row['calibration'])
        except ValueError:pass
        else:raise AssertionError('Malformed input accepted')
        assert predictor.hidden is None
    np.testing.assert_array_equal(cpu(frame,row['calibration']).values17,gpu(frame,row['calibration']).values17)
    for renderer in renderers.values():renderer.close()
    result={'passed':True,'stereo_pairs':len(bundle['samples']),
        'categories':len({r['category'] for r in bundle['samples']}),'synthetic_stereo_pairs':len(patterns),
        'maximum_differences':differences,'calibration_change_and_stream_reset_checked':True,
        'noncontiguous_inputs_and_error_recovery_checked':True,
        'checkpoint_sha256':hashlib.sha256((args.inputs/'best_val.pt').read_bytes()).hexdigest(),
        'baseline_config':str(args.baseline_config) if args.baseline_config else 'CPU Pillow/NumPy',
        'candidate_config':str(args.candidate_config) if args.candidate_config else 'original CUDA ATen',
        'meaning':'Implementation equivalence to the selected baseline plus Pillow pixel checks, not accuracy against ground truth.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
