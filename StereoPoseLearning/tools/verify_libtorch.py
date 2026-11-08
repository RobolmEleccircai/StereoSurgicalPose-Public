"""Check eager C++ LibTorch against the unchanged Python model on saved RGB."""
import argparse
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
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(project));sys.path.insert(0,str(project/'reference'))
    from stereo_pose.predict import StereoPredictor
    from stereo_pose.libtorch_predict import LibTorchPredictor
    from pose_display import StereoFrame,CppStereoOverlay
    from pose_display.decoding import decode
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    original=StereoPredictor(args.inputs/'best_val.pt',device='cuda')
    native=LibTorchPredictor(args.inputs/'best_val.pt',preprocessing='cpu')
    bundle=json.loads((args.inputs/'samples.json').read_text())
    captured={}
    hooks=[original.model.backbone.register_forward_hook(lambda m,a,v:captured.update({f'backbone_{i}':x for i,x in enumerate(v)})),
           original.model.spatial.register_forward_hook(lambda m,a,v:captured.update(spatial=v)),
           original.model.fusion.register_forward_hook(lambda m,a,v:captured.update(tokens=v))]
    previous=None;maxima={};point_errors=[];angle_errors=[];image_error=0
    engine_hidden=None;renderers={}
    for number,row in enumerate(bundle['samples']):
        if row['sequence_id']!=previous:
            original.reset();native.reset();engine_hidden=torch.zeros((3,2,256),device='cuda')
        images=[np.asarray(Image.open(args.inputs/row['images'][e]).convert('RGB')) for e in ('left','right')]
        frame=StereoFrame(f"{row['sequence_id']}/{row['frame_index']:06d}",row['timestamp_ns'],*images)
        inputs=original._preprocess(frame,row['calibration'])
        hires=torch.as_tensor(np.stack([x.transpose(2,0,1) for x in images]).copy(),device='cuda')[None,None]
        with torch.inference_mode():
            py_prediction,py_hidden=original.model(*inputs,hidden=engine_hidden,hires_images=hires)
            v,h,detail=native.model.engine.forward(*inputs,engine_hidden,hires,True)
            expected=dict(captured,**py_prediction)
            expected['values17']=original.model.export_values(py_prediction);expected['hidden']=py_hidden
            detail.update(values17=v,hidden=h)
            differences={k:float((value-expected[k]).abs().max().cpu()) for k,value in detail.items()}
            for k,x in differences.items():maxima[k]=max(maxima.get(k,0),x)
            if number==0:print(json.dumps({'first_frame_differences':differences}),flush=True)
            for k,value in detail.items():
                torch.testing.assert_close(value,expected[k],rtol=1e-5,atol=1e-4,msg=lambda msg,k=k:f'{k}: {msg}')
            engine_hidden=h
        a=original(frame,row['calibration']);b=native(frame,row['calibration'])
        np.testing.assert_allclose(b.values17,a.values17,rtol=1e-5,atol=1e-4)
        torch.testing.assert_close(native.hidden,original.hidden,rtol=1e-5,atol=1e-4)
        maxima['stream_values17']=max(maxima.get('stream_values17',0),float(np.abs(a.values17-b.values17).max()))
        aa,bb=decode(a.values17,original.instrument_model),decode(b.values17,original.instrument_model)
        point_errors.extend(float(np.linalg.norm(aa[e][p]-bb[e][p])) for e in ('left','right') for p in ('C','H1','H2','J1','J2','T'))
        angle_errors.extend(np.rad2deg(np.abs(a.values17[14:]-b.values17[14:])).tolist())
        key=json.dumps(row['calibration'],sort_keys=True)
        if key not in renderers:renderers[key]=CppStereoOverlay(original.instrument_model,row['calibration'])
        im_a,_=renderers[key].render(frame,a);im_b,_=renderers[key].render(frame,b)
        image_error=max(image_error,int(np.abs(im_a.astype(np.int16)-im_b.astype(np.int16)).max()))
        previous=row['sequence_id']
    before=native.reset_count;native(frame,row['calibration']);assert native.reset_count==before+1
    for hook in hooks:hook.remove()
    for overlay in renderers.values():overlay.close()
    result={'passed':True,'pairs':len(bundle['samples']),'categories':len({r['category'] for r in bundle['samples']}),
        'layer_max_abs_differences':maxima,'point_mm_max':max(point_errors),'joint_deg_max':max(angle_errors),
        'rendered_rgb_max_channel_difference':image_error,'stream_reset_checked':True,
        'scope':'Same checkpoint, FP32, original preprocessing, eager C++ LibTorch, no graph export or operator optimization.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
