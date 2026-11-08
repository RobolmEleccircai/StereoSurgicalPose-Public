"""Validate the requested three-vector display using existing stereo images."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(project));sys.path.insert(0,str(project/'reference'))
    from stereo_pose.libtorch_predict import LibTorchPredictor
    from pose_display import StereoFrame,PosePrediction,CppStereoOverlay
    from pose_display.decoding import decode
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    predictor=LibTorchPredictor(a.inputs/'best_val.pt')
    bundle=json.loads((a.inputs/'samples.json').read_text());seen=set();cases=[];max_error=0
    for r in bundle['samples']:
        if r['category'] in seen:continue
        seen.add(r['category']);predictor.reset()
        images=[np.asarray(Image.open(a.inputs/r['images'][e]).convert('RGB')) for e in ('left','right')]
        frame=StereoFrame('test_frame_000000',0,*images);pred=predictor(frame,r['calibration'])
        normal=CppStereoOverlay(predictor.instrument_model,r['calibration'])
        detail=CppStereoOverlay(predictor.instrument_model,r['calibration'],
            library=project/'reference/pose_display/native/build/profile/libsurgical_overlay.so')
        full=CppStereoOverlay(predictor.instrument_model,r['calibration'],display_mode='full')
        geometry=normal.compute_geometry(pred.values17)
        assert geometry==full.compute_geometry(pred.values17)
        expected=decode(pred.values17,predictor.instrument_model)
        values={}
        for eye in ('left','right'):
            for field in ('C','d','H1','H2','J1','J2'):
                max_error=max(max_error,float(np.abs(np.asarray(geometry[eye][field])-expected[eye][field]).max()))
            g=expected[eye]
            values[eye]={'d_unit':g['d'].tolist(),'J1_minus_H1_mm':(g['J1']-g['H1']).tolist(),
                         'J2_minus_H2_mm':(g['J2']-g['H2']).tolist()}
        im,_=normal.render(frame,pred);diagnostic,_=detail.render(frame,pred)
        assert im.shape==(720,2560,3)
        np.testing.assert_array_equal(im,diagnostic)
        changed=StereoFrame('completely_different_identifier',123456789,*images)
        changed_pred=PosePrediction(changed.frame_id,changed.timestamp_ns,pred.values17,source='prediction')
        second,_=normal.render(changed,changed_pred)
        np.testing.assert_array_equal(im,second) # no frame ID/time text in pixels
        fn=detail._lib.sd_profile_paint;fn.argtypes=[ct.c_void_p,ct.c_void_p];fn.restype=ct.c_int
        cases.append((frame,pred,detail,values))
        normal.close();full.close()
    assert max_error<1e-9
    times=[];wall=[];buffer=np.empty((720,2560,3),np.uint8);tmp=np.empty(3,np.float64)
    for repeat in range(40):
        for f,pred,renderer,_ in cases:
            start=time.perf_counter_ns();renderer.render(f,pred,out=buffer);end=time.perf_counter_ns()
            assert renderer._lib.sd_profile_paint(renderer._handle,tmp.ctypes.data)==0
            if repeat>=10:times.append(tmp.copy());wall.append((end-start)/1e6)
    report={'passed':True,'categories':len(cases),'geometry_max_abs_difference':max_error,
        'frame_id_timestamp_pixel_invariant':True,'production_diagnostic_canvas_equal':True,
        'output_shape':[720,2560,3],'vector_definitions':{'shaft':'d, unit direction','jaw1':'J1-H1, mm','jaw2':'J2-H2, mm'},
        'coordinate_frame':'each corresponding camera optical frame',
        'draw_only_timed_pairs':len(times),'draw_only_mean_ms':float(np.mean(wall)),
        'paint_cpu_ms':dict(zip(('rounded_panels','arrows','numeric_text'),np.mean(times,axis=0).tolist())),
        'example_vectors':cases[-1][3],
        'note':'CPU-only drawing timing is separate from end-to-end inference. No simulation rendering.'}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    Image.fromarray(buffer).save(a.output/'preview.jpg',quality=95)
    for _,_,renderer,_ in cases:renderer.close()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
