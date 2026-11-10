"""CPU-only pixel and geometry comparison using already saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'reference'))
    from pose_display import CppStereoOverlay,PosePrediction,StereoFrame
    model=torch.load(a.inputs/'best_val.pt',map_location='cpu',weights_only=True)['instrument_model']
    rows=json.loads((a.inputs/'samples.json').read_text())['samples']
    preds={(r['sequence_id'],r['frame_index']):r['tensorrt_values17'] for r in map(json.loads,a.predictions.read_text().splitlines())}
    libs=[root/'reference/pose_display/native/build/libsurgical_overlay.so',root/'reference/pose_display/native/build/optimized/libsurgical_overlay.so']
    overlays={};cases=[];previews={};rng=np.random.default_rng(101)
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        for row in rows:
            key=json.dumps(row['calibration'],sort_keys=True)
            if key not in overlays:
                overlays[key]=[CppStereoOverlay(model,row['calibration'],library=path,geometry_log=True) for path in libs]
            pair=overlays[key]
            real=[np.asarray(Image.open(a.inputs/row['images'][eye]).convert('RGB')) for eye in ('left','right')]
            # Noise catches alpha/antialias rounding; solids catch cached-border errors.
            variants=[('real',real)]
            if row['category'] not in previews:
                variants += [('noise',[rng.integers(0,256,size=x.shape,dtype=np.uint8) for x in real]),
                             ('black',[np.zeros_like(x) for x in real]),('white',[np.full_like(x,255) for x in real])]
            for kind,images in variants:
                frame=StereoFrame(f"{row['sequence_id']}/{row['frame_index']:06d}",row['timestamp_ns'],*images)
                pred=PosePrediction(frame.frame_id,frame.timestamp_ns,np.asarray(preds[row['sequence_id'],row['frame_index']]))
                images_out=[renderer.render(frame,pred)[0] for renderer in pair]
                diff=np.abs(images_out[0].astype(np.int16)-images_out[1].astype(np.int16))
                assert pair[0].compute_geometry(pred.values17)==pair[1].compute_geometry(pred.values17)
                entry={'category':row['category'],'frame':frame.frame_id,'background':kind,
                       'max_channel_difference':int(diff.max()),'changed_channels':int(np.count_nonzero(diff))}
                cases.append(entry)
                if kind=='real' and row['category'] not in previews:
                    filename=row['category']+'.jpg'
                    Image.fromarray(np.concatenate(images_out,axis=0)).save(a.output/filename,quality=95)
                    previews[row['category']]=filename
        passed=max(x['max_channel_difference'] for x in cases)<=1
        report={'passed':passed,'cases':cases,'previews':previews,
                'max_channel_difference':max(x['max_channel_difference'] for x in cases),
                'geometry_identical':True,'library_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in libs]}
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('cases','previews')},indent=2))
        if not passed:raise SystemExit('Overlay changed beyond allowed rounding')
    finally:
        for pair in overlays.values():
            for o in pair:o.close()
if __name__=='__main__':main()
