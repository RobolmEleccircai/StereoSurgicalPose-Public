"""Compare real predictor state/reset behavior without ground-truth inputs."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--baseline-config',type=Path,required=True)
    p.add_argument('--candidate-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root),str(root/'reference')]
    from stereo_pose.libtorch_predict import build_predictor
    from pose_display import StereoFrame
    torch.set_num_threads(4)
    def build(path):
        c=json.loads(path.read_text());c['_config_dir']=str(path.resolve().parent);return build_predictor(c)
    base,candidate=build(args.baseline_config),build(args.candidate_config)
    rows=json.loads((args.inputs/'samples.json').read_text())['samples'][:8]
    records=[]
    def compare(name,frame,calibration,fail=False):
        results=[]
        for model in (base,candidate):
            try:
                pred=model(frame,calibration);results.append(pred.values17.copy())
            except ValueError:
                if not fail:raise
                results.append(None)
        if fail:
            assert results==[None,None]
            assert base.hidden is None and candidate.hidden is None
            records.append({'case':name,'rejected_and_reset':True});return
        delta=float(np.max(np.abs(results[0]-results[1])))
        hdelta=float((base.hidden-candidate.hidden).abs().max())
        assert np.allclose(results[0],results[1],atol=1e-5,rtol=1e-6)
        assert torch.allclose(base.hidden,candidate.hidden,atol=1e-5,rtol=1e-6)
        assert base.reset_count==candidate.reset_count
        records.append({'case':name,'values_max_abs':delta,'hidden_max_abs':hdelta,'reset_count':base.reset_count})
    for i,row in enumerate(rows):
        images=[np.asarray(Image.open(args.inputs/row['images'][eye]).convert('RGB')) for eye in ('left','right')]
        frame=StereoFrame(f"sequence/{i:06d}",round(i*1e9/60),*images)
        compare('continuous_'+str(i),frame,row['calibration'])
    base.reset();candidate.reset();compare('explicit_reset',frame,row['calibration'])
    compare('duplicate_timestamp',frame,row['calibration'])
    frame=StereoFrame('sequence/000008',frame.timestamp_ns+100000000,*images)
    compare('time_gap',frame,row['calibration'])
    frame=StereoFrame('other/000009',frame.timestamp_ns+16666667,*images)
    compare('sequence_change',frame,row['calibration'])
    changed=copy.deepcopy(row['calibration']);changed['left']['K'][0][0]*=1.01
    changed['right']['D'][0]+=.001
    frame=StereoFrame('other/000010',frame.timestamp_ns+16666667,*images)
    compare('calibration_change',frame,changed)
    bad=StereoFrame('other/000011',frame.timestamp_ns+16666667,images[0][:-1],images[1])
    compare('invalid_image',bad,changed,True)
    frame=StereoFrame('other/000012',bad.timestamp_ns+16666667,*images)
    compare('recovery',frame,changed)
    result={'passed':True,'cases':records,'candidate_execution':getattr(candidate.model,'actual_execution',candidate.execution_mode)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
