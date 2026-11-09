"""Check the fused FP64 ray calculation against the unchanged ATen formula."""
import argparse
import json
from pathlib import Path
import sys
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'native/libtorch/build/optimized'))
    import stereo_libtorch_opt as native
    bundle=json.loads((args.inputs/'samples.json').read_text())
    profiles={json.dumps(row['calibration'],sort_keys=True):row['calibration'] for row in bundle['samples']}
    cases=[]
    with torch.inference_mode():
        for profile in profiles.values():
            for eye in ('left','right'):
                K=torch.tensor(profile[eye]['K'],dtype=torch.float32,device='cuda')
                # Test source-camera coordinates, including all image borders.
                uv=torch.cartesian_prod(torch.tensor([0.,.5,320.,640.,960.,1279.],device='cuda'),
                                        torch.tensor([0.,.5,180.,360.,540.,719.],device='cuda'))
                uv=uv.repeat(5,1);n=len(uv)
                z=torch.tensor([10.,30.,100.,300.,1000.],device='cuda').repeat_interleave(n//5).view(n,1)
                kk=K.expand(n,3,3).contiguous()
                for kind,d in [('zero',[0.]*5),('actual',profile[eye]['D'])]:
                    dd=torch.tensor(d,dtype=torch.float32,device='cuda').expand(n,5).contiguous()
                    inverse=torch.linalg.inv_ex(kk.double())[0].contiguous()
                    a=native.position_reference(uv,z,kk,dd)
                    b=native.position_cuda(uv,z,inverse,dd.double())
                    diff=(a-b).norm(dim=-1)
                    cases.append({'eye':eye,'distortion':kind,'points':n,'max_mm':float(diff.max()),
                                  'finite':bool(torch.isfinite(b).all())})
    passed=all(x['finite'] and x['max_mm']<=.0001 for x in cases)
    result={'passed':passed,'threshold_mm':.0001,'points':sum(x['points'] for x in cases),
            'max_mm':max(x['max_mm'] for x in cases),'cases':cases}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if not passed:raise SystemExit('Position comparison failed')
if __name__=='__main__':main()
