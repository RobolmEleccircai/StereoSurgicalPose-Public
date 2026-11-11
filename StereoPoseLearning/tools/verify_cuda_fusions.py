"""Adversarial operator comparisons; separate from task accuracy and timing."""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

def crop_cases(native):
    torch.manual_seed(1809)
    rgb=torch.randint(0,256,(2,3,720,1280),device='cuda',dtype=torch.uint8)
    random=torch.randn(2,1,90,160,device='cuda')*5
    masks={'random':random,'empty':torch.full_like(random,-100),'full':torch.full_like(random,100),
           'threshold':torch.full_like(random,torch.logit(torch.tensor(.2)).item())+torch.randn_like(random)*1e-6}
    axis=(torch.arange(160,device='cuda',dtype=torch.float32)+.5)*2/160-1
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    regular=torch.stack((x,y),-1)[None].repeat(2,2,1,1)
    grids={'regular':regular,'outside':regular*1.3,'edge':regular*.001+.999,
           'strided':regular[:,::2,::2]}
    records=[]
    with torch.inference_mode():
      for name,m in masks.items():
        for dilation in (0,1,2):
          expected=((F.interpolate(m.sigmoid(),(720,1280),mode='bilinear',align_corners=False)-.2)/.8).clamp(0,1)
          if dilation:expected=F.max_pool2d(expected,2*dilation+1,1,dilation)
          actual=native.mask_cuda(m.sigmoid(),.2,dilation)
          torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
          for grid_name,g in grids.items():
            a=native.crop_reference(rgb,m,g,.2,dilation);b=native.crop_cuda(rgb,m,g,.2,dilation)
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
            records.append({'mask':name,'dilation':dilation,'grid':grid_name,
                'mask_max_abs':float((expected-actual).abs().max()),'crop_max_abs':float((a-b).abs().max())})
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'native/libtorch/build/fused'))
    import stereo_libtorch_fused as native
    records=crop_cases(native)
    result={'passed':True,'crop_cases':records,'scope':'operator numerical agreement, not pose accuracy'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'cases':len(records),'crop_max_abs':max(r['crop_max_abs'] for r in records)}))
if __name__=='__main__':main()
