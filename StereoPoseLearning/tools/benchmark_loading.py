import sys,time,json,argparse,itertools,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from stereo_pose.data import StereoDataset
from stereo_pose.loading import make_loader

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 reports=[];reference=None
 for workers in (0,4):
  ds=StereoDataset(a.manifest,'train',window=8,image_size=(360,640))
  settings={'batch_size':1,'workers':workers,'pin_memory':True,'prefetch_factor':2,'persistent_workers':True}
  loader=make_loader(torch.utils.data.Subset(ds,range(24)),settings); iterator=iter(loader);timings=[]
  for i in range(24):
   started=time.perf_counter();batch=next(iterator);timings.append(time.perf_counter()-started)
   if i==0:
    if reference is None:reference={k:v.clone() for k,v in batch.items() if isinstance(v,torch.Tensor)}
    else:
     for k,v in reference.items():torch.testing.assert_close(batch[k],v,rtol=0,atol=0)
    assert batch['images'].is_pinned()
  try:next(iterator);raise AssertionError('Unexpected extra sample')
  except StopIteration:pass
  reports.append({'workers':workers,'first_batch_seconds':timings[0],'steady_mean_batch_seconds':sum(timings[4:])/20,'same_tensors':True,'pinned':True})
  del iterator,loader,ds;gc.collect()
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
if __name__=='__main__':main()
