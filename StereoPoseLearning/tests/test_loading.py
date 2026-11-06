import tempfile
from pathlib import Path
import torch
from test_data import save_label,save_plan
from stereo_pose.data import build_manifest,StereoDataset
from stereo_pose.loading import make_loader,device_batches

def test_worker_order_and_targets():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp)/'run';root.mkdir();save_plan(root,4)
  for i in range(4):save_label(root,i)
  manifest=build_manifest([root],Path(tmp)/'manifest.json')
  ds=StereoDataset(manifest,'train',window=2,image_size=(16,32))
  expected=list(make_loader(ds,{'batch_size':1,'workers':0}))
  actual=list(make_loader(ds,{'batch_size':1,'workers':2,'persistent_workers':False,'prefetch_factor':2}))
  for a,b in zip(actual,expected):
   assert a['metadata']==b['metadata']
   for k in a:
    if isinstance(a[k],torch.Tensor):torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)

def test_async_copy_preserves_order_and_values():
 if not torch.cuda.is_available():return
 batches=[{'x':torch.arange(100).float().add(i).pin_memory(),'id':i} for i in range(4)]
 actual=list(device_batches(batches,torch.device('cuda'),True))
 torch.cuda.synchronize()
 for a,b in zip(actual,batches):
  assert a['id']==b['id'];torch.testing.assert_close(a['x'].cpu(),b['x'],atol=0,rtol=0)
