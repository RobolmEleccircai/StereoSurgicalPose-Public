"""Draw saved test predictions without substituting label poses or rerendering."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
import numpy as np
from PIL import Image
import torch
from pose_display.pipeline import StereoFrame,PosePrediction,StereoOverlay

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--report',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 manifest=json.loads(Path(a.manifest).read_text());report=json.loads(Path(a.report).read_text())
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 records={(r['sequence_id'],r['frame_index']):r for r in manifest['records'] if r['split']=='test'}
 rows=report['frames_detail'];out=Path(a.output);out.mkdir(parents=True,exist_ok=True);index=[]
 for seq in sorted({r['sequence_id'] for r in rows}):
  seqrows=[r for r in rows if r['sequence_id']==seq]
  # Fixed middle/last frames; never select by estimated error.
  for row in (seqrows[len(seqrows)//2],seqrows[-1]):
   r=records[(seq,row['frame_index'])];label=json.loads(Path(r['label_path']).read_text())
   images=[np.asarray(Image.open(r['image_paths'][e]).convert('RGB')) for e in ('left','right')]
   cameras={'image':{'width':images[0].shape[1],'height':images[0].shape[0],'space':'raw_distorted'}}
   for e in ('left','right'):cameras[e]={k:label['eyes'][e][k] for k in ('K','D')}
   fid=f"{seq}/{row['frame_index']:06d}"
   frame=StereoFrame(fid,row['timestamp_ns'],*images)
   pred=PosePrediction(fid,row['timestamp_ns'],row['prediction_values17'],source='prediction',visibility=None,visibility_source='unknown')
   overlay=StereoOverlay(saved['instrument_model'],cameras)
   canvas,metadata=overlay.render(frame,pred)
   name=f"{seq}_{row['frame_index']:06d}"
   Image.fromarray(canvas).save(out/(name+'.jpg'),quality=95)
   metadata['evaluation_metrics']=row;metadata['checkpoint_epoch']=saved['epoch']+1
   (out/(name+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
   index.append({'image':name+'.jpg','sequence':seq,'frame':row['frame_index'],'tip_mm':row['tip_mm'],'rotation_deg':row['rotation_deg']})
 (out/'index.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps(index))
if __name__=='__main__':main()
