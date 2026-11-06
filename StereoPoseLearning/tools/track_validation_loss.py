"""Supplement a running legacy job without interrupting its optimizer state."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stereo_pose.runtime import load_model, evaluate_loss
from stereo_pose.geometry import FixedKinematics
from tools.plot_training import plot, read_rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--exit-file',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    archive=a.run/'loss_checkpoints';archive.mkdir(exist_ok=True)
    signatures={}
    while True:
        history=read_rows(a.run/'history.jsonl')
        done={r['epoch'] for r in read_rows(a.run/'validation_loss.jsonl')}
        history_epochs={r['epoch'] for r in history}
        for name in ('best.pt','last.pt'):
            source=a.run/name
            if not source.exists():continue
            signature=(source.stat().st_ino,source.stat().st_mtime_ns)
            if signatures.get(name)==signature:continue
            # Opening once reads an immutable checkpoint inode despite atomic replacement.
            with source.open('rb') as handle:
                saved=torch.load(handle,map_location='cpu',weights_only=True)
            epoch=saved['epoch']
            if epoch not in history_epochs or epoch in done:continue
            checkpoint=archive/f'epoch_{epoch+1:03d}.pt'
            if not checkpoint.exists():torch.save(saved,checkpoint)
            signatures[name]=signature
            del saved
        if not a.exit_file.exists():
            time.sleep(10)
            continue
        # Avoid competing with training for the laptop's 8 GB GPU memory.
        for checkpoint in sorted(archive.glob('epoch_*.pt')):
            epoch=int(checkpoint.stem.split('_')[-1])-1
            if epoch in done:continue
            device=torch.device('cuda')
            model,saved=load_model(checkpoint,device)
            result=evaluate_loss(model,FixedKinematics(saved['instrument_model']).to(device),manifest,saved['config'],device)
            result.update(epoch=epoch,source_checkpoint=str(checkpoint))
            with (a.run/'validation_loss.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print(json.dumps(result),flush=True)
            done.add(epoch)
            del model,saved
            torch.cuda.empty_cache()
            plot(a.run,[a.run/'progress.jsonl'])
        plot(a.run,[a.run/'progress.jsonl'])
        break

if __name__=='__main__':main()
