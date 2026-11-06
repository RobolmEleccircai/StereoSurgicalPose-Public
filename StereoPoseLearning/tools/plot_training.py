"""Regenerate loss/validation figures from actual logs; missing history stays missing."""
import argparse,json,time,os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_rows(path):
 rows=[]
 if not path.exists():return rows
 for line in path.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 return rows

def plot(run,logs):
 rows=read_rows(run/'history.jsonl')
 if not rows:return
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 epochs=[r['epoch']+1 for r in rows]
 axs[0,0].plot(epochs,[r['training_loss'] for r in rows],'o-');axs[0,0].set(title='Mean training loss per epoch',xlabel='Epoch',ylabel='Weighted loss',yscale='log')
 axs[0,1].plot(epochs,[r['validation_tip_mm_equal_category'] for r in rows],'o-',color='tab:orange');axs[0,1].set(title='Validation tip error (available categories equally weighted)',xlabel='Epoch',ylabel='3D error (mm)')
 progress=[r for path in logs for r in read_rows(path) if r.get('event')=='progress']
 for epoch in sorted({r['epoch'] for r in progress}):
  p=[r for r in progress if r['epoch']==epoch];axs[1,0].plot([r['step'] for r in p],[r['mean_loss'] for r in p],label=f'Epoch {epoch}')
 axs[1,0].set(title='Cumulative mean within each epoch (not per-step loss)',xlabel='Window step',ylabel='Weighted loss',yscale='log')
 if progress:axs[1,0].legend()
 keys=sorted({k for r in rows for k in r.get('training_metrics',{}) if k.startswith('loss/') and k!='loss/total'})
 for key in keys:
  selected=[r for r in rows if key in r.get('training_metrics',{})]
  axs[1,1].plot([r['epoch']+1 for r in selected],[r['training_metrics'][key] for r in selected],'o-',label=key[5:])
 axs[1,1].set(title='Loss components (recorded epochs only; before weights)',xlabel='Epoch',ylabel='Loss')
 axs[1,1].set_yscale('symlog',linthresh=1e-5)
 if keys:axs[1,1].legend(fontsize=7,ncol=2)
 else:axs[1,1].text(.5,.5,'Component history starts after loader upgrade',ha='center',transform=axs[1,1].transAxes)
 for ax in axs.flat:ax.grid(alpha=.2)
 plots=run/'plots';plots.mkdir(exist_ok=True)
 temporary=plots/'training_curves.tmp.png';fig.savefig(temporary,dpi=150);temporary.replace(plots/'training_curves.png');plt.close(fig)
 (plots/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--log',type=Path,action='append',default=[]);p.add_argument('--watch-pid',type=int);a=p.parse_args()
 while True:
  plot(a.run,a.log)
  if not a.watch_pid:break
  try:os.kill(a.watch_pid,0)
  except ProcessLookupError:break
  time.sleep(20)
if __name__=='__main__':main()
