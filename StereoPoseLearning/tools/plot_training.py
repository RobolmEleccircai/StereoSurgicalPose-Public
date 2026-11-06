"""Regenerate loss/validation figures from actual logs; missing history stays missing."""
import argparse,json,time,os,math
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

def plot(run,logs,through_epoch=None):
 rows=read_rows(run/'history.jsonl')
 if through_epoch is not None:rows=[r for r in rows if r['epoch']+1<=through_epoch]
 if not rows:return
 supplements={r['epoch']:r for r in read_rows(run/'validation_loss.jsonl')}
 rows=[dict(r, **{k:v for k,v in supplements.get(r['epoch'],{}).items() if k.startswith('validation_loss')}) for r in rows]
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 epochs=[r['epoch']+1 for r in rows]
 axs[0,0].plot(epochs,[r['training_loss'] for r in rows],'o-');axs[0,0].set(title='Mean training loss per epoch',xlabel='Epoch',ylabel='Weighted loss',yscale='log')
 axs[0,1].plot(epochs,[r['validation_tip_mm_equal_category'] for r in rows],'o-',color='tab:orange');axs[0,1].set(title='Validation tip error (available categories equally weighted)',xlabel='Epoch',ylabel='3D error (mm)')
 progress=[r for path in logs for r in read_rows(path) if r.get('event')=='progress' and r['epoch']<=max(epochs)]
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
 temporary=plots/'training_curves.tmp.png';fig.savefig(temporary,dpi=150)
 temporary.replace(plots/'diagnostics.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
 ax.plot(epochs,[r['training_loss'] for r in rows],'o-',label='Training loss')
 validation=[r.get('validation_loss',float('nan')) for r in rows]
 ax.plot(epochs,validation,'s-',label='Validation loss')
 baseline_file=run/'baseline_validation_loss.json'
 baseline_loss=json.loads(baseline_file.read_text()) if baseline_file.exists() else {}
 baseline_epoch=baseline_loss.get('source_epoch')
 baseline_value=baseline_loss.get('validation_loss')
 ticks=list(epochs)
 if (baseline_loss.get('new_objective') is True
     and isinstance(baseline_epoch,int) and baseline_epoch<=max(epochs)
     and isinstance(baseline_value,(int,float)) and math.isfinite(baseline_value)):
  # A rescored frozen checkpoint is a validation reference, not a fabricated
  # training epoch. Never splice losses measured under the old objective.
  ax.plot([baseline_epoch,max(epochs)],[baseline_value,baseline_value],':',color='tab:orange',
          label=f'Epoch {baseline_epoch} rescored with current loss')
  ax.scatter([baseline_epoch],[baseline_value],marker='D',facecolors='none',
             edgecolors='tab:orange',zorder=4)
  ticks=sorted(set(ticks+[baseline_epoch]))
 ax.set(title='Training and validation total loss per epoch',xlabel='Epoch',ylabel='Weighted total loss')
 ax.set_xticks(ticks);ax.grid(alpha=.25);ax.legend()
 missing=[str(r['epoch']+1) for r in rows if 'validation_loss' not in r]
 if missing:fig.supxlabel('Validation loss unavailable for epochs: '+', '.join(missing),fontsize=9)
 temporary=plots/'loss.tmp.png';fig.savefig(temporary,dpi=150)
 temporary.replace(plots/'training_curves.png');plt.close(fig)
 (plots/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
 if baseline_loss:
  (plots/'baseline_validation_loss.json').write_text(json.dumps(baseline_loss,indent=2)+'\n')
 weighted=sorted({k for r in rows for k in r.get('training_metrics',{}) if k.startswith('weighted_loss/')})
 if weighted:
  fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
  for key in weighted:ax.plot(epochs,[r.get('training_metrics',{}).get(key,float('nan')) for r in rows],label=key.split('/',1)[1])
  ax.set(title='Weighted training loss contributions per epoch',xlabel='Epoch',ylabel='Contribution to total loss',yscale='symlog');ax.grid(alpha=.2);ax.legend(ncol=3)
  fig.savefig(plots/'weighted_loss_components.png',dpi=150);plt.close(fig)
 angles=['wrist_pitch_deg','wrist_yaw_deg','jaw_half_deg']
 if any(k in r.get('training_metrics',{}) for k in angles for r in rows):
  config=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
  baseline_path=config.get('training',{}).get('comparison_report')
  baseline=json.loads(Path(baseline_path).read_text()) if baseline_path and Path(baseline_path).exists() else {}
  fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
  for ax,key in zip(axs,angles):
   ax.plot(epochs,[r.get('training_metrics',{}).get(key,float('nan')) for r in rows],label='Training windows')
   ax.plot(epochs,[r.get('validation_metrics',{}).get(key,float('nan')) for r in rows],label='Validation (category equal)')
   if key in baseline.get('category_equal_mean',{}):
    ax.axhline(baseline['category_equal_mean'][key],color='gray',linestyle='--',label=config.get('training',{}).get('comparison_label','Frozen original baseline'))
   ax.set(title=key,xlabel='Epoch',ylabel='MAE (degrees)');ax.grid(alpha=.2);ax.legend()
  fig.savefig(plots/'joint_angle_errors.png',dpi=150);plt.close(fig)
  if config.get('training',{}).get('comparison_history') and all('validation_loss' in r for r in rows):
   if config.get("model", {}).get("foreground_parts", False):
    from tools.compare_parts import update
   else:
    from tools.compare_local4 import update
   update(run)


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--log',type=Path,action='append',default=[]);p.add_argument('--watch-pid',type=int);a=p.parse_args()
 while True:
  plot(a.run,a.log)
  if not a.watch_pid:break
  try:os.kill(a.watch_pid,0)
  except ProcessLookupError:break
  time.sleep(20)
if __name__=='__main__':main()
