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

def plot_loss(run,rows):
 """Separate linear scales show epoch changes without altering recorded losses."""
 plots=run/'plots';plots.mkdir(exist_ok=True)
 epochs=[r['epoch']+1 for r in rows]
 fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
 for ax,key,title,color in zip(axs,('training_loss','validation_loss'),
                              ('Training loss','Validation loss'),('tab:blue','tab:orange')):
  values=[r.get(key,float('nan')) for r in rows]
  ax.plot(epochs,values,'o-',color=color,linewidth=1.7,markersize=5)
  finite=[(e,v) for e,v in zip(epochs,values) if isinstance(v,(int,float)) and math.isfinite(v)]
  if finite:
   low=min(v for _,v in finite);high=max(v for _,v in finite)
   span=max(high-low,abs((high+low)/2)*.01,1e-10)
   ax.set_ylim(max(0,low-.15*span),high+.15*span)
   ax.set_title(f'{title} | latest {finite[-1][1]:.6f} | best {low:.6f}',loc='left')
  else:
   ax.set_title(title,loc='left');ax.text(.5,.5,'Not recorded',transform=ax.transAxes,ha='center')
  ax.set_ylabel('Weighted total loss')
  ax.ticklabel_format(axis='y',style='plain',useOffset=False)
  ax.grid(alpha=.25)
 axs[-1].set_xlabel('Epoch')
 if len(epochs)<=20:axs[-1].set_xticks(epochs)
 else:
  from matplotlib.ticker import MaxNLocator
  axs[-1].xaxis.set_major_locator(MaxNLocator(nbins=12,integer=True))
 fig.suptitle('Training and validation loss — independent zoomed y-axes')
 notes=['Linear scales; compare changes within each panel.']
 baseline_file=run/'baseline_validation_loss.json'
 baseline=json.loads(baseline_file.read_text()) if baseline_file.exists() else {}
 value=baseline.get('validation_loss');epoch=baseline.get('source_epoch')
 if baseline.get('new_objective') is True and isinstance(epoch,int) and isinstance(value,(int,float)) and math.isfinite(value):
  notes.append(f'Frozen epoch {epoch} validation reference: {value:.6f} (reference only, excluded from zoom limits).')
 missing=[str(r['epoch']+1) for r in rows if 'validation_loss' not in r]
 if missing:notes.append('Validation loss unavailable for epochs: '+', '.join(missing))
 fig.supxlabel('\n'.join(notes),fontsize=9)
 temporary=plots/'loss_zoom.tmp.png';fig.savefig(temporary,dpi=150)
 temporary.replace(plots/'training_curves.png');plt.close(fig)
 (plots/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
 if baseline:(plots/'baseline_validation_loss.json').write_text(json.dumps(baseline,indent=2)+'\n')

def plot(run,logs,through_epoch=None,loss_only=False):
 rows=read_rows(run/'history.jsonl')
 if through_epoch is not None:rows=[r for r in rows if r['epoch']+1<=through_epoch]
 if not rows:return
 supplements={r['epoch']:r for r in read_rows(run/'validation_loss.jsonl')}
 rows=[dict(r, **{k:v for k,v in supplements.get(r['epoch'],{}).items() if k.startswith('validation_loss')}) for r in rows]
 if loss_only:
  plot_loss(run,rows);return
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
 plot_loss(run,rows)
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
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--log',type=Path,action='append',default=[]);p.add_argument('--watch-pid',type=int);p.add_argument('--loss-only',action='store_true');a=p.parse_args()
 def signature():
  paths=[a.run/'history.jsonl',a.run/'validation_loss.jsonl',a.run/'baseline_validation_loss.json',a.run/'plots/training_curves.png']
  return tuple(x.stat().st_mtime_ns if x.exists() else None for x in paths)
 previous=None
 while True:
  # Also notice a running trainer replacing the chart using its already-imported
  # old plotting function. This watcher changes images only, not training state.
  current=signature()
  if not a.loss_only or current!=previous:
   plot(a.run,a.log,loss_only=a.loss_only);previous=signature()
  if not a.watch_pid:break
  try:os.kill(a.watch_pid,0)
  except ProcessLookupError:break
  time.sleep(20)
if __name__=='__main__':main()
