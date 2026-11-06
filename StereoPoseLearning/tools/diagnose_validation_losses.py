"""Read-only CPU loss attribution for immutable checkpoint snapshots.

Reuses the production loader, model, 8-frame windows, targets and PoseLoss.
CPU arithmetic may differ from CUDA BF16; report that discrepancy explicitly.
Never imports a training entry point or opens existing artifacts for writing.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from stereo_pose.data import StereoDataset, collate_samples
from stereo_pose.runtime import load_model, forward_batch, verify_manifest
from stereo_pose.geometry import FixedKinematics, project_brown, rotation_6d_to_matrix
from stereo_pose.losses import PoseLoss


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sliced(values, indices, batch_size):
    return {k: v[indices] if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == batch_size
            else [v[i] for i in indices] if k == 'metadata' else v for k,v in values.items()}


def encoded_frames(model, ds, raw_cache, dtype, frame_count=None):
    """Cache only frame-independent work; all three GRUs rerun each window."""
    modules=[model.temporal,model.foreground_parts_net.wrist_branch.temporal,
             model.foreground_parts_net.jaw_branch.temporal]
    cache={};captured={};handles=[]
    for i,m in enumerate(modules):
        def capture(_module,args,i=i):captured[i]=args[0].detach().clone()
        handles.append(m.register_forward_pre_hook(capture))
    try:
        total=len(ds.records) if frame_count is None else frame_count
        for begin in range(0,total,32):
            ids=list(range(begin,min(begin+32,total)))
            for i in ids:
                if i not in raw_cache:raw_cache[i]=ds.load_frame(ds.records[i])
            # Preserve the 4x8 production spatial-kernel batch size, including
            # the encoder tail; padded features are discarded, never evaluated.
            padded=ids+[ids[-1]]*(32-len(ids))
            sample={k:torch.stack([raw_cache[i][k] for i in padded]) for k in ds.tensor_keys}
            sample['metadata']=[raw_cache[i]['metadata'] for i in padded]
            batch=collate_samples([sample])
            ctx=torch.autocast('cpu',dtype=torch.bfloat16) if dtype=='bf16' else nullcontext()
            with ctx:pred,_=forward_batch(model,batch)
            temporal=[captured[i].reshape(1,2,len(padded),-1).permute(0,2,1,3)[0] for i in range(3)]
            for j,frame in enumerate(ids):
                cache[frame]={k:v[0,j].detach().clone() for k,v in pred.items() if k not in ('C','R','joints')}
                cache[frame]['_temporal']=[v[j].clone() for v in temporal]
            print(json.dumps({'event':'encoded_frames','frames':len(cache),'total':total}),flush=True)
    finally:
        for h in handles:h.remove()
    return cache


def cached_prediction(model, encoded, windows):
    b,t=len(windows),len(windows[0])
    def sequence(i):
        return torch.stack([torch.stack([encoded[j]['_temporal'][i] for j in window])
                            for window in windows]).permute(0,2,1,3).reshape(b*2,t,-1)
    states,_=model.temporal(sequence(0),None)
    states=states.reshape(b,2,t,-1).permute(0,2,1,3)
    pose=torch.stack([model.pose_heads[e](states[:,:,e]) for e in range(2)],2).float()
    z=F.softplus(pose[...,2:3])*model.config.get('depth_scale_mm',100.)+.001
    result={'C':torch.cat((pose[...,:2]*z,z),-1),'R':rotation_6d_to_matrix(pose[...,3:])}
    logits=[]
    for i,branch in enumerate((model.foreground_parts_net.wrist_branch,model.foreground_parts_net.jaw_branch),1):
        state,_=branch.temporal(sequence(i),None)
        state=state.reshape(b,2,t,-1).permute(0,2,1,3)
        logits.append(branch.head(state.flatten(-2)).float())
    result['joints']=model.joint_lower+(model.joint_upper-model.joint_lower)*torch.cat(logits,-1).sigmoid()
    for k in encoded[windows[0][0]]:
        if k!='_temporal':result[k]=torch.stack([torch.stack([encoded[j][k] for j in window]) for window in windows])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--checkpoint',action='append',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--dtype',choices=['fp32','bf16'],default='bf16')
    p.add_argument('--limit-batches',type=int)
    p.add_argument('--cache-frame-features',action='store_true')
    a=p.parse_args()
    os.nice(19)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    # Explicit attention path works with CPU autocast and does not alter weights.
    torch.backends.mha.set_fastpath_enabled(False)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(a.manifest).read_text())
    manifest_hash=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    cache={};ds=None
    for checkpoint in a.checkpoint:
        start=time.perf_counter()
        model,saved=load_model(checkpoint,torch.device('cpu'))
        cfg=saved['config'];epoch=saved['epoch']+1
        if a.cache_frame_features and not hasattr(model, 'pose_heads'):
            raise ValueError('The feature cache is specific to legacy_c. Omit --cache-frame-features for the current model; full production forward is supported.')
        if ds is not None and (tuple(ds.image_size) != tuple(cfg['image_size']) or ds.window != cfg['window']):
            raise ValueError('Compare checkpoints with identical input size and temporal window')
        verify_manifest(manifest,saved['instrument_model'])
        if ds is None:
            ds=StereoDataset(manifest,'validation',window=cfg['window'],image_size=tuple(cfg['image_size']),
                             include_hires=cfg['model'].get('foreground_parts',False))
        kin=FixedKinematics(saved['instrument_model'])
        criterion=PoseLoss(kin,cfg['loss_weights'],**cfg.get('loss_scales',{}))
        batch_size=cfg['training']['batch_size']
        sums=defaultdict(float);counts=defaultdict(int);by_cat=defaultdict(lambda:defaultdict(float));rows=[]
        print(json.dumps({'event':'start','epoch':epoch,'windows':len(ds),'dtype':a.dtype}),flush=True)
        with torch.inference_mode():
            frame_count=(max(max(w) for w in ds.windows[:a.limit_batches*batch_size])+1) if a.limit_batches else None
            encoded=encoded_frames(model,ds,cache,a.dtype,frame_count) if a.cache_frame_features else None
            cache_checks=[]
            for batch_index,begin in enumerate(range(0,len(ds),batch_size)):
                if a.limit_batches is not None and batch_index>=a.limit_batches:break
                samples=[]
                for wi in range(begin,min(begin+batch_size,len(ds))):
                    ids=ds.windows[wi]
                    for i in ids:
                        if i not in cache:cache[i]=ds.load_frame(ds.records[i])
                    sample={k:torch.stack([cache[i][k] for i in ids]) for k in ds.tensor_keys}
                    sample['metadata']=[cache[i]['metadata'] for i in ids]
                    samples.append(sample)
                batch=collate_samples(samples);n=len(samples)
                ctx=torch.autocast('cpu',dtype=torch.bfloat16) if a.dtype=='bf16' else nullcontext()
                with ctx:
                    if encoded is None or n!=batch_size:pred,_=forward_batch(model,batch)
                    else:pred=cached_prediction(model,encoded,ds.windows[begin:begin+n])
                pred={k:v.float() for k,v in pred.items()}
                loss,metrics=criterion(pred,batch)
                if encoded is not None and batch_index in (0,len(range(0,len(ds),batch_size))-1):
                    ctx=torch.autocast('cpu',dtype=torch.bfloat16) if a.dtype=='bf16' else nullcontext()
                    with ctx:reference,_=forward_batch(model,batch)
                    reference={k:v.float() for k,v in reference.items()}
                    ref_loss,_=criterion(reference,batch)
                    check={'batch':batch_index,'loss_difference':float(loss-ref_loss),
                           'max_abs_by_output':{k:float((pred[k]-reference[k]).abs().max()) for k in pred}}
                    cache_checks.append(check)
                    # Numerical batching differences are recorded; reject material pose/loss drift.
                    if abs(check['loss_difference'])>1e-4 or check['max_abs_by_output']['C']>.1 or check['max_abs_by_output']['joints']>1e-3:
                        raise AssertionError('Frame cache changes output materially: '+str(check))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite diagnostic loss')
                for k,v in metrics.items():sums[k]+=float(v)*n
                counts['all']+=n
                categories=[m[-1]['category'] for m in batch['metadata']]
                for c in sorted(set(categories)):
                    idx=[i for i,x in enumerate(categories) if x==c]
                    pp,bb=sliced(pred,idx,n),sliced(batch,idx,n)
                    _,mm=criterion(pp,bb)
                    for k,v in mm.items():by_cat[c][k]+=float(v)*len(idx)
                    counts[c]+=len(idx)
                # Per-window diagnostics are separate from the exact batched total.
                for j in range(n):
                    pp,bb=sliced(pred,[j],n),sliced(batch,[j],n)
                    _,mm=criterion(pp,bb)
                    gtuv=project_brown(bb['points'][...,:5,:],bb['K'],bb['D'])
                    valid=bb['heatmap_valid'].bool()
                    pixel=(pp['landmark_uv']-gtuv).norm(dim=-1)
                    keypoint_error=float(pixel[valid].mean()) if valid.any() else None
                    mask=pp['mask_logits'].sigmoid()>=.5;truth=bb['masks'].bool()
                    iou=float((mask&truth).sum()/(mask|truth).sum().clamp_min(1))
                    m=bb['metadata'][-1][-1]
                    rows.append({'category':m['category'],'sequence_id':m['sequence_id'],
                                 'end_frame':m['frame_index'],'overexposed':m['overexposed'],
                                 'tip_region_visibility':m.get('tip_region_visibility'),
                                 'image_paths':m['image_paths'],
                                 'metrics':{k:float(v) for k,v in mm.items()},
                                 'landmark_error_px_640':keypoint_error,'visible_mask_iou':iou})
                elapsed=time.perf_counter()-start
                if batch_index%5==0 or begin+n==len(ds):
                    progress={'event':'progress','epoch':epoch,'windows':counts['all'],'total':len(ds),
                              'elapsed_seconds':elapsed,'seconds_per_window':elapsed/counts['all']}
                    print(json.dumps(progress),flush=True);write(out/'progress.json',progress)
        global_metrics={k:v/counts['all'] for k,v in sums.items()}
        cats={c:{k:v/counts[c] for k,v in values.items()} for c,values in by_cat.items()}
        category_weighted=sum(cats[c]['loss/total']*counts[c] for c in cats)/counts['all']
        report={'epoch':epoch,'checkpoint':checkpoint,'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                'manifest_sha256':manifest_hash,'windows':counts['all'],'complete':counts['all']==len(ds),
                'device':'cpu','dtype':a.dtype,'batch_size':batch_size,'window':cfg['window'],
                'global_metrics':global_metrics,'category_metrics':cats,'window_counts':dict(counts),
                'per_window':rows,'elapsed_seconds':time.perf_counter()-start,
                'cached_frame_features':encoded is not None,'cache_equivalence_checks':cache_checks,
                'category_reaggregation_residual':category_weighted-global_metrics['loss/total'],
                'protocol':'Full sequential validation windows; same batched total loss formula. Category subsets are independently reduced for mixed boundary batches; residual reported. No augmentation. CPU recomputation, not an exact CUDA-BF16 replay.'}
        if hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()!=manifest_hash:raise RuntimeError('Manifest changed')
        write(out/f'epoch_{epoch:03d}.json',report)
        print(json.dumps({'event':'complete','epoch':epoch,'loss':global_metrics['loss/total'],'seconds':report['elapsed_seconds']}),flush=True)
        del pred,batch,model,saved,criterion,kin,encoded


if __name__=='__main__':main()
