"""Read-only CPU comparison of full streaming and 8-frame hidden resets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.root))
    from stereo_pose.runtime import load_model,read_json
    from stereo_pose.data import StereoDataset,collate_samples
    from stereo_pose.geometry import FixedKinematics
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    frozen=a.output/'frozen_best.pt';shutil.copy2(a.checkpoint,frozen)
    sha=digest(frozen)
    model,saved=load_model(frozen,torch.device('cpu'))
    model.requires_grad_(False)
    kin=FixedKinematics(saved['instrument_model']).eval()
    ds=StereoDataset(read_json(a.manifest),'validation',window=1,image_size=tuple(saved['config']['image_size']))
    selected=[(i,r) for i,r in enumerate(ds.records) if r['category'] in ('lighting','occlusion')]
    assert len(selected)==172,len(selected)
    rows=[];hidden_full=None;hidden_reset=None;previous=None;offset=0;start=time.monotonic()
    for index,record in selected:
        seq=(record['sequence_id'],record['trajectory_group_id'])
        if seq!=previous:
            hidden_full=hidden_reset=None;offset=0
        if offset%8==0:hidden_reset=None
        batch=collate_samples([ds[index]])
        row=dict(category=record['category'],sequence_id=record['sequence_id'],frame_index=record['frame_index'],
                 offset=offset,reset_boundary=offset>0 and offset%8==0,
                 truth_joints_deg=torch.rad2deg(batch['joints'][0,0]).tolist(),
                 truth_C=batch['C'][0,0].tolist())
        with torch.no_grad():
            pred,hidden_full=model(batch['images'],batch['K'],batch['D'],hidden_full)
            reset,hidden_reset=model(batch['images'],batch['K'],batch['D'],hidden_reset)
            for mode,value in [('stream',pred),('reset8',reset)]:
                geometry=kin(value['C'],value['R'],value['joints'])
                row[mode]=dict(joints_deg=torch.rad2deg(value['joints'][0,0]).tolist(),
                               C=value['C'][0,0].tolist(),
                               values17=model.export_values(value)[0,0].tolist(),
                               tip_mm=float((geometry['points'][...,3:5,:]-batch['points'][...,3:5,:]).norm(dim=-1).mean()))
        rows.append(row);previous=seq;offset+=1
        if len(rows)%20==0:print(json.dumps(dict(processed=len(rows),seconds=time.monotonic()-start)),flush=True)

    def summarize(rr):
        if not rr:return {'frames':0}
        truth=np.array([r['truth_joints_deg'] for r in rr])
        result={'frames':len(rr)}
        for mode in ('stream','reset8'):
            pred=np.array([r[mode]['joints_deg'] for r in rr]);error=pred-truth
            result[mode]={'angle_mae_deg':np.abs(error).mean(0).tolist(),
                          'angle_bias_deg':error.mean(0).tolist(),
                          'tip_mm':float(np.mean([r[mode]['tip_mm'] for r in rr]))}
        result['mean_absolute_prediction_difference_deg']=np.abs(np.array([r['stream']['joints_deg'] for r in rr])-np.array([r['reset8']['joints_deg'] for r in rr])).mean(0).tolist()
        return result

    report={'checkpoint_epoch':saved['epoch']+1,'checkpoint_sha256':sha,'manifest_sha256':digest(a.manifest),
            'protocol':'CPU FP32, 2 threads; same frozen model/same actual frames. Full causal stream versus resetting hidden every 8 frames. Reset8 chunks are diagnostic, not the overlapping-window validation-loss aggregation.',
            'angles_order':['wrist_pitch','wrist_yaw','jaw_half'],
            'categories':{},'seconds':time.monotonic()-start,'test_used':False,'model_updated':False,'gpu_used':False}
    for category in ('lighting','occlusion'):
        rr=[r for r in rows if r['category']==category]
        result={ 'all':summarize(rr),'first8':summarize(rr[:8]),'after8':summarize(rr[8:]),
                 'last32':summarize(rr[-32:]),'blocks':[summarize(rr[k:k+8]) for k in range(0,len(rr),8)]}
        for mode in ('stream','reset8'):
            values=np.array([r[mode]['joints_deg'] for r in rr]);truth=np.array([r['truth_joints_deg'] for r in rr])
            steps=np.diff(values,axis=0);error_steps=np.diff(values-truth,axis=0)
            boundaries=np.array([r['reset_boundary'] for r in rr[1:]])
            C=np.array([r[mode]['C'] for r in rr]);Cgt=np.array([r['truth_C'] for r in rr])
            result.setdefault('discontinuities',{})[mode]={
                'max_abs_angle_step_deg':np.abs(steps).max(0).tolist(),
                'max_abs_angle_error_step_deg':np.abs(error_steps).max(0).tolist(),
                'reset_boundary_mean_abs_angle_error_step_deg':np.abs(error_steps[boundaries]).mean(0).tolist(),
                'reset_boundary_max_abs_angle_error_step_deg':np.abs(error_steps[boundaries]).max(0).tolist(),
                'max_C_error_step_mm':float(np.linalg.norm(np.diff(C-Cgt,axis=0),axis=-1).max())}
        assert result['first8']['mean_absolute_prediction_difference_deg']==[0.,0.,0.]
        report['categories'][category]=result
    report['checkpoint_unchanged']=digest(frozen)==sha
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'frames.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'epoch':saved['epoch']+1,'seconds':report['seconds'],'summary':{k:v['all'] for k,v in report['categories'].items()}}),flush=True)


if __name__=='__main__':main()
