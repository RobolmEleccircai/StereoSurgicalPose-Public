"""CPU output-angle gradient diagnostic; never updates network parameters."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser()
    for n in ['root','manifest','checkpoint','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.root))
    from stereo_pose.runtime import load_model,read_json
    from stereo_pose.data import StereoDataset,collate_samples
    from stereo_pose.geometry import FixedKinematics
    from stereo_pose.losses import PoseLoss
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    model,saved=load_model(a.checkpoint,torch.device('cpu'));model.requires_grad_(False)
    kin=FixedKinematics(saved['instrument_model']).eval()
    ds=StereoDataset(read_json(a.manifest),'validation',window=1,image_size=tuple(saved['config']['image_size']))
    selected=[]
    for category in sorted({r['category'] for r in ds.records}):
        indices=[i for i,r in enumerate(ds.records) if r['category']==category]
        selected.extend(indices[i] for i in [len(indices)//4,3*len(indices)//4])
    terms=['joints','geometry','part_rotation','projection','landmark_consistency','depth']
    losses={term:PoseLoss(kin,{k:(saved['config']['loss_weights'].get(k,0) if k==term else 0) for k in PoseLoss.DEFAULT_WEIGHTS},**saved['config']['loss_scales']) for term in terms}
    rows=[];started=time.monotonic()
    for i in selected:
        b=collate_samples([ds[i]])
        with torch.no_grad():pred,_=model(b['images'],b['K'],b['D'],None)
        pred={k:v.detach().float() for k,v in pred.items()}
        err=(pred['joints']-b['joints']).reshape(3)
        row=dict(category=ds.records[i]['category'],frame_index=ds.records[i]['frame_index'],
                 predicted_angles_deg=torch.rad2deg(pred['joints']).reshape(3).tolist(),
                 truth_angles_deg=torch.rad2deg(b['joints']).reshape(3).tolist(),
                 error_deg=torch.rad2deg(err).tolist(),
                 teacher_confidence=pred['landmark_confidence'][0,0].tolist(),
                 in_frame=b['heatmap_valid'][0,0].bool().tolist())
        pnorm=(pred['joints']-model.joint_lower)/(model.joint_upper-model.joint_lower)
        row['sigmoid_slope_rad_per_logit']=((model.joint_upper-model.joint_lower)*pnorm*(1-pnorm)).reshape(3).tolist()
        for variant in ['predicted_C_R','oracle_C_R']:
            candidate=dict(pred);candidate['joints']=pred['joints'].clone().requires_grad_(True)
            if variant=='oracle_C_R':candidate.update(C=b['C'],R=b['R'])
            record={};total=torch.zeros(3)
            for term in terms:
                loss,_=losses[term](candidate,b)
                grad=torch.autograd.grad(loss,candidate['joints'])[0].reshape(3)
                assert torch.isfinite(grad).all()
                record[term]={'weighted_loss':float(loss.detach()),'gradient_per_radian':grad.tolist(),
                              'opposes_truth_direction':((grad*err<0)&(torch.rad2deg(err).abs()>.1)&(grad.abs()>1e-8)).tolist()}
                total+=grad.detach()
            record['total_output_angle_terms']={'gradient_per_radian':total.tolist(),
               'opposes_truth_direction':((total*err<0)&(torch.rad2deg(err).abs()>.1)&(total.abs()>1e-8)).tolist()}
            row[variant]=record
        rows.append(row)
    summary={}
    for variant in ['predicted_C_R','oracle_C_R']:
        summary[variant]={}
        for term in terms+['total_output_angle_terms']:
            grads=np.array([r[variant][term]['gradient_per_radian'] for r in rows])
            summary[variant][term]={
               'mean_abs_gradient_per_radian':np.abs(grads).mean(0).tolist(),
               'opposes_truth_direction_count':np.array([r[variant][term]['opposes_truth_direction'] for r in rows]).sum(0).tolist()}
    out=dict(checkpoint=str(a.checkpoint),checkpoint_epoch=saved['epoch']+1,
             checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
             protocol='14 fixed validation stereo pairs (quarter and three-quarter frame each of 7 categories), CPU FP32 single-pair hidden=None. Frozen network. Weighted derivatives with respect to predicted output joint angles, not shared network parameters. Oracle C/R leaves other predicted joint errors and detached heatmap teacher unchanged.',
             terms_order=terms,angle_order=['wrist_pitch','wrist_yaw','jaw_half'],
             direction_threshold_deg=.1,frames=len(rows),test_used=False,model_updated=False,gpu_used=False,
             seconds=time.monotonic()-started,summary=summary,rows=rows)
    (a.output/'report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
