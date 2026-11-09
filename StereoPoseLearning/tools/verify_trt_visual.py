"""Compare the native TensorRT frontend to the preserved eager C++ model.

Seven categories, all saved consecutive pairs, independent recurrent states.
Metrics are numerical disagreement, not task error against ground truth.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch


def stats(x):
    a=np.asarray(x,dtype=float)
    return {'mean':float(a.mean()),'p95':float(np.percentile(a,95)),'max':float(a.max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--engine',type=Path)
    p.add_argument('--baseline-config',type=Path)
    p.add_argument('--candidate-config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--truth',type=Path,help='Optional hash-matched label values, used only after inference')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    project=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(project));sys.path.insert(0,str(project/'reference'))
    from stereo_pose.libtorch_predict import LibTorchPredictor
    from pose_display import StereoFrame,CppStereoOverlay,PosePrediction
    from pose_display.decoding import decode
    from pose_display.projection import project as project_points
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def configured(path, engine=None):
        if path:
            from stereo_pose.libtorch_predict import build_predictor
            config=json.loads(path.read_text());config['_config_dir']=str(path.resolve().parent)
            return build_predictor(config)
        return LibTorchPredictor(a.inputs/'best_val.pt',preprocessing='gpu',visual_engine=engine)
    if not a.candidate_config and not a.engine:raise ValueError('Provide candidate configuration or engine')
    original=configured(a.baseline_config)
    accelerated=configured(a.candidate_config,a.engine)
    bundle=json.loads((a.inputs/'samples.json').read_text())
    truths={}
    if a.truth:
        truths={(r['sequence_id'],r['frame_index']):r for r in json.loads(a.truth.read_text())['records']}
    previous=None;h0=h1=None;rows=[];layers={};overlays={};previews={}
    with torch.inference_mode():
        for number,row in enumerate(bundle['samples']):
            if row['sequence_id']!=previous:
                original.reset();accelerated.reset()
                h0=torch.zeros(3,2,256,device='cuda');h1=h0.clone()
            images=[np.asarray(Image.open(a.inputs/row['images'][e]).convert('RGB')) for e in ('left','right')]
            truth=truths.get((row['sequence_id'],row['frame_index']))
            if a.truth and truth is None:raise ValueError('Missing requested truth')
            if truth:
                for eye in ('left','right'):
                    with (a.inputs/row['images'][eye]).open('rb') as f:
                        assert hashlib.file_digest(f,'sha256').hexdigest()==truth['image_sha256'][eye]
            frame=StereoFrame(f"{row['sequence_id']}/{row['frame_index']:06d}",row['timestamp_ns'],*images)
            pred0=original(frame,row['calibration']);pred1=accelerated(frame,row['calibration'])
            inputs=original._preprocess(frame,row['calibration']);hires=original._hires
            v0,h0,d0=original.model.engine.forward(*inputs,h0,hires,True)
            v1,h1,d1=accelerated.model.engine.forward(*inputs,h1,hires,True)
            for name in ('spatial','mask_logits','heatmap_logits','landmark_uv','hidden'):
                t0,t1=(h0,h1) if name=='hidden' else (d0[name],d1[name])
                diff=(t1-t0).float()
                if not torch.isfinite(t1).all():raise ValueError('Non-finite '+name)
                r=layers.setdefault(name,{'max_abs':0.,'rms':[],'reference_rms':[]})
                r['max_abs']=max(r['max_abs'],float(diff.abs().max()))
                r['rms'].append(float(diff.square().mean().sqrt()))
                r['reference_rms'].append(float(t0.float().square().mean().sqrt()))
            x,y=pred0.values17,pred1.values17
            if not np.allclose(y,v1.flatten().cpu().numpy(),atol=1e-5,rtol=1e-6):
                raise ValueError('Runtime and diagnostic forward disagree')
            if not np.isfinite(y).all():raise ValueError('Non-finite pose')
            xx,yy=decode(x,original.instrument_model),decode(y,original.instrument_model)
            record={'sequence_id':row['sequence_id'],'category':row['category'],'frame_index':row['frame_index'],
                'joint_deg':np.rad2deg(np.abs(x[14:]-y[14:])).tolist(),
                'original_values17':x.tolist(),'tensorrt_values17':y.tolist(),
                'C_mm':[],'tip_mm':[],'hinge_mm':[],'rotation_deg':[],'point_projection_px':[]}
            for eye in ('left','right'):
                record['C_mm'].append(float(np.linalg.norm(xx[eye]['C']-yy[eye]['C'])))
                record['tip_mm'].extend(float(np.linalg.norm(xx[eye][k]-yy[eye][k])) for k in ('J1','J2'))
                record['hinge_mm'].extend(float(np.linalg.norm(xx[eye][k]-yy[eye][k])) for k in ('H1','H2'))
                record['rotation_deg'].append(float(np.rad2deg(Rotation.from_matrix(xx[eye]['R_shaft'].T@yy[eye]['R_shaft']).magnitude())))
                # Independent distorted projection of all five physical/reference points.
                for key in ('C','H1','H2','J1','J2'):
                    camera=row['calibration'][eye]
                    u0=project_points(xx[eye][key][None],camera['K'],camera['D'])[0]
                    u1=project_points(yy[eye][key][None],camera['K'],camera['D'])[0]
                    record['point_projection_px'].append(float(np.linalg.norm(u0-u1)))
            m0=d0['mask_logits']>0;m1=d1['mask_logits']>0
            record['mask_agreement']=float((m0==m1).float().mean())
            record['mask_iou']=float((m0&m1).sum()/((m0|m1).sum().clamp_min(1)))
            if truth:
                record['task_error']={}
                for name,value,decoded in (('original',x,xx),('tensorrt',y,yy)):
                    record['task_error'][name]={
                        'C_mm':[float(np.linalg.norm(decoded[e]['C']-truth['eyes'][e]['C'])) for e in ('left','right')],
                        'tip_mm':[float(np.linalg.norm(decoded[e][k]-truth['eyes'][e][k])) for e in ('left','right') for k in ('J1','J2')],
                        'joint_deg':np.rad2deg(np.abs(value[14:]-truth['joints'])).tolist()}
            rows.append(record)
            key=json.dumps(row['calibration'],sort_keys=True)
            if key not in overlays:overlays[key]=CppStereoOverlay(original.instrument_model,row['calibration'],display_mode='vectors')
            if row['category'] not in previews:
                im0,_=overlays[key].render(frame,PosePrediction(frame.frame_id,frame.timestamp_ns,x))
                im1,_=overlays[key].render(frame,PosePrediction(frame.frame_id,frame.timestamp_ns,y))
                comparison=np.concatenate((im0,im1),axis=0)
                filename=row['category']+'.jpg';Image.fromarray(comparison).save(a.output/filename,quality=93)
                previews[row['category']]=filename
            previous=row['sequence_id']
    for overlay in overlays.values():overlay.close()
    summary={name:stats([v for r in rows for v in r[name]]) for name in
             ('C_mm','tip_mm','hinge_mm','rotation_deg','point_projection_px')}
    joint=np.asarray([r['joint_deg'] for r in rows])
    summary['joint_deg']={n:stats(joint[:,i]) for i,n in enumerate(('pitch','yaw','jaw_half'))}
    summary['mask_iou']=stats([r['mask_iou'] for r in rows])
    # Engineering conversion tolerances, not scientific model accuracy targets.
    limits={'position_max_mm':.1,'rotation_max_deg':.1,'joint_max_deg':.1,'projection_max_px':.25}
    passed=(max(summary[k]['max'] for k in ('C_mm','tip_mm','hinge_mm'))<=limits['position_max_mm'] and
            summary['rotation_deg']['max']<=limits['rotation_max_deg'] and
            joint.max()<=limits['joint_max_deg'] and summary['point_projection_px']['max']<=limits['projection_max_px'])
    for r in layers.values():
        r['rms_mean']=float(np.mean(r.pop('rms')))
        r['reference_rms_mean']=float(np.mean(r.pop('reference_rms')))
    result={'completed':True,'passed_numerical_tolerances':bool(passed),'pairs':len(rows),
        'categories':len(previews),'engine':accelerated.visual_engine,
        'baseline_execution':original.execution_mode,'candidate_execution':accelerated.execution_mode,
        'baseline_engine':original.visual_engine,'summary':summary,'layers':layers,
        'conversion_limits':limits,'scope':'Difference from original eager C++ FP32 model, not error against ground truth. Independent recurrent state within each sequence. Shared unchanged GPU preprocessing.',
        'preview_layout':'top: original; bottom: TensorRT; each row left/right stereo',
        'previews':previews,'per_category':{c:{k:stats([v for r in rows if r['category']==c for v in r[k]]) for k in
           ('C_mm','tip_mm','joint_deg','point_projection_px')} for c in previews}}
    if truths:
        result['accuracy_vs_ground_truth']={}
        for name in ('original','tensorrt'):
            result['accuracy_vs_ground_truth'][name]={
                k:stats([v for r in rows for v in r['task_error'][name][k]]) for k in ('C_mm','tip_mm')}
            result['accuracy_vs_ground_truth'][name]['joint_deg']={
                n:stats([r['task_error'][name]['joint_deg'][i] for r in rows])
                for i,n in enumerate(('pitch','yaw','jaw_half'))}
        result['accuracy_scope']='Same 56 test pairs only, seven independent 8-frame clips; not full test evaluation. Image hashes match immutable label manifest.'
    if truths and a.baseline_config:
        comparisons={}
        for category in ['all']+sorted({r['category'] for r in rows}):
            selected=[r for r in rows if category=='all' or r['category']==category]
            comparison={}
            for metric in ('C_mm','tip_mm','pitch_deg','yaw_deg','jaw_half_deg'):
                def errors(name):
                    if metric.endswith('_mm'):return [v for r in selected for v in r['task_error'][name][metric]]
                    index=('pitch_deg','yaw_deg','jaw_half_deg').index(metric)
                    return [r['task_error'][name]['joint_deg'][index] for r in selected]
                base=stats(errors('original'));cand=stats(errors('tensorrt'))
                delta={k:cand[k]-base[k] for k in ('mean','p95')}
                comparison[metric]={'baseline':base,'candidate':cand,'increase':delta,'passed':max(delta.values())<=.005}
            comparisons[category]=comparison
        result['truth_accuracy_comparison']=comparisons
        result['passed_truth_accuracy']=all(m['passed'] for c in comparisons.values() for m in c.values())
        passed=passed and result['passed_truth_accuracy']
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps(result,indent=2),flush=True)
    if not passed:raise SystemExit('Numerical conversion tolerances failed; engine not accepted.')


if __name__=='__main__':main()
