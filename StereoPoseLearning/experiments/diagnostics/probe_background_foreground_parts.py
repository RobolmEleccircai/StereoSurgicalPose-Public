"""Frozen models on paired pixel-preserving background interventions (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def native_visible_masks(record, raw_size):
    """Load native visible masks solely to construct the diagnostic intervention."""
    height,width=raw_size
    masks=[]
    for eye in ('left','right'):
        with Image.open(record['mask_paths'][eye]) as source:
            mask=np.asarray(source.convert('L').resize((width,height),Image.Resampling.NEAREST))
        masks.append(torch.from_numpy((mask>0).copy()))
    return torch.stack(masks)[None,None,:,None]


def lowres_from_native(raw, image_size):
    """Use the dataset's PIL RGB resize, not a second inconsistent tensor path."""
    height,width=image_size
    if raw.dtype!=torch.uint8 or raw.shape[:3]!=(1,1,2):
        raise ValueError('Background probe expects one native uint8 stereo pair')
    mean=np.asarray([.485,.456,.406],np.float32)[:,None,None]
    std=np.asarray([.229,.224,.225],np.float32)[:,None,None]
    values=[]
    for eye in range(2):
        array=raw[0,0,eye].permute(1,2,0).cpu().numpy()
        rgb=np.asarray(Image.fromarray(array).resize((width,height),Image.Resampling.BILINEAR),np.float32)/255
        values.append(torch.from_numpy(((rgb.transpose(2,0,1)-mean)/std).copy()))
    return torch.stack(values)[None,None]


def paired_native_background_swap(first, second, first_mask, second_mask, image_size):
    """Swap native backgrounds, then derive BOTH models' localization inputs.

    The native union of both visible instruments is protected with the native
    equivalent of ten localization pixels. Mask labels are used for this
    controlled perturbation only; model.forward receives RGB and K/D alone.
    """
    raw,donor=first['hires_images'],second['hires_images']
    if raw.shape!=donor.shape or raw.dtype!=torch.uint8 or donor.dtype!=torch.uint8:
        raise ValueError('Source and donor must have matching native RGB dimensions')
    height,width=raw.shape[-2:]
    rh=int(np.ceil(10*height/image_size[0]));rw=int(np.ceil(10*width/image_size[1]))
    mask=(first_mask|second_mask).float().reshape(2,1,height,width)
    keep=F.max_pool2d(mask,(2*rh+1,2*rw+1),stride=1,padding=(rh,rw)).bool().reshape(1,1,2,1,height,width)
    swapped_raw=torch.where(keep,raw,donor)
    reverse_raw=torch.where(keep,donor,raw)
    # Regenerate even the unmodified localization inputs through the same path.
    first=dict(first,images=lowres_from_native(raw,image_size))
    second=dict(second,images=lowres_from_native(donor,image_size))
    swapped=dict(first,images=lowres_from_native(swapped_raw,image_size),hires_images=swapped_raw)
    reverse=dict(second,images=lowres_from_native(reverse_raw,image_size),hires_images=reverse_raw)
    delta=(swapped_raw.float()-raw.float()).abs()
    reverse_delta=(reverse_raw.float()-donor.float()).abs()
    protected_change=float((delta*keep).max())
    reverse_change=float((reverse_delta*keep).max())
    if protected_change!=0 or reverse_change!=0:
        raise AssertionError('Native protected instrument pixels changed')
    # Compare actual tool pixels in the downsampled image separately. The outer
    # margin boundary can mix during antialiased resize, so do not mislabel it
    # as bit-identical at the low-resolution boundary.
    low_tool=F.interpolate(mask,size=image_size,mode='nearest').reshape(1,1,2,1,*image_size)
    low_delta=(swapped['images']-first['images']).abs()
    audit={
        'protected_pixel_max_change':protected_change,
        'reverse_protected_pixel_max_change':reverse_change,
        'protected_pixel_units':'native uint8 RGB levels (0..255)',
        'protected_lowres_tool_pixel_max_change':float((low_delta*low_tool).max()),
        'self_swap_max_change':float((torch.where(keep,raw,raw).float()-raw.float()).abs().max()),
        'background_mean_change_normalized':float((low_delta*(1-low_tool)).sum()/((1-low_tool).sum()*3).clamp_min(1)),
        'native_background_mean_change_rgb':float((delta*(~keep)).sum()/((~keep).sum()*3).clamp_min(1)),
        'native_protection_margin_px_yx':[rh,rw],
        'native_image_size_hw':[height,width],
    }
    return first,second,swapped,reverse,audit


def main():
    p=argparse.ArgumentParser()
    for name in ('root','manifest','baseline','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--donor-category',default='lighting')
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root))
    from stereo_pose.runtime import load_model, forward_batch
    from stereo_pose.data import StereoDataset, project_points
    from stereo_pose.geometry import FixedKinematics
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    started=time.monotonic()
    # Snapshot atomically replaced best.pt so continued training cannot change
    # the evaluated weights halfway through the probe.
    frozen=out/'guided_checkpoint.pt';shutil.copy2(a.checkpoint,frozen)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    models={};identities={};sizes=[]
    for name,path in [('original',Path(a.baseline)),('guided',frozen)]:
        model,saved=load_model(path,torch.device('cpu'));model.requires_grad_(False)
        models[name]=(model,FixedKinematics(saved['instrument_model']).eval())
        identities[name]={'epoch':saved['epoch']+1,'sha256':digest(path),'path':str(path),
                          'foreground_parts':bool(saved['config']['model'].get('foreground_parts',False))}
        sizes.append(tuple(saved['config']['image_size']))
    if sizes[0]!=sizes[1]:raise ValueError('Paired background comparison requires the same localization resolution')
    image_size=sizes[0];height,width=image_size
    manifest=json.loads(Path(a.manifest).read_text())
    # Always retain native RGB so even legacy controls see a downsample of the
    # same intervention used by the new raw-image local branches.
    data={s:StereoDataset(manifest,s,window=1,image_size=image_size,include_hires=True) for s in ['train','validation']}
    def batch(split,index):
        return {k:v.unsqueeze(0) if torch.is_tensor(v) else v for k,v in data[split][index].items()}
    # Exact preselected pairs from the earlier diagnosis; no selection on new outputs.
    pairs=[(0,71),(4,76),(9,82),(13,87),(18,93),(22,97),(27,103),(31,108)]
    lookup={s:{r['frame_index']:i for i,r in enumerate(ds.records) if r['category']==('lighting' if s=='validation' else a.donor_category)} for s,ds in data.items()}
    if a.donor_category!='lighting':
        ids=sorted(lookup['train'])
        if not ids:raise ValueError('No training donor category '+a.donor_category)
        pairs=[(vi,ids[len(ids)//2]) for vi,_ in pairs]
    def infer(name,b):
        model,kin=models[name]
        with torch.inference_mode():
            pred,_=forward_batch(model,b,None)
            g=kin(pred['C'],pred['R'],pred['joints'])
            mask=pred['mask_logits'].sigmoid()>.5;truth=b['masks']>.5
            iou=float((mask&truth).sum()/(mask|truth).sum().clamp_min(1))
            logits=pred['heatmap_logits'];h,w=logits.shape[-2:]
            index=logits.flatten(-2).argmax(-1)
            pixels=torch.stack(((index%w+.5)*width/w-.5,(index//w+.5)*height/h-.5),-1)[0,0]
            uv=np.stack([project_points(b['points'][0,0,e,:5].numpy(),b['K'][0,0,e].numpy(),b['D'][0,0,e].numpy())[0] for e in range(2)])
            return {'joints_deg':torch.rad2deg(pred['joints'][0,0]).tolist(),
                    'tip_error_mm':float((g['points'][...,3:5,:]-b['points'][...,3:5,:]).norm(dim=-1).mean()),
                    'tip_points_mm':g['points'][0,0,:,3:5,:].tolist(),
                    'C_mm':pred['C'][0,0].tolist(),'mask_iou':iou,
                    'tip_heatmap_error_px':float((pixels[:,3:5]-torch.as_tensor(uv[:,3:5])).norm(dim=-1).mean())}
    rows=[];example=None
    for vi,ti in pairs:
        b=batch('validation',lookup['validation'][vi]);tb=batch('train',lookup['train'][ti])
        validation_record=data['validation'].records[lookup['validation'][vi]]
        training_record=data['train'].records[lookup['train'][ti]]
        native_size=b['hires_images'].shape[-2:]
        vm=native_visible_masks(validation_record,native_size)
        tm=native_visible_masks(training_record,native_size)
        b,tb,swapped,reverse,intervention=paired_native_background_swap(b,tb,vm,tm,image_size)
        row={'validation_frame':vi,'training_frame':ti,
             'validation_sequence':data['validation'].records[lookup['validation'][vi]]['sequence_id'],
             'training_sequence':data['train'].records[lookup['train'][ti]]['sequence_id'],
             'validation_truth_deg':torch.rad2deg(b['joints'][0,0]).tolist(),
             'training_truth_deg':torch.rad2deg(tb['joints'][0,0]).tolist(),
             **intervention,
             'results':{}}
        for name in models:
            row['results'][name]={key:infer(name,value) for key,value in [
                ('validation',b),('validation_swapped',swapped),('training',tb),('training_swapped',reverse)]}
        rows.append(row)
        if vi==18:example=(b['images'],swapped['images'],row)
        print(json.dumps({'completed_pairs':len(rows),'validation_frame':vi,
              'original_jaw':[row['results']['original'][key]['joints_deg'][2] for key in ['validation','validation_swapped']],
              'guided_jaw':[row['results']['guided'][key]['joints_deg'][2] for key in ['validation','validation_swapped']]}),flush=True)
    summary={}
    for name in models:
        summary[name]={}
        for direction in ['validation','training']:
            truth=np.array([r[direction+'_truth_deg'] for r in rows])
            old=np.array([r['results'][name][direction]['joints_deg'] for r in rows])
            new=np.array([r['results'][name][direction+'_swapped']['joints_deg'] for r in rows])
            summary[name][direction]={
                'joint_mae_original_deg':np.abs(old-truth).mean(0).tolist(),
                'joint_mae_swapped_deg':np.abs(new-truth).mean(0).tolist(),
                'prediction_shift_mean_abs_deg':np.abs(new-old).mean(0).tolist(),
                'prediction_shift_max_abs_deg':np.abs(new-old).max(0).tolist(),
                'tip_prediction_shift_mm':float(np.mean([np.linalg.norm(np.array(r['results'][name][direction]['tip_points_mm'])-np.array(r['results'][name][direction+'_swapped']['tip_points_mm']),axis=-1).mean() for r in rows])),
                **{key:{condition:float(np.mean([r['results'][name][direction+suffix][key] for r in rows])) for condition,suffix in [('original',''),('swapped','_swapped')]} for key in ['tip_error_mm','mask_iou','tip_heatmap_error_px']}}
    report={'checkpoints':identities,'joint_order':['wrist_pitch','wrist_yaw','jaw_half'],
            'donor_category':a.donor_category,
            'protocol':('Same 8 lighting validation/training pairs as prior diagnosis. ' if a.donor_category=='lighting' else 'Same 8 lighting validation frames; fixed midpoint training frame from '+a.donor_category+'. ')+
                'CPU FP32; independent single stereo pairs with hidden=None. Both-eye background exchange is applied at native RGB resolution outside the union of native true visible-tool masks plus the native equivalent of 10 localization pixels (20 raw pixels at 720p/360p). Localization RGB for BOTH models is regenerated by the dataset PIL resize from those native originals/composites. Foreground-parts models also receive that same native RGB. Ground-truth masks are used only to construct controlled interventions, never passed to model.forward().',
            'localization_image_size_hw':list(image_size),'native_rgb_intervention':True,
            'limitations':'Synthetic composites can introduce seams and inconsistent background stereo geometry. This tests sensitivity, not validated new scenes or real-world generalization. Eight lighting pairs do not establish all-category invariance.',
            'manifest_sha256':manifest['manifest_sha256'],'source_images_modified':False,'training_modified':False,'test_split_used':False,
            'elapsed_seconds':time.monotonic()-started,'summary':summary,'rows':rows}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    first,second,row=example
    mean=torch.tensor([.485,.456,.406])[None,None,None,:,None,None]
    std=torch.tensor([.229,.224,.225])[None,None,None,:,None,None]
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,18) if Path(font_path).exists() else ImageFont.load_default()
    rowheight=height+90
    canvas=Image.new('RGB',(2*width,2*rowheight),(20,26,30));draw=ImageDraw.Draw(canvas)
    for index,(title,image,key) in enumerate([('Original validation background',first,'validation'),(a.donor_category+' background; tool and 10px margin unchanged',second,'validation_swapped')]):
        y=index*rowheight
        draw.text((12,y+8),title,fill='white',font=font)
        draw.text((12,y+32),'Truth pitch / yaw / jaw-half: '+ ' / '.join(f'{x:.2f} deg' for x in row['validation_truth_deg']),fill=(170,240,170),font=font)
        old=row['results']['original'][key]['joints_deg'];new=row['results']['guided'][key]['joints_deg']
        draw.text((12,y+56),'Old model: '+' / '.join(f'{x:.2f}' for x in old)+'     Guided model: '+' / '.join(f'{x:.2f}' for x in new),fill=(255,215,110),font=font)
        rgb=((image*std+mean).clamp(0,1)[0,0].permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
        for e in range(2):canvas.paste(Image.fromarray(rgb[e]),(e*width,y+90))
    canvas.save(out/'background_swap_comparison.jpg',quality=95)
    assert digest(frozen)==identities['guided']['sha256']
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
