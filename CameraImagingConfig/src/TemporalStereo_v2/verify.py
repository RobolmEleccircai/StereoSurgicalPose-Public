"""Independent projection arithmetic, temporal integrity, coverage and preview videos."""
import json
import numpy as np
import cv2
from PIL import Image
from runtime import arguments
from io_utils import read,write,sha,frame_valid,require_seal

def project_independent(points,K,D):
    p=np.asarray(points,float);x,y=(p[:,:2]/p[:,2:3]).T;k1,k2,p1,p2,k3=D;r=x*x+y*y
    f=1+k1*r+k2*r*r+k3*r*r*r
    return np.c_[K[0][0]*(x*f+2*p1*x*y+p2*(r+2*x*x))+K[0][2],
                 K[1][1]*(y*f+p1*(r+2*y*y)+2*p2*x*y)+K[1][2]]

def validate_record(label,fp,c):
    if not frame_valid(label,fp):raise ValueError('Incomplete/corrupt pair: '+str(label))
    r=read(label);truth=read(label.parent/r['camera_truth']);relative=np.array(truth['T_right_from_left_mm']);R=relative[:3,:3];t=relative[:3,3]
    if abs(r['time_s']-r['frame_index']/c['fps'])>1e-12 or r['timestamp_ns']!=round(r['frame_index']*1e9/c['fps']):raise ValueError('Timestamp mismatch')
    if r['fps']!=60 or r['position_unit']!='mm':raise ValueError('Unexpected time/length units')
    if not -1e-6<=r['jaw_half_angle_deg']<=30+1e-6:raise ValueError('Jaw angle out of range')
    output={'projection_error_px':0.,'stereo_position_error_mm':0.,'visible_tip_mask_distance_px':[]};images=[]
    for eye,e in r['eyes'].items():
        im=np.asarray(Image.open(label.parent/e['image']));mask=np.asarray(Image.open(label.parent/e['mask']))
        if im.shape!=(c['height'],c['width'],3) or mask.shape!=im.shape[:2] or not set(np.unique(mask)).issubset({0,255}):raise ValueError('Image/mask shape or encoding')
        if np.count_nonzero(mask)<c['minimum_visible_mask_pixels']:raise ValueError('Empty instrument')
        g=e['tool'];points=np.array([g[k] for k in ['shaft_reference_position_mm','tip_position_mm','jaw1_tip_position_mm','jaw2_tip_position_mm']])
        if not np.isfinite(points).all() or abs(np.linalg.norm(g['shaft_direction_unit'])-1)>1e-8:raise ValueError('Invalid pose target')
        if not np.allclose(points[1],points[2:].mean(0),atol=1e-8):raise ValueError('Virtual tip definition changed')
        pixel=np.array([g['shaft_reference_pixel'],g['tip_pixel'],*g['jaw_tip_pixels']])
        err=float(abs(project_independent(points,e['K'],e['D'])-pixel).max());output['projection_error_px']=max(output['projection_error_px'],err)
        if err>=1e-6:raise ValueError('Projection arithmetic disagrees')
        ys,xs=np.nonzero(mask)
        inside_distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        for tip,visibility in zip(g['jaw_tip_pixels'],e['jaw_visibility']):
            if visibility['visible']:
                d=float(np.sqrt(((xs-tip[0])**2+(ys-tip[1])**2).min()));output['visible_tip_mask_distance_px'].append(d)
                if d>2.:raise ValueError('Visible physical endpoint is misaligned with rendered mask')
            if visibility['in_frame'] and visibility['occluder_kind']=='tissue':
                x,y=np.rint(tip).astype(int);x=min(c['width']-1,max(0,x));y=min(c['height']-1,max(0,y))
                if inside_distance[y,x]>2.:raise ValueError('Tissue-hidden endpoint contradicts the visible mask')
        images.append(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
    for field in ['shaft_reference_position_mm','tip_position_mm','jaw1_tip_position_mm','jaw2_tip_position_mm']:
        error=float(np.linalg.norm(R@r['eyes']['left']['tool'][field]+t-r['eyes']['right']['tool'][field]))
        output['stereo_position_error_mm']=max(output['stereo_position_error_mm'],error)
        if error>1e-6:raise ValueError('Stereo truth inconsistent')
    if np.linalg.norm(R@r['eyes']['left']['tool']['shaft_direction_unit']-r['eyes']['right']['tool']['shaft_direction_unit'])>1e-8:raise ValueError('Stereo direction inconsistent')
    return r,output,images

def main():
    a=arguments();c=a.settings_data;run=a.run;m=read(run/'config/manifest.json');inv=read(run/'validation/render_inventory.json')
    require_seal(run/'trajectories/index.json',m['fingerprint'])
    rows=[];clips={};max_projection=0.;max_stereo=0.;tip_distances=[];key_masks={};mask_change_counts={}
    for job in inv['jobs']:
        label=run/'sequences'/job['sequence_id']/'labels'/f"{job['job_id']}.json"
        r,result,_=validate_record(label,m['fingerprint'],c)
        max_projection=max(max_projection,result['projection_error_px']);max_stereo=max(max_stereo,result['stereo_position_error_mm'])
        tip_distances.extend(result['visible_tip_mask_distance_px'])
        if r['reason'] in ('closed_jaw','maximum_opening'):
            key_masks[r['reason']]={eye:np.asarray(Image.open(label.parent/e['mask'])) for eye,e in r['eyes'].items()}
        rows.append({'sequence_id':r['sequence_id'],'frame_index':r['frame_index'],'time_s':r['time_s'],'label':str(label.relative_to(run)),'split':r['split'],'reason':r['reason']})
        if r['reason'] in ('continuous_category_clip','dataset_frame'):clips.setdefault(r['sequence_id'],[]).append(label)
    for sid,labels in clips.items():
        labels.sort(key=lambda p:read(p)['frame_index']);video_path=run/'previews'/f'{sid}_stereo_60fps.mp4'
        writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),c['fps'],(c['width']*2,c['height']))
        if not writer.isOpened():raise RuntimeError('Video codec unavailable')
        previous=None;previous_masks=None;changed=0
        for label in labels:
            r,_,images=validate_record(label,m['fingerprint'],c)
            masks={eye:np.asarray(Image.open(label.parent/e['mask'])) for eye,e in r['eyes'].items()}
            if previous:
                if r['frame_index']!=previous['frame_index']+1 or abs(r['time_s']-previous['time_s']-1/c['fps'])>1e-12:raise ValueError('Temporal discontinuity')
                for group,positions in r['joint_positions'].items():
                    for name,value in positions.items():
                        speed=abs(value-previous['joint_positions'][group][name])*c['fps']
                        limit=c['jaw_speed_rad_s'] if 'gripper' in name else c['insertion_speed_m_s'] if 'insertion_joint' in name else c['joint_speed_rad_s']
                        if speed>limit+1e-7:raise ValueError('Frame-to-frame joint speed exceeded')
                changed+=sum(int(np.count_nonzero(masks[eye]!=previous_masks[eye])) for eye in masks)
            previous=r;previous_masks=masks;writer.write(np.hstack(images))
        writer.release();reader=cv2.VideoCapture(str(video_path))
        if round(reader.get(cv2.CAP_PROP_FRAME_COUNT))!=len(labels) or abs(reader.get(cv2.CAP_PROP_FPS)-60)>1e-6:raise ValueError('Video frame count/rate mismatch')
        reader.release()
        mask_change_counts[sid]=changed
        if len(labels)>1 and changed==0:raise ValueError('Rendered tool masks did not update across the moving sequence: '+sid)
    jaw_change=None
    if a.mode=='validate':
        if not tip_distances:raise ValueError('No visible physical endpoint was available for the image-alignment check')
        if set(key_masks)!={'closed_jaw','maximum_opening'}:raise ValueError('Missing jaw limit keyframes')
        jaw_change={eye:int(np.count_nonzero(key_masks['closed_jaw'][eye]!=key_masks['maximum_opening'][eye])) for eye in ('left','right')}
        if min(jaw_change.values())<100:raise ValueError('Jaw pose change did not update both rendered masks')
    (run/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    # Verify source assets remain byte-identical after the full pipeline.
    for entry in m['dependencies']['files']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('A source asset changed during capture')
    index=read(run/'trajectories/index.json');sequences=[read(run/'trajectories'/f'{sid}.json') for sid in index['sequences']]
    insertion=[]
    joint_info=read(run/'config/joints.json');joints=joint_info['psm']
    insertion_name=next(n for n,j in joints.items() if j['linear'])
    insertion_index=joint_info['order']['psm'].index(insertion_name)
    for seq in sequences:
        for frame in seq['frames']:insertion.append(frame['q_psm'][insertion_index])
    lower=joints[insertion_name]['lower'];near_lower=sum(abs(x-lower)<1e-6 for x in insertion)
    if near_lower>max(1,len(insertion)//100):raise ValueError('Insertion samples accumulate at their lower limit')
    report={'mode':a.mode,'passed':True,'rendered_pairs':len(rows),'planned_sequences':len(sequences),'planned_frames':index['total_pairs'],
        'fps':60,'per_eye_resolution':[c['width'],c['height']],'main_target_scalars':18,
        'maximum_projection_error_px':max_projection,'maximum_stereo_consistency_error_mm':max_stereo,
        'visible_tip_checks':len(tip_distances),'maximum_visible_tip_mask_distance_px':max(tip_distances) if tip_distances else None,
        'mask_changed_pixels_by_sequence':mask_change_counts,'jaw_limit_changed_mask_pixels':jaw_change,
        'insertion_sample_range_m':[min(insertion),max(insertion)],'insertion_near_lower_limit_count':near_lower,'split_by_sequence':True,
        'limitations':['Geometric contact approximation; no tissue mechanics or contact forces.','Existing ECM optical mount is a preview convention, not a measured official calibration.','No shutter integration or motion blur.','Numerical projection checks do not establish real-world model accuracy.']}
    expected=len(index['validation_jobs']) if a.mode=='validate' else index['total_pairs']
    if len(rows)!=expected:raise ValueError('Incomplete output count')
    write(run/'validation/report.json',report)
    write(run/'status.json',{'state':'validated' if a.mode=='validate' else 'complete','validation_complete':True,
        'dataset_complete':a.mode=='generate','rendered_pairs':len(rows),'planned_pairs':index['total_pairs']})
    print('VALIDATION_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
