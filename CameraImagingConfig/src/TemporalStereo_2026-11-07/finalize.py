"""Validate all sequential labels, hashes and movies after the unattended render."""
from pathlib import Path
import argparse,json,hashlib,math,os
import numpy as np
import cv2
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));a=p.parse_args()
c=json.loads(a.config.read_text());root=Path(c['output']);schedule=json.loads((root/'schedule.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
rows=[];sequences=[];maximum_projection_error=0.;maximum_jaw_speed=0.
for seq in schedule['sequences']:
    base=root/'sequences'/seq['sequence_id'];last=None
    (root/'videos').mkdir(exist_ok=True)
    video=cv2.VideoWriter(str(root/'videos'/f"{seq['sequence_id']}_stereo.mp4"),cv2.VideoWriter_fourcc(*'mp4v'),c['nominal_fps'],(c['width']*2,c['height']))
    assert video.isOpened(),'Cannot create sequence preview video'
    for index in range(c['frames_per_sequence']):
        path=base/'labels'/f'{index:06d}.json';r=json.loads(path.read_text())
        assert r['frame_index']==index and r['sequence_id']==seq['sequence_id']
        assert abs(r['time_s']-index/c['nominal_fps'])<1e-12
        assert 0<=r['jaw_half_angle_deg']<=30+1e-6
        q=r['joint_positions_rad_or_m']; assert abs(q['psm_tool_gripper1_joint']+q['psm_tool_gripper2_joint'])<1e-10
        if last:
            dt=r['time_s']-last['time_s'];assert abs(dt-1/c['nominal_fps'])<1e-12
            speed=abs(np.deg2rad(r['jaw_half_angle_deg']-last['jaw_half_angle_deg']))/dt
            maximum_jaw_speed=max(maximum_jaw_speed,speed);assert speed<=c['jaw_max_velocity_rad_s']+1e-8
        last=r;images=[]
        for eye in ('left','right'):
            e=r['eyes'][eye];impath=path.parent/e['image'];maskpath=path.parent/e['mask']
            assert sha(impath)==e['image_sha256'] and sha(maskpath)==e['mask_sha256']
            im=np.array(Image.open(impath));assert im.shape==(c['height'],c['width'],3)
            mask=np.array(Image.open(maskpath));assert mask.shape==im.shape[:2]
            images.append(cv2.cvtColor(im,cv2.COLOR_RGB2BGR));g=e['tool']
            for name in ('shaft_reference_position_mm','shaft_direction_unit','tip_position_mm'):
                assert len(g[name])==3 and np.isfinite(g[name]).all()
            assert abs(np.linalg.norm(g['shaft_direction_unit'])-1)<1e-8
            assert np.allclose(g['tip_position_mm'],(np.array(g['jaw1_tip_position_mm'])+g['jaw2_tip_position_mm'])/2)
            xyz=np.array([g['shaft_reference_position_mm'],g['tip_position_mm']])
            uv=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),np.array(e['K']),np.array(e['D']))[0].reshape(-1,2)
            error=float(np.max(abs(uv-np.array([g['shaft_reference_pixel'],g['tip_pixel']]))))
            maximum_projection_error=max(maximum_projection_error,error);assert error<1e-8
        video.write(np.hstack(images));rows.append({'sequence_id':seq['sequence_id'],'frame_index':index,
            'label':str(path.relative_to(root)),'split':seq['split'],'time_s':r['time_s']})
    video.release();sequences.append(seq['sequence_id'])
assert len(rows)==schedule['total_pairs']
(root/'index.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
report={'complete':True,'verified_stereo_pairs':len(rows),'verified_rgb_images':len(rows)*2,
    'sequences':len(sequences),'frames_per_sequence':c['frames_per_sequence'],'nominal_fps':c['nominal_fps'],
    'max_projection_error_px':maximum_projection_error,'max_jaw_speed_rad_s':maximum_jaw_speed,
    'jaw_half_angle_limit_deg':[0,30],'total_opening_limit_deg':[0,60],
    'label_contract':'18 main values, C/d/T for both eyes; additional individual jaw endpoints and jaw angle',
    'validation_scope':'File integrity, geometry and temporal consistency; not learned or real-world accuracy.'}
(root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'status.json').write_text(json.dumps({'state':'complete','completed_pairs':len(rows),'total_pairs':len(rows),'validation':'validation.json'},indent=2)+'\n')
print('DATASET_COMPLETE',json.dumps(report),flush=True)
