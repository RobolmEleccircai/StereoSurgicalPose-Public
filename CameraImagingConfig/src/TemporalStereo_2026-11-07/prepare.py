"""Prepare every continuous trajectory before the single unattended rendering run."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'.deps'))
sys.path.insert(0,'/home/leorobot/isaacsim-5.1.0/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311')
import numpy as np
from pxr import Usd,UsdGeom
from common import *
from collision import CollisionCheck

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=HERE/'config.json');a=p.parse_args()
c=json.loads(a.config.read_text()); out=Path(c['output']); out.mkdir(parents=True,exist_ok=True)
if (out/'schedule.json').exists():
    prior=json.loads((out/'config.json').read_text()); assert c==prior,'Existing dataset has a different config'
    print('SCHEDULE_ALREADY_PREPARED',flush=True);sys.exit(0)
root=Path(c['runtime_root']); source=root/c['source_scene']
stage=Usd.Stage.CreateNew(str(out/'capture_scene.usda')) if not (out/'capture_scene.usda').exists() else Usd.Stage.Open(str(out/'capture_scene.usda'))
stage.GetRootLayer().subLayerPaths=[str(source)]; stage.GetRootLayer().Save()
kin=Kinematics(stage); refs=tip_references(stage); profile=camera_profile(c)
base_ecm=world(stage,ECM); base_cam=world(stage,RIG+'/Left')@AXES
base_rig=np.linalg.inv(base_cam)@(world(stage,RIG+'/Right')@AXES)
assert not stage.GetPrimAtPath('/SurgicalTools/PSM_Left').IsActive()
kin.solve(kin.base)
collision=CollisionCheck(stage)
print('COLLISION_MODEL',collision.audit,flush=True)
mapped,mapmeta=maps(profile);np.savez_compressed(out/'ray_maps.npz',**mapped)
dump(out/'config.json',c); dump(out/'camera_profile.json',profile);dump(out/'map_config.json',mapmeta)
dump(out/'joint_audit.json',{'source_sha256':sha(source),'joints':kin.audit(),'collision':collision.audit,
    'manufacturer_source':c['jaw_source'],'jaw_half_angle_range_deg':c['jaw_half_angle_deg'],
    'control_software_extension_not_used':'dVRK tool config permits larger commands; dataset uses manufacturer/model intersection',
    'kinematic_only':True,'motion_blur':False})
dump(out/'reference_transforms.json',{'ecm':base_ecm.tolist(),'left_camera':base_cam.tolist(),'right_in_left':base_rig.tolist()})
rng=np.random.default_rng(c['seed']); duration=(c['frames_per_sequence']-1)/c['nominal_fps'];schedule=[];rejections={}
cats=[cat for cat,n in c['categories'].items() for _ in range(n)]
assert len(cats)==c['sequence_count']
for sid,category in enumerate(cats):
    for attempt in range(300):
        q0=dict(kin.base);q1=dict(kin.base)
        # Angular offsets in radians and insertion in native joint meters.
        spread={'psm_yaw_joint':np.deg2rad(3.0),'psm_pitch_end_joint':np.deg2rad(3.0),
                'psm_main_insertion_joint':.010,'psm_tool_roll_joint':np.deg2rad(25),
                'psm_tool_pitch_joint':np.deg2rad(15),'psm_tool_yaw_joint':np.deg2rad(8)}
        for name,span in spread.items():
            mid=kin.base[name]+rng.uniform(-span,span)
            delta=rng.uniform(-span*.5,span*.5)
            j=kin.by_name[name]
            q0[name]=float(np.clip(mid-delta/2,j['lower']+1e-5,j['upper']-1e-5))
            q1[name]=float(np.clip(mid+delta/2,j['lower']+1e-5,j['upper']-1e-5))
        alpha0=rng.uniform(0,30)
        alpha1=float(np.clip(alpha0+rng.uniform(-15,15),0,30))
        if category=='combined_pose_jaw':
            alpha0,alpha1=((0.,15.) if sid%2==0 else (30.,15.))
        # Endpoints and smoothstep's maximal derivative enforce the published actuator speed.
        for q,alpha in ((q0,alpha0),(q1,alpha1)):
            q['psm_tool_gripper1_joint']=-float(np.deg2rad(alpha))
            q['psm_tool_gripper2_joint']=float(np.deg2rad(alpha))
        assert 1.5*abs(np.deg2rad(alpha1-alpha0))/duration<=c['jaw_max_velocity_rad_s']
        e0=np.zeros(6);e1=np.zeros(6)
        if category in ('ecm_motion','background_edge','combined_pose_jaw'):
            e0=np.r_[rng.uniform(-4,4,3),rng.uniform(-2,2,3)]
            e1=e0+np.r_[rng.uniform(-3,3,3),rng.uniform(-1,1,3)]
        seq=dict(sequence_id=f'seq_{sid:03d}',category=category,seed=int(c['seed']+sid),
                 split=('test' if sid in (11,13) else 'validation' if sid in (5,9) else 'train'),
                 q_start=q0,q_end=q1,ecm_start=e0.tolist(),ecm_end=e1.tolist(),
                 baseline_scale=float(rng.uniform(.95,1.05)) if category=='stereo_rig' else 1.,
                 relative_yaw_delta_deg=float(rng.uniform(-.5,.5)) if category=='stereo_rig' else 0.,
                 light_start=float(rng.uniform(.7,1.3)) if category in ('lighting','combined_pose_jaw') else 1.,
                 light_end=float(rng.uniform(.7,1.3)) if category in ('lighting','combined_pose_jaw') else 1.,
                 noise_multiplier=float(rng.uniform(.5,1.5)) if category=='sensor_quality' else 1.,
                 blur_multiplier=float(rng.uniform(.75,1.25)) if category=='sensor_quality' else 1.,
                 frames=c['frames_per_sequence'],fps=c['nominal_fps'])
        valid=True;reason=''
        # Inspect all states, not just the two endpoints, before committing a trajectory.
        for index in range(c['frames_per_sequence']):
            state=frame_state(seq,index,c);apply_frame(stage,kin,seq,state,base_ecm,base_cam,base_rig)
            record=labels(stage,profile,refs)
            for eye in EYES:
                r=record[eye]
                if not all(r['in_frame']): valid=False;reason='outside_frame';break
                span=np.linalg.norm(np.array(r['tip_pixel'])-r['shaft_reference_pixel'])
                if span<70:valid=False;reason='insufficient_tip_axis_span';break
            if not valid:break
            ok,path=collision.check()
            if not ok: valid=False;reason='tool_anatomy_intersection';break
        if valid:
            seq['preflight']={'all_frames_in_view':True,'all_frames_collision_free':True,'attempt':attempt+1,
                'max_jaw_speed_rad_s':1.5*abs(np.deg2rad(alpha1-alpha0))/duration}
            schedule.append(seq);print('PREPARED',seq['sequence_id'],category,'jaw_half_deg',alpha0,alpha1,flush=True);break
        rejections[reason]=rejections.get(reason,0)+1
        if attempt%20==0:print('RESAMPLING',sid,attempt,reason,flush=True)
    else:raise RuntimeError(f'No valid continuous trajectory for {category}: {rejections}')
dump(out/'schedule.json',{'config_sha256':sha(out/'config.json'),'sequences':schedule,'rejected_candidates':rejections,
    'total_pairs':len(schedule)*c['frames_per_sequence'],'all_frames_preflighted':True})
print('PREPARED_ALL',len(schedule)*c['frames_per_sequence'],flush=True)
