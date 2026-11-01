"""Render the complete 60 Hz trajectory schedule, resuming only incomplete frames."""
from pathlib import Path
import argparse,json,time,os,hashlib
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));a=p.parse_args()
c=json.loads(a.config.read_text());out=Path(c['output']);schedule=json.loads((out/'schedule.json').read_text())
assert schedule['config_sha256']==hashlib.sha256((out/'config.json').read_bytes()).hexdigest()
assert c==json.loads((out/'config.json').read_text())
meta=json.loads((out/'map_config.json').read_text())
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':c['renderer'],'width':c['width'],'height':c['height'],
    'multi_gpu':False,'anti_aliasing':0,'samples_per_pixel_per_frame':c['samples_per_pixel_per_frame'],
    'disable_viewport_updates':True})
import numpy as np
import cv2
from PIL import Image,ImageDraw
import carb,omni.usd,omni.timeline,omni.replicator.core as rep
from pxr import Usd,UsdGeom,UsdLux
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.semantics import add_update_semantics
from common import *

cv2.setNumThreads(3)
assert omni.usd.get_context().open_stage(str(out/'capture_scene.usda'))
stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetRootLayer())
stage.SetTimeCodesPerSecond(c['nominal_fps'])
timeline=omni.timeline.get_timeline_interface();timeline.stop()
settings=carb.settings.get_settings()
for name in ['chromaticAberration','lensFlares','tvNoise','motionblur','colorcorr','colorgrad','histogram']:
    settings.set_bool('/rtx/post/'+name+'/enabled',False)
settings.set_int('/rtx/post/aa/op',0); settings.set_bool('/rtx/post/dof/enabled',True)
settings.set_bool('/rtx/post/dof/override',False)
settings.set_int('/rtx/pathtracing/spp',c['samples_per_pixel_per_frame'])
settings.set_int('/rtx/pathtracing/totalSpp',c['samples_per_pixel_per_frame']*c['rt_subframes'])
settings.set_bool('/rtx/pathtracing/optixDenoiser/enabled',True)
settings.set_bool('/rtx-transient/resourcemanager/enableTextureStreaming',True)
rep.orchestrator.set_capture_on_play(False)
assert not stage.GetPrimAtPath('/SurgicalTools/PSM_Left').IsActive()
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith(ROBOT+'/') and prim.IsA(UsdGeom.Mesh):
        add_update_semantics(prim,'surgical_instrument')
profile=json.loads((out/'camera_profile.json').read_text());rays=np.load(out/'ray_maps.npz')
initial=json.loads((Path(c['runtime_root'])/c['appearance']).read_text())
dump(out/'appearance_source.json',initial)
kin=Kinematics(stage);refs=tip_references(stage)
reference=json.loads((out/'reference_transforms.json').read_text())
base_ecm=np.array(reference['ecm']);base_cam=np.array(reference['left_camera']);base_rig=np.array(reference['right_in_left'])
cams={};annotators={}
for eye in EYES:
    cam=Camera(prim_path=RIG+'/'+eye.capitalize(),resolution=(meta['width'],meta['height']),name='temporal_'+eye)
    cam.initialize();cam.set_clipping_range(.05,1000.)
    cam.set_opencv_pinhole_properties(fx=meta['f'],fy=meta['f'],cx=meta['cx']+.5,cy=meta['cy']+.5,pinhole=[0.]*12)
    for attr,value in {'focalLength':4.,'horizontalAperture':meta['horizontal_aperture_mm'],
                       'verticalAperture':meta['vertical_aperture_mm'],'focusDistance':10.,'fStop':5.6}.items():
        cam.prim.GetAttribute(attr).Set(value)
    cams[eye]=cam;annotators[eye]={}
    for name in ('HdrColor','semantic_segmentation','distance_to_image_plane'):
        ann=rep.AnnotatorRegistry.get_annotator(name,init_params={'colorize':False}) if name=='semantic_segmentation' else rep.AnnotatorRegistry.get_annotator(name)
        ann.attach([cam.get_render_product_path()]);annotators[eye][name]=ann

def png(path,image):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.png')
    Image.fromarray(image).save(tmp);os.replace(tmp,path)

def data(ann):
    d=ann.get_data();return np.array(d['data'] if isinstance(d,dict) else d)

def valid_existing(path):
    if not path.exists():return False
    d=json.loads(path.read_text())
    return all((path.parent/r['image']).is_file() and sha(path.parent/r['image'])==r['image_sha256'] and
               (path.parent/r['mask']).is_file() and sha(path.parent/r['mask'])==r['mask_sha256'] for r in d['eyes'].values())

completed=0;elapsed=[];started=time.monotonic();total=schedule['total_pairs']
dump(out/'status.json',{'state':'rendering','completed_pairs':0,'total_pairs':total,'fps':c['nominal_fps'],
                      'started_utc':datetime.now(timezone.utc).isoformat()})
print('RENDER_READY',meta,'total_pairs',total,flush=True)
for seq in schedule['sequences']:
    sequence=out/'sequences'/seq['sequence_id'];sequence.mkdir(parents=True,exist_ok=True)
    dump(sequence/'sequence.json',seq)
    for index in range(c['frames_per_sequence']):
        sid=f'{index:06d}';label_path=sequence/'labels'/f'{sid}.json'
        if valid_existing(label_path):completed+=1;continue
        frame_started=time.monotonic();state=frame_state(seq,index,c)
        apply_frame(stage,kin,seq,state,base_ecm,base_cam,base_rig)
        UsdLux.LightAPI(stage.GetPrimAtPath(RIG+'/EndoscopePointLight')).GetIntensityAttr().Set(1e6*state['light_multiplier'])
        ground=labels(stage,profile,refs)
        # Geometry is frozen for the stereo exposure; nominal sequence time is explicit metadata.
        rep.orchestrator.step(rt_subframes=c['rt_subframes'],pause_timeline=True,delta_time=0.,wait_for_render=True)
        record={'sequence_id':seq['sequence_id'],'frame_index':index,'time_s':index/c['nominal_fps'],
            'timestamp_ns':round(index*1e9/c['nominal_fps']),'fps':c['nominal_fps'],
            'previous_frame':index-1 if index else None,'next_frame':index+1 if index+1<c['frames_per_sequence'] else None,
            'position_unit':'mm','direction_unit':'unit vector','optical_axes':'X right, Y down, Z forward',
            'tip_definition':'T is the midpoint of the two actual articulated jaw endpoints; both endpoints are also provided. It is not a physical surface point when open.',
            'jaw_half_angle_deg':float(np.rad2deg(state['joints']['psm_tool_gripper2_joint'])),
            'jaw_total_opening_deg':float(np.rad2deg(state['joints']['psm_tool_gripper2_joint']-state['joints']['psm_tool_gripper1_joint'])),
            'joint_positions_rad_or_m':state['joints'],'appearance':{k:state[k] for k in ('light_multiplier','noise_multiplier','blur_multiplier')},
            'eyes':{},'split':seq['split'],'category':seq['category']}
        rgb_images={}
        for eye in EYES:
            uv=rays[eye];hdr=data(annotators[eye]['HdrColor'])[...,:3].astype(np.float32)
            assert hdr.shape==(meta['height'],meta['width'],3) and np.isfinite(hdr).all()
            linear=cv2.remap(hdr,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR)
            rgb=image_effects(linear,eye,profile,initial,state,seq['seed']*1000+index*2+EYES.index(eye),c)
            rawseg=annotators[eye]['semantic_segmentation'].get_data();seg=np.asarray(rawseg['data'])
            if seg.ndim==3:seg=seg.view(np.uint32).reshape(seg.shape[:2])
            ids=[int(k) for k,v in rawseg['info']['idToLabels'].items() if 'surgical_instrument' in v.get('class','')]
            assert ids,'Missing instrument semantic class'
            visible=np.isin(seg,ids).astype(np.uint8)*255
            mask=cv2.remap(visible,uv[:,:,0],uv[:,:,1],cv2.INTER_NEAREST)
            ys,xs=np.nonzero(mask);assert len(xs)>100,'Instrument not visible in rendered image'
            image_path=sequence/'images'/eye/f'{sid}.png';mask_path=sequence/'masks'/eye/f'{sid}.png'
            png(image_path,rgb);png(mask_path,mask);rgb_images[eye]=rgb
            dep=data(annotators[eye]['distance_to_image_plane']).squeeze()
            rawdep=cv2.remap(dep,uv[:,:,0],uv[:,:,1],cv2.INTER_NEAREST)
            tip_checks=[]
            for point,key in zip(ground[eye]['jaw_tip_pixels'],('jaw1_tip_position_mm','jaw2_tip_position_mm')):
                x,y=np.rint(point).astype(int);patch=rawdep[max(0,y-1):y+2,max(0,x-1):x+2]*1000
                finite=patch[np.isfinite(patch)&(patch>0)];z=ground[eye][key][2]
                tip_checks.append({'depth_agreement_within_1mm':bool(len(finite) and np.min(abs(finite-z))<=1.),
                                   'method':'nearest 3x3 rendered-depth agreement; not a visibility label for midpoint or internal C'})
            optical=world(stage,RIG+'/'+eye.capitalize())@AXES
            record['eyes'][eye]={'image':f'../images/{eye}/{sid}.png',
                'image_sha256':sha(image_path),'mask':f'../masks/{eye}/{sid}.png','mask_sha256':sha(mask_path),
                'K':profile[eye]['K'],'D':profile[eye]['D'],'tool':ground[eye],
                'visible_bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                'visible_mask_area_px':int(len(xs)),'jaw_tip_depth_checks':tip_checks}
            if c['save_depth']:
                (sequence/'depth'/eye).mkdir(parents=True,exist_ok=True);np.save(sequence/'depth'/eye/f'{sid}.npy',rawdep)
        # Ground-truth camera relationship is kept in a separate evaluation record.
        Tw={e:world(stage,RIG+'/'+e.capitalize())@AXES for e in EYES}
        rel=np.linalg.inv(Tw['right'])@Tw['left'];rel[:3,3]*=.01
        camera_truth={e:{'T_world_from_optical':Tw[e].tolist(),'world_translation_unit':'cm'} for e in EYES}
        camera_truth.update(T_right_from_left_m=rel.tolist(),baseline_mm=float(np.linalg.norm(rel[:3,3])*1000))
        dump(sequence/'camera_truth'/f'{sid}.json',camera_truth)
        dump(label_path,record)
        if index in (0,c['frames_per_sequence']//2,c['frames_per_sequence']-1):
            panels=[]
            for eye in EYES:
                im=Image.fromarray(rgb_images[eye]);draw=ImageDraw.Draw(im);g=ground[eye]
                draw.line([tuple(p) for p in g['shaft_polyline_pixels']],fill='yellow',width=3)
                for point,color in [(g['shaft_reference_pixel'],'cyan'),(g['tip_pixel'],'magenta'),
                                    (g['jaw_tip_pixels'][0],'lime'),(g['jaw_tip_pixels'][1],'orange')]:
                    x,y=point;draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
                draw.text((12,12),f"{seq['sequence_id']} {index} {eye} t={index/c['nominal_fps']:.3f}s",fill='white')
                panels.append(im)
            canvas=Image.new('RGB',(c['width']*2,c['height']));canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(c['width'],0))
            (out/'previews').mkdir(exist_ok=True);canvas.save(out/'previews'/f"{seq['sequence_id']}_{sid}.jpg",quality=90)
        completed+=1;dt=time.monotonic()-frame_started;elapsed.append(dt)
        avg=float(np.mean(elapsed[-20:]))
        dump(out/'status.json',{'state':'rendering','completed_pairs':completed,'total_pairs':total,
            'sequence_id':seq['sequence_id'],'frame_index':index,'nominal_fps':c['nominal_fps'],
            'seconds_last_pair':dt,'seconds_per_pair_recent':avg,'estimated_remaining_seconds':avg*(total-completed),
            'elapsed_this_process_seconds':time.monotonic()-started,'updated_utc':datetime.now(timezone.utc).isoformat()})
        print('FRAME_COMPLETE',completed,'/',total,seq['sequence_id'],index,'seconds',round(dt,3),flush=True)
    stage.GetRootLayer().Export(str(sequence/'last_frame.usda'))
dump(out/'render_complete.json',{'complete':True,'pairs':completed,'expected':total})
print('RENDER_COMPLETE',completed,flush=True)
app.close()
