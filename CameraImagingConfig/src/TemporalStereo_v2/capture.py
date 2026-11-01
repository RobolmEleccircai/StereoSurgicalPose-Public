"""Bounded laptop validation or explicit server generation; atomic stereo records."""
from pathlib import Path
import argparse,time,os,json
from io_utils import read,write,sha,source_hash,load_machine,assert_mode,reserve_render,frame_valid,commit_label,require_seal,validate_settings

def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',type=Path,required=True);p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--mode',required=True,choices=['validate','generate'])
    a=p.parse_args();m=load_machine(a.machine);c=read(a.settings);assert_mode(a.mode,m)
    validate_settings(c)
    run=a.run;manifest=read(run/'config/manifest.json');index=read(run/'trajectories/index.json');fp=manifest['fingerprint']
    require_seal(run/'trajectories/index.json',fp)
    if manifest['code_sha256']!=source_hash() or manifest['settings']!=c or index['fingerprint']!=fp:raise ValueError('Run source/configuration changed')
    sequences={sid:read(run/'trajectories'/f'{sid}.json') for sid in index['sequences']}
    for sid,h in index['trajectory_hashes'].items():
        if sha(run/'trajectories'/f'{sid}.json')!=h:raise ValueError('Trajectory was changed after inspection')
        require_seal(run/'trajectories'/f'{sid}.json',fp)
    jobs=index['validation_jobs'] if a.mode=='validate' else [{'sequence_id':sid,'frame_index':i,'job_id':f'{i:06d}','reason':'dataset_frame'} for sid in index['sequences'] for i in range(c['frames_per_sequence'])]
    if a.mode=='validate' and len(jobs)>min(32,c['validation_render_limit']):raise ValueError('Validation cap exceeded')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'renderer':c['renderer'],'width':c['width'],'height':c['height'],'multi_gpu':False,
        'anti_aliasing':0,'samples_per_pixel_per_frame':c['samples_per_pixel_per_frame'],'disable_viewport_updates':True})
    import numpy as np
    import cv2
    from PIL import Image,ImageDraw
    import carb,omni.usd,omni.timeline,omni.replicator.core as rep
    from pxr import UsdGeom,UsdLux,Sdf
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.semantics import add_update_semantics
    from runtime import Scene
    from safety import Safety
    from paths import Trajectory,jaw_positions,smoother
    from kinematics import rigid_mm,usd_world,AXES
    from imaging import maps,image_effects
    cv2.setNumThreads(3)
    scene_file=run/'config/capture_scene.usda'
    layer=Sdf.Layer.CreateNew(str(scene_file)) if not scene_file.exists() else Sdf.Layer.FindOrOpen(str(scene_file))
    layer.Clear();layer.subLayerPaths=[manifest['source_paths']['source_scene']];layer.Save()
    if not omni.usd.get_context().open_stage(str(scene_file)):raise RuntimeError('Stage open failed')
    scene=Scene(m,c,omni.usd.get_context().get_stage());stage=scene.stage
    for attr,path in manifest['dependencies']['asset_overrides'].items():stage.GetAttributeAtPath(attr).Set(Sdf.AssetPath(path))
    stage.SetTimeCodesPerSecond(c['fps']);stage.SetFramesPerSecond(c['fps']);omni.timeline.get_timeline_interface().stop()
    settings=carb.settings.get_settings()
    for name in ['chromaticAberration','lensFlares','tvNoise','motionblur','colorcorr','colorgrad','histogram']:settings.set_bool('/rtx/post/'+name+'/enabled',False)
    settings.set_int('/rtx/post/aa/op',0);settings.set_bool('/rtx/post/dof/enabled',True);settings.set_bool('/rtx/post/dof/override',False)
    settings.set_int('/rtx/pathtracing/spp',c['samples_per_pixel_per_frame']);settings.set_int('/rtx/pathtracing/totalSpp',c['samples_per_pixel_per_frame']*c['rt_subframes'])
    settings.set_bool('/rtx/pathtracing/optixDenoiser/enabled',True);settings.set_bool('/rtx-transient/resourcemanager/enableTextureStreaming',True)
    rep.orchestrator.set_capture_on_play(False)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(c['psm_root']+'/') and prim.IsA(UsdGeom.Mesh):add_update_semantics(prim,'surgical_instrument')
    safety=Safety(scene);rays,meta=maps(scene.profile,c['native_camera']['focalLength']);write(run/'config/render_map.json',meta)
    anns={};cameras={}
    for eye in ('left','right'):
        cam=Camera(prim_path=scene.rig+'/'+eye.capitalize(),resolution=(meta['width'],meta['height']),name='temporal_v2_'+eye)
        cam.initialize();cam.set_clipping_range(*c['native_camera']['clippingRange'])
        cam.set_opencv_pinhole_properties(fx=meta['f'],fy=meta['f'],cx=meta['cx']+.5,cy=meta['cy']+.5,pinhole=[0.]*12)
        for attr,value in {**{k:v for k,v in c['native_camera'].items() if k!='clippingRange'},'horizontalAperture':meta['horizontal_aperture_mm'],'verticalAperture':meta['vertical_aperture_mm']}.items():cam.prim.GetAttribute(attr).Set(value)
        cameras[eye]=cam;anns[eye]={}
        for name in ('HdrColor','semantic_segmentation','distance_to_image_plane'):
            ann=rep.AnnotatorRegistry.get_annotator(name,init_params={'colorize':False}) if name=='semantic_segmentation' else rep.AnnotatorRegistry.get_annotator(name)
            ann.attach([cam.get_render_product_path()]);anns[eye][name]=ann
    write(run/'config/native_camera_readback.json',{'record_type':'initialized_native_camera_attributes',
        'stage_meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),
        'eyes':{eye:{key:float(cam.prim.GetAttribute(key).Get()) for key in ('focalLength','horizontalAperture','verticalAperture','focusDistance','fStop')} for eye,cam in cameras.items()}})
    def png(path,array):
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'.partial.png');Image.fromarray(array).save(tmp);os.replace(tmp,path)
    completed=0;times=[]
    for job in jobs:
        seq=sequences[job['sequence_id']];frame=seq['frames'][job['frame_index']];folder=run/'sequences'/seq['sequence_id'];ident=job['job_id'];label=folder/'labels'/f'{ident}.json'
        if frame_valid(label,fp):completed+=1;continue
        qp=np.array(frame['q_psm']);qe=np.array(frame['q_ecm']);path=Trajectory(seq['path']);vp,ve=path.velocity(frame['time_s'])
        if 'q_psm_override' in job:qp=np.array(job['q_psm_override']);vp*=0;ve*=0
        scene.apply(qp,qe,seq,vp,ve);check=safety.check(qp,qe)
        if not check['safe']:raise ValueError('Unsafe render state: '+str(check))
        ground=scene.labels(qp,qe,seq);cams=scene.cameras(qe,seq);_,_,physical_tips=scene.points(qp)
        for eye in ('left','right'):
            observed=rigid_mm(usd_world(stage,scene.rig+'/'+eye.capitalize()))@AXES
            if not np.allclose(observed,cams[eye],atol=1e-6):raise ValueError('Camera pose and labels disagree')
        u=frame['frame_index']/(c['frames_per_sequence']-1)
        effects={'light_multiplier':float(seq['light_start']+(seq['light_end']-seq['light_start'])*smoother(u)),
                 'noise_multiplier':seq['noise_multiplier'],'blur_multiplier':seq['blur_multiplier']}
        UsdLux.LightAPI(stage.GetPrimAtPath(scene.rig+'/EndoscopePointLight')).GetIntensityAttr().Set(c['endoscope_light_intensity']*effects['light_multiplier'])
        reserve_render(m,c,run.name,seq['sequence_id']+'/'+ident);started=time.monotonic()
        rep.orchestrator.step(rt_subframes=c['rt_subframes'],pause_timeline=True,delta_time=0.,wait_for_render=True)
        record={'schema_version':2,'fingerprint':fp,'mode':a.mode,'sequence_id':seq['sequence_id'],'job_id':ident,
            'frame_index':frame['frame_index'],'time_s':frame['time_s'],'timestamp_ns':frame['timestamp_ns'],'fps':c['fps'],
            'reason':job['reason'],'category':seq['category'],'phase':frame['phase'],'split':seq['split'],
            'isolated_validation_pose_override':'q_psm_override' in job,
            'optical_axes':'X right, Y down, Z forward','position_unit':'mm','direction_unit':'unit vector',
            'C_definition':'roll-link origin, internal shaft-axis reference','T_definition':'virtual midpoint of both physical jaw endpoints',
            'jaw_half_angle_deg':float(np.rad2deg(qp[scene.psm.names.index('psm_tool_gripper2_joint')])),
            'joint_positions':{'psm':scene.psm.positions(qp),'ecm':scene.ecm.positions(qe)},
            'joint_velocities':{'psm':scene.psm.positions(vp),'ecm':scene.ecm.positions(ve)},
            'contact':{'geometric_contact':check['organ_distance_mm']<=c['contact_max_mm'],'distance_mm':check['organ_distance_mm'],
                'closest_pair':check['organ_pair'],'model':'near_zero_gap_not_force_simulation'},'appearance':effects,'eyes':{}}
        panels=[]
        for eye in ('left','right'):
            uv=rays[eye];raw=anns[eye]['HdrColor'].get_data();hdr=np.asarray(raw['data'] if isinstance(raw,dict) else raw)[...,:3].astype(np.float32)
            if hdr.shape!=(meta['height'],meta['width'],3) or not np.isfinite(hdr).all():raise ValueError('Invalid HDR render')
            linear=cv2.remap(hdr,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR)
            rgb=image_effects(linear,eye,scene.profile,scene.appearance,effects,seq['seed']*1000+frame['frame_index']*2+int(eye=='right'),c)
            seg=anns[eye]['semantic_segmentation'].get_data();data=np.asarray(seg['data'])
            if data.ndim==3:data=data.view(np.uint32).reshape(data.shape[:2])
            ids=[int(k) for k,v in seg['info']['idToLabels'].items() if 'surgical_instrument' in v.get('class','')]
            if not ids:raise ValueError('Instrument semantic class missing')
            mask=cv2.remap(np.isin(data,ids).astype(np.uint8)*255,uv[:,:,0],uv[:,:,1],cv2.INTER_NEAREST)
            ys,xs=np.nonzero(mask)
            if len(xs)<c['minimum_visible_mask_pixels']:raise ValueError('Visible instrument area too small')
            ip=folder/'images'/eye/f'{ident}.png';mp=folder/'masks'/eye/f'{ident}.png';png(ip,rgb);png(mp,mask)
            visibility=[safety.visible(cams[eye][:3,3],point) for point in physical_tips]
            for v,inside in zip(visibility,ground[eye]['in_frame'][2:]):v['in_frame']=inside;v['visible']=bool(v['visible'] and inside)
            inside_distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
            for pixel,v in zip(ground[eye]['jaw_tip_pixels'],visibility):
                if v['visible'] and float(np.sqrt(((xs-pixel[0])**2+(ys-pixel[1])**2).min()))>2.:
                    raise ValueError('Visible endpoint does not align with the rendered mask')
                if v['in_frame'] and v['occluder_kind']=='tissue':
                    x,y=np.rint(pixel).astype(int);x=min(c['width']-1,max(0,x));y=min(c['height']-1,max(0,y))
                    if inside_distance[y,x]>2.:raise ValueError('Tissue-hidden endpoint contradicts the rendered mask')
            record['eyes'][eye]={'image':f'../images/{eye}/{ident}.png','image_sha256':sha(ip),'mask':f'../masks/{eye}/{ident}.png','mask_sha256':sha(mp),
                'K':scene.profile[eye]['K'],'D':scene.profile[eye]['D'],'tool':ground[eye],'jaw_visibility':visibility,
                'visible_bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],'visible_mask_area_px':int(len(xs))}
            im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);g=ground[eye]
            draw.line([tuple(x) for x in g['shaft_polyline_pixels']],fill='yellow',width=3)
            for point,color in [(g['shaft_reference_pixel'],'cyan'),(g['tip_pixel'],'magenta'),(g['jaw_tip_pixels'][0],'lime'),(g['jaw_tip_pixels'][1],'orange')]:
                x,y=point;draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
            draw.text((12,12),f"{seq['sequence_id']} {ident} {eye} t={frame['time_s']:.6f}s",fill='white');panels.append(im)
        relative=np.linalg.inv(cams['right'])@cams['left'];truth={'T_world_from_optical_mm':{k:v.tolist() for k,v in cams.items()},
             'T_right_from_left_mm':relative.tolist(),'baseline_mm':float(np.linalg.norm(relative[:3,3]))}
        tp=folder/'camera_truth'/f'{ident}.json';write(tp,truth);record['camera_truth']=f'../camera_truth/{ident}.json';record['camera_truth_sha256']=sha(tp)
        commit_label(label,record) # Receipt is the final commit marker for a complete pair.
        canvas=Image.new('RGB',(2*c['width'],c['height']));canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(c['width'],0))
        preview=run/'previews'/f"{seq['sequence_id']}_{ident}.jpg";preview.parent.mkdir(exist_ok=True);canvas.save(preview,quality=90)
        completed+=1;times.append(time.monotonic()-started)
        write(run/'status.json',{'state':'rendering','mode':a.mode,'completed_pairs':completed,'target_pairs':len(jobs),'nominal_fps':c['fps'],
            'seconds_last_pair':times[-1],'complete':False})
        print('PAIR_COMPLETE',completed,'/',len(jobs),seq['sequence_id'],ident,'seconds',round(times[-1],2),flush=True)
    write(run/'validation/render_inventory.json',{'mode':a.mode,'fingerprint':fp,'jobs':jobs,'pairs':completed,'complete':True})
    app.close()

if __name__=='__main__':main()
