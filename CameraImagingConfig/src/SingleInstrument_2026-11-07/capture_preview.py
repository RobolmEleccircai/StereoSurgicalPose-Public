"""One final stereo preview, without redundant DOF reference or explicit update frames."""
import argparse
import hashlib
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--count', type=int, default=4)
parser.add_argument('--initial-profile', type=Path, default=Path('/home/leorobot/PoseProject/CameraImagingConfig/CameraRefinement_2026-11-06')/'InitialImaging_v1/initial_parameters.json')
parser.add_argument('--spp', type=int, default=32)
parser.add_argument('--renderer', default='PathTracing', choices=['PathTracing','RaytracedLighting'])
args = parser.parse_args()
initial = json.loads(args.initial_profile.read_text())
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
for d in ['images', 'ideal', 'ground_truth', 'validation', 'scene', 'config', 'linear', 'reference', 'baseline', 'depth']:
    (out / d).mkdir()
source = Path('/home/leorobot/sufia-bc/scenes/OR_scene_CTLiver-Prostate-Bladder/organs_psm_ecm.usda')
profile_path = Path('/home/leorobot/PoseProject/CameraImagingConfig/calibration/profiles/drends_exvivo_1080.json')
profile = json.loads(profile_path.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
protected = [source, source.parent/'organs_with_psm.usda', source.parent/'main_scene_isaac51.usda', profile_path]
before = {str(p): digest(p) for p in protected}

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'renderer': args.renderer,
    'width': 1920, 'height': 1080, 'multi_gpu': False, 'anti_aliasing': 0, 'samples_per_pixel_per_frame': args.spp, 'disable_viewport_updates': True})
import numpy as np
import shutil
map_config=json.loads((Path('/home/leorobot/PoseProject/CameraImagingConfig/CameraRefinement_2026-11-06')/'map_config.json').read_text())
map_data=np.load(Path('/home/leorobot/PoseProject/CameraImagingConfig/CameraRefinement_2026-11-06')/'raw_pixel_to_ideal_maps.npz')
ideal_w,ideal_h=map_config['ideal_resolution']
shutil.copy2(Path('/home/leorobot/PoseProject/CameraImagingConfig/CameraRefinement_2026-11-06')/'map_config.json',out/'config/map_config.json')
import carb
import omni.usd
import omni.replicator.core as rep
import omni.timeline
from pxr import Usd, UsdGeom, Gf, UsdLux, UsdShade, Sdf
from isaacsim.sensors.camera import Camera
from PIL import Image, ImageDraw

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

def world(path):
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

def set_local(path, mat):
    x = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    x.ClearXformOpOrder()
    op = x.GetPrim().GetAttribute('xformOp:transform')
    if op.IsValid():
        op.Set(Gf.Matrix4d(mat.T.tolist()))
        x.SetXformOpOrder([UsdGeom.XformOp(op)])
    else: x.AddTransformOp().Set(Gf.Matrix4d(mat.T.tolist()))

def shift_root(path, delta):
    stage.GetPrimAtPath(path).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*(original_translation[path]+delta)))

def project(xyz, eye):
    xyz = np.asarray(xyz)
    x, y = xyz[...,0]/xyz[...,2], xyz[...,1]/xyz[...,2]
    k1,k2,p1,p2,k3 = profile[eye]['D']
    r2=x*x+y*y
    radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xd=x*radial+2*p1*x*y+p2*(r2+2*x*x)
    yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    K=np.array(profile[eye]['K'])
    return np.stack([K[0,0]*xd+K[0,2],K[1,1]*yd+K[1,2]],axis=-1)

scene_path = out/'scene/stereo_preview.usda'
layer = Sdf.Layer.CreateNew(str(scene_path))
layer.subLayerPaths = [str(source)]
layer.Save()
assert omni.usd.get_context().open_stage(str(scene_path))
stage = omni.usd.get_context().get_stage()
stage.SetEditTarget(stage.GetRootLayer())
UsdGeom.SetStageMetersPerUnit(stage, .01)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
omni.timeline.get_timeline_interface().stop()
settings = carb.settings.get_settings()
for setting in ['chromaticAberration', 'lensFlares', 'tvNoise', 'motionblur', 'colorcorr', 'colorgrad']:
    settings.set_bool('/rtx/post/'+setting+'/enabled', False)
settings.set_bool('/rtx/post/histogram/enabled', False)
settings.set_int('/rtx/post/aa/op', 0)
settings.set_bool('/rtx/post/dof/enabled', True)
settings.set_bool('/rtx/post/dof/override', False)
settings.set_int('/rtx/pathtracing/spp', args.spp)
settings.set_int('/rtx/pathtracing/totalSpp', args.spp*4)
settings.set_bool('/rtx/pathtracing/optixDenoiser/enabled', True)
settings.set_bool('/rtx-transient/resourcemanager/enableTextureStreaming', True)
rep.orchestrator.set_capture_on_play(False)

stage.OverridePrim('/SurgicalTools/PSM_Left').SetActive(False)
roots = {'right_tool':'/SurgicalTools/PSM_Right', 'ecm':'/Endoscope/ECM'}
original_translation = {p:np.array(stage.GetPrimAtPath(p).GetAttribute('xformOp:translate').Get()) for p in roots.values()}
tip_paths = {'right_tool': roots['right_tool']+'/Robot/psm_tool_tip_link'}
ecm_tip = roots['ecm']+'/Robot/ecm_tip_link'
original_tip = {key: world(path)[:3,3].copy() for key,path in tip_paths.items()}
original_ecm_tip = world(ecm_tip)[:3,3].copy()

# Keep one complete official PSM instrument. Sample placement remains unchanged.
tool_targets = {'right_tool':np.array([-2.8,104.,-10.])}
for key in tool_targets: shift_root(roots[key], tool_targets[key]-original_tip[key])
ecm_target = np.array([-5.,112.,-16.])
shift_root(roots['ecm'], ecm_target-original_ecm_tip)

# Left optical frame: X right, Y down, Z forward along the verified ECM tip +Z.
# This is a preview mount convention, not a measured ECM-to-optics hand-eye transform.
rig_path = ecm_tip+'/StereoPreviewRig'
UsdGeom.Xform.Define(stage, rig_path)
T_tip = world(ecm_tip)
forward = T_tip[:3,2]/np.linalg.norm(T_tip[:3,2])
right = -T_tip[:3,1]/np.linalg.norm(T_tip[:3,1])
down = np.cross(forward,right)
Rwl=np.column_stack([right,down,forward])
Twl=np.eye(4); Twl[:3,:3]=Rwl; Twl[:3,3]=T_tip[:3,3]
mount = np.linalg.inv(T_tip)@Twl
set_local(rig_path,mount)  # Cancels inherited 100x asset scale; children use stage cm.
R=np.array(profile['stereo']['R_right_from_left'])
t=np.array(profile['stereo']['t_right_from_left_m'])
Tlr=np.eye(4); Tlr[:3,:3]=R.T; Tlr[:3,3]=-R.T@t/.01
C=np.diag([1.,-1.,-1.,1.])  # USD camera -> OpenCV optical frame
cameras={}
hdr_annotators={}
depth_annotators={}
for eye, optical_local in [('left',np.eye(4)),('right',Tlr)]:
    path=rig_path+'/'+eye.capitalize()
    cam=Camera(prim_path=path, resolution=(ideal_w,ideal_h), name=eye)
    cam.initialize()
    set_local(path,optical_local@C)
    cam.set_clipping_range(.05,1000.)
    cam.prim.GetAttribute('fStop').Set(0.)
    K=map_config['native_render_K_pixel_centers']
    cam.set_opencv_pinhole_properties(fx=K[0][0],fy=K[1][1],cx=K[0][2],cy=K[1][2],pinhole=[0.]*12)
    optical=initial['optics']['derived_intermediate_render']
    for attr in ['focalLength','horizontalAperture','verticalAperture','focusDistance','fStop']:
        cam.prim.GetAttribute(attr).Set(float(optical['usd_'+attr]))
    cameras[eye]=cam
    ann=rep.AnnotatorRegistry.get_annotator('HdrColor')
    ann.attach([cam.get_render_product_path()])
    hdr_annotators[eye]=ann
    depth_ann=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
    depth_ann.attach([cam.get_render_product_path()])
    depth_annotators[eye]=depth_ann

# Paper-inspired near-field lighting; no manufacturer radiometry is inferred.
disabled_lights=[]
for prim in stage.Traverse():
    if prim.HasAPI(UsdLux.LightAPI):
        UsdLux.LightAPI(prim).CreateIntensityAttr(0.)
        disabled_lights.append(str(prim.GetPath()))
for path in ['/World/OR_Room','/World/Operating_table']:
    prim=stage.GetPrimAtPath(path)
    if prim.IsValid(): UsdGeom.Imageable(prim).MakeInvisible()
# One shared effective source, midpoint of stereo optical centers. This is a model assumption.
light=UsdLux.SphereLight.Define(stage,rig_path+'/EndoscopePointLight')
light.CreateIntensityAttr(1000000.)  # Numerical gauge avoids float16 underflow; response absorbs scale.
light.CreateRadiusAttr(0.)
light.CreateTreatAsPointAttr(True)
light.CreateNormalizeAttr(True)
light.CreateColorAttr(Gf.Vec3f(1,1,1))
light.CreateExposureAttr(0.)
light_matrix=C.copy();light_matrix[:3,3]=Tlr[:3,3]/2
set_local(str(light.GetPath()),light_matrix)
shaping=UsdLux.ShapingAPI.Apply(light.GetPrim())
shaping.CreateShapingConeAngleAttr(90.)  # Forward hemisphere, NOT measured endoscope beam angle.
shaping.CreateShapingConeSoftnessAttr(0.)
shaping.CreateShapingFocusAttr(0.)  # No unverified angular exponent is added.
ref=json.loads(Path('/home/leorobot/PoseProject/CameraImagingConfig/CameraRefinement_2026-11-06/photometric_reference.json').read_text())
write_json(out/'config/photometric_reference.json',ref)

def encode_srgb(x):
    x=np.maximum(x,0.)
    return np.where(x<=.0031308,12.92*x,1.055*np.power(x,1/2.4)-.055)
def decode_srgb(x):
    x=np.asarray(x)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
response_scale={eye:np.array(initial['illumination']['combined_rgb_scales'][eye]) for eye in cameras}
camera_public={k:profile[k] for k in ['image','optical_frame_axes','left','right','source']}
write_json(out/'config/camera_intrinsics.json', camera_public)
write_json(out/'ground_truth/stereo_extrinsics.json', profile['stereo'])
manifest={
    'profile_id':profile['profile_id'], 'source_scene':str(source),
    'instrument_configuration':{'count':1,'active_tools':['right_tool'],
        'active_prim':'/SurgicalTools/PSM_Right','inactive_prim':'/SurgicalTools/PSM_Left',
        'ecm_retained':True,'stereo_camera_count':2},
    'status':'initial_optics_native_dof_pending_target_camera_validation', 'simulator':'Isaac Sim 5.1.0',
    'resolution_per_eye':[1920,1080], 'lens_model':'OpenCV raw distortion via inverse ray map from native overscanned ideal rendering', 'render_adapter':map_config,
    'samples':[], 'stage_meters_per_unit':.01,
    'sampling':{'continuous_video':False,'capture_fps':None,'physics_enabled':False,
        'synchronization':'Both render products captured from one frozen stage with delta_time=0'},
    'mount':{'parent':ecm_tip,'source':'preview assumption, not measured official mount',
        'left_center':'ECM tip frame origin','optical_z':'ECM tip +Z','optical_x':'ECM tip -Y',
        'local_matrix_with_scale_compensation':mount.tolist()},
    'preview_choices':{'dof_disabled':False,'sensor_noise_added':False,'motion_blur':False,
        'physical_exposure_gain_whitebalance_calibrated':False,
        'light':{'type':'shared ideal point source','intensity_gauge':1000000.,'distance_law':'inverse square',
                 'location':'stereo-center midpoint','cone_half_angle_deg':90.,'angular_focus':0.,
                 'provenance':'paper-inspired geometry; hemisphere and uniform angular emission are explicit neutral approximations'},
        'existing_scene_lighting_retained':False,'disabled_light_paths':disabled_lights,
        'room_and_table_hidden':True,
        'tool_and_camera_positions':'static variations; no RCM/joint feasibility validation'},
    'labels':{'tip':'official psm_tool_tip_link origin (tool control point, not an inferred extreme mesh vertex)',
        'shaft':'normalized +Z of psm_tool_roll_link, pointing toward distal end',
        '6d_pose':'official tip-link frame; supplied separately from shaft direction',
        'pixel_projection':'OpenCV five-parameter raw distorted projection',
        'visibility':'inside_image is geometric only; no occlusion visibility claim'},
    'source_files_sha256_before':before,
}
def raw_from_ideal(im,eye):
    uv=map_data[eye];x=uv[:,:,0];y=uv[:,:,1]
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    assert x0.min()>=0 and y0.min()>=0 and x0.max()+1<im.shape[1] and y0.max()+1<im.shape[0]
    wx=(x-x0)[:,:,None];wy=(y-y0)[:,:,None]
    rgb=(1-wy)*((1-wx)*im[y0,x0]+wx*im[y0,x0+1])+wy*((1-wx)*im[y0+1,x0]+wx*im[y0+1,x0+1])
    return rgb.astype(np.float32)

last_ideal={};last_linear={};last_depth={}
write_json(out/'config/initial_parameters.json',initial)
write_json(out/'config/effective_response.json',{'physical_parameters_calibrated':False,'scene_exposure_multiplier':16.0,'eyes':{eye:{'response_scale_rgb':response_scale[eye].tolist()} for eye in cameras}})
manifest['renderer']=args.renderer
manifest['preview_execution']={'dof_reference_capture':False,'explicit_extra_update_frames':0,'replicator_wait_for_render':True,'reason':'Only the requested final stereo pair is captured; render sampling and imaging parameters retained.'}
manifest['render_sampling']={'spp_per_frame':args.spp,'total_spp_setting':args.spp*4,'optix_denoiser_enabled':True,'unused_viewport_updates_disabled':True}
manifest['native_dof_readback']={eye:{attr:cam.prim.GetAttribute(attr).Get() for attr in ['focalLength','horizontalAperture','verticalAperture','focusDistance','fStop']} for eye,cam in cameras.items()}
manifest['initial_profile']=initial
def capture():
    print('CAPTURE_START',time.strftime('%H:%M:%S'),flush=True)
    rep.orchestrator.step(rt_subframes=4,pause_timeline=True,delta_time=0.0,wait_for_render=True)
    last_ideal.clear();last_linear.clear();last_depth.clear()
    result={}
    for eye,cam in cameras.items():
        dep=depth_annotators[eye].get_data()
        if isinstance(dep,dict):dep=dep['data']
        dep=np.asarray(dep).squeeze()
        uv=map_data[eye]
        last_depth[eye]=dep[np.rint(uv[:,:,1]).astype(int),np.rint(uv[:,:,0]).astype(int)]
        hdr=hdr_annotators[eye].get_data()
        if isinstance(hdr,dict):hdr=hdr['data']
        hdr=np.asarray(hdr)[...,:3].astype(np.float32)
        assert hdr.shape==(ideal_h,ideal_w,3),hdr.shape
        assert np.isfinite(hdr).all()
        linear=raw_from_ideal(hdr,eye)
        last_linear[eye]=linear
        last_ideal[eye]=np.uint8(np.clip(np.rint(encode_srgb(hdr*response_scale[eye])*255),0,255))
        result[eye]=np.uint8(np.clip(np.rint(encode_srgb(linear*response_scale[eye])*255),0,255))
    print('CAPTURE_END',time.strftime('%H:%M:%S'),flush=True)
    return result

print('CAMERAS_READY',time.strftime('%H:%M:%S'),flush=True)
# Use frozen initial response; avoid changing it by re-fitting a white patch.
excluded_ecm_meshes=[]
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith(roots['ecm']+'/') and prim.IsA(UsdGeom.Mesh):
        UsdGeom.Imageable(prim).MakeInvisible()
        excluded_ecm_meshes.append(str(prim.GetPath()))
stage.OverridePrim('/WhiteReference').SetActive(False)
manifest['ecm_render_exclusion']={'mesh_paths':excluded_ecm_meshes,'status':'existing explicit approximation, mesh has no calibrated optical openings'}
manifest['render_pipeline']='native aperture DOF -> HdrColor -> existing distorted ray map -> fixed RGB response with +4 EV -> sRGB; field blur/noise applied in a separately recorded step'
print('INITIAL_OPTICS_READY',json.dumps(manifest['native_dof_readback']),flush=True)
variants=[([0,0,0],[0,0,0]),
          ([.5,-.2,-.3],[.5,0,0]),
          ([-.4,.5,.4],[-.4,.6,-.3]),
          ([.2,.1,-.4],[.2,-.4,.4])]
for idx,(dr,dc) in enumerate(variants[:args.count]):
    for key,delta in [('right_tool',dr)]:
        shift_root(roots[key],tool_targets[key]+np.array(delta)-original_tip[key])
    shift_root(roots['ecm'],ecm_target+np.array(dc)-original_ecm_tip)
    Tw={eye:world(cam.prim_path)@C for eye,cam in cameras.items()}
    baseline=float(np.linalg.norm(Tw['right'][:3,3]-Tw['left'][:3,3])*.01)
    assert abs(baseline-profile['stereo']['baseline_m'])<1e-8
    relative=np.linalg.inv(Tw['right'])@Tw['left']
    assert np.allclose(relative[:3,:3],R,atol=1e-7)
    assert np.allclose(relative[:3,3]*.01,t,atol=1e-8)
    print('SAMPLE_START',idx,time.strftime('%H:%M:%S'),flush=True)
    print('NATIVE_DOF_ON',idx,flush=True)
    images=capture()
    record={'sample_id':f'{idx:04d}','simulation_time_s':omni.timeline.get_timeline_interface().get_current_time(),
            'static_state_id':idx,'baseline_m':baseline,'cameras':{},'tools':{}}
    for eye in cameras:
        T=Tw[eye].copy();T[:3,3]*=.01
        record['cameras'][eye]={'T_world_from_optical_m':T.tolist()}
        Image.fromarray(images[eye]).save(out/f'images/{idx:04d}_{eye}.png')
        Image.fromarray(last_ideal[eye]).save(out/f'ideal/{idx:04d}_{eye}.png')
        np.save(out/f'linear/{idx:04d}_{eye}.npy',last_linear[eye].astype(np.float16))
        np.save(out/f'depth/{idx:04d}_{eye}.npy',last_depth[eye].astype(np.float32))
    for key,path in tip_paths.items():
        Ttool=world(path); Ttool[:3,:3]/=np.linalg.norm(Ttool[:3,0])
        shaft=world(roots[key]+'/Robot/psm_tool_roll_link')[:3,2]
        shaft/=np.linalg.norm(shaft)
        info={}
        for eye in cameras:
            Tcl=np.linalg.inv(Tw[eye]); Tpose=Tcl@Ttool;Tpose[:3,3]*=.01
            xyz=Tpose[:3,3];axis=Tcl[:3,:3]@shaft
            uv=project(xyz,eye)
            # A 30 mm reference segment lies proximally along the shaft line from roll-link origin.
            wrist=(Tcl@world(roots[key]+'/Robot/psm_tool_roll_link'))[:3,3]*.01
            shaft_uv=project(np.array([wrist-.03*axis,wrist]),eye)
            info[eye]={'tip_position_m':xyz.tolist(),'shaft_direction_unit':axis.tolist(),
                'T_camera_from_tip_link_m':Tpose.tolist(),'tip_pixel_raw':uv.tolist(),
                'shaft_segment_pixels_raw':shaft_uv.tolist(),
                'inside_image':bool(xyz[2]>0 and 0<=uv[0]<1920 and 0<=uv[1]<1080)}
        record['tools'][key]=info
    write_json(out/f'ground_truth/{idx:04d}.json',record)
    stage.GetRootLayer().Export(str(out/f'scene/sample_{idx:04d}.usda'))
    manifest['samples'].append(record['sample_id'])
    print('CAPTURED',idx,'baseline_mm',baseline*1000,'means',{k:float(v.mean()) for k,v in images.items()},flush=True)
    write_json(out/'manifest.json',manifest)

# Save the final sample stage; individual sample layers are saved above.
stage.GetRootLayer().Save()

after={str(p):digest(p) for p in protected}
assert before==after,'Protected source changed'
manifest['source_files_sha256_after']=after
manifest['source_files_unchanged']=True
manifest['completed']=True
write_json(out/'manifest.json',manifest)
print('COMPLETE',str(out),flush=True)
app.close()
