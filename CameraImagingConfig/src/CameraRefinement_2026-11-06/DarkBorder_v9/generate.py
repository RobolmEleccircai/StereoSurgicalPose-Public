from pathlib import Path
import argparse, json, hashlib, shutil, copy
import numpy as np
import cv2
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
src=a.source.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
for d in ['images','config','ground_truth','gain_maps','preview']:(out/d).mkdir(parents=True)
profile=json.loads(Path(__file__).with_name('profiles.json').read_text());cfg=json.loads((src/'config/applied_parameters.json').read_text());intr=json.loads((src/'config/camera_intrinsics.json').read_text());m=json.loads((src/'native/manifest.json').read_text());assert m['completed']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=[*sorted((src/'native/linear').glob('*.npy')),*sorted((src/'ground_truth').glob('*')),*sorted((src/'images').glob('*.png')),src/'config/applied_parameters.json',src/'config/camera_intrinsics.json']
before={str(p):sha(p) for p in protected}
for d in ['config','ground_truth']:
 for f in (src/d).iterdir():
  if f.is_file():shutil.copy2(f,out/d/f.name)

def srgb(x):
 x=np.maximum(x,0);return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)
def u8(x):return np.clip(np.rint(x),0,255).astype(np.uint8)
def blur_field(x,eye):
 h,w=x.shape[:2];cy=intr[eye]['K'][1][2];cx=intr[eye]['K'][0][2];yy,xx=np.indices((h,w),dtype=np.float32)
 far=max(np.hypot(px-cx,py-cy) for px,py in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)])
 r=np.clip(np.hypot(xx-cx,yy-cy)/far,0,1);sigma=np.sqrt(.3**2+(.8*r*r)**2)
 levels=np.linspace(float(sigma.min()),float(sigma.max()),17);step=levels[1]-levels[0];result=np.zeros_like(x,dtype=np.float32)
 for s in levels:result+=np.maximum(1-np.abs(sigma-s)/step,0).astype(np.float32)[:,:,None]*cv2.GaussianBlur(x,(0,0),float(s),borderType=cv2.BORDER_REFLECT_101)
 return result

yy,xx=np.indices((1080,1920),dtype=np.float64);cx,cy=profile['center_pixel'];rx,ry=profile['semiaxes_pixel'];radius=np.sqrt(((xx-cx)/rx)**2+((yy-cy)/ry)**2)
global_gain=float(profile['global_linear_multiplier']);maps={}
for name,par in profile['profiles'].items():
 t=np.clip((radius-par['start_radius'])/(par['end_radius']-par['start_radius']),0,1);maps[name]=(1-(1-par['outer_linear_gain'])*t*t*(3-2*t)).astype(np.float32)
 np.save(out/f'gain_maps/{name}.npy',maps[name]);Image.fromarray(u8(maps[name]*255)).save(out/f'gain_maps/{name}.png')
report={'completed':False,'source':str(src),'new_pathtracing_render':False,'reused_pathtracing_settings':m['render_sampling'],'profiles':profile,'selected':'border_3p5_global075','global_linear_multiplier':global_gain,'effective_outer_linear_gain':global_gain*.035,'source_files_sha256_before':before,'frames':[]}
for idx,sid in enumerate(m['samples']):
 for ei,eye in enumerate(['left','right']):
  linear=np.load(src/f'native/linear/{sid}_{eye}.npy').astype(np.float32);base=blur_field(linear,eye)*np.asarray(cfg['illumination']['combined_rgb_scales'][eye],dtype=np.float32)
  seed=int(cfg['noise']['seed_base'])+idx*2+ei
  for name in (list(maps) if idx==0 else ['border_3p5_global075']):
   atten=base*maps[name][:,:,None]*global_gain;encoded=np.clip(srgb(atten),0,1);dn=encoded*255;y=encoded@np.array([.2126,.7152,.0722],dtype=np.float32)
   rng=np.random.default_rng(seed);pixel=rng.standard_normal(dn.shape,dtype=np.float32)*np.sqrt(1+6*y)[:,:,None];row=rng.normal(0,.3,(1080,1,1)).astype(np.float32);im=u8(dn+pixel+row)
   path=out/(f'images/{sid}_{eye}.png' if name=='border_3p5_global075' else f'candidates/{name}/{sid}_{eye}.png');Image.fromarray(im).save(path)
   if name=='border_3p5_global075':
    old=np.asarray(Image.open(src/f'images/{sid}_{eye}.png'));center=radius<=.30;outer=radius>=1.10
    center_delta=int(np.abs(im[center].astype(np.int16)-old[center].astype(np.int16)).max());ratio=float(atten[outer].sum(dtype=np.float64)/base[outer].sum(dtype=np.float64))
    center_ratio=float(atten[center].sum(dtype=np.float64)/base[center].sum(dtype=np.float64))
    center_encoded=np.clip(srgb(base*global_gain),0,1);center_dn=center_encoded*255;center_y=center_encoded@np.array([.2126,.7152,.0722],dtype=np.float32)
    reference_rng=np.random.default_rng(seed);reference_noise=reference_rng.standard_normal(center_dn.shape,dtype=np.float32)*np.sqrt(1+6*center_y)[:,:,None];reference_row=reference_rng.normal(0,.3,(1080,1,1)).astype(np.float32)
    center_reference=u8(center_dn+reference_noise+reference_row)
    center_reference_delta=int(np.abs(im[center].astype(np.int16)-center_reference[center].astype(np.int16)).max())
    assert center_reference_delta==0 and abs(center_ratio-global_gain)<1e-6 and abs(ratio-.035*global_gain)<1e-6
    border=np.concatenate([im[:12].reshape(-1,3),im[-12:].reshape(-1,3),im[:,:12].reshape(-1,3),im[:,-12:].reshape(-1,3)])
    report['frames'].append({'sample':sid,'eye':eye,'center_change_from_source_max_dn':center_delta,'center_matches_global_only_max_difference_dn':center_reference_delta,'center_effective_linear_gain':center_ratio,'outer_effective_linear_gain':ratio,'outer_mask_gain':.035,'border_12px_median_dn':float(np.median(border)),'border_12px_p95_dn':float(np.percentile(border,95)),'mean_rgb_dn':im.mean((0,1)).tolist(),'sha256':sha(path),'seed':seed})
    print(sid,eye,'border_median',float(np.median(border)),'outer_effective_gain',ratio,flush=True)
assert before=={str(p):sha(p) for p in protected}
report['source_unchanged']=True;report['completed']=True;report['all_checks_passed']=True
(out/'effect_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
applied=copy.deepcopy(cfg);applied['profile_id']='initial_imaging_with_complete_border_3p5_global075';applied['status']='debugging_initial_not_fitted_to_real_camera';applied['field_edge_attenuation']=profile;applied['processing_order']='existing native DoF HDR -> distortion map (already in cache) -> existing field blur -> source fixed RGB x16 response -> border_3p5_global075 elliptical gain -> global linear x0.75 (effective RGB x12) -> sRGB -> original noise model and seeds -> uint8'
applied['global_brightness_adjustment']={'linear_multiplier':global_gain,'ev_delta':float(np.log2(global_gain)),'application':'Applied once after mask in generator, BEFORE sRGB/noise. Already folded into effective illumination scales below; do not apply a second time when using these scales.','source_post_hdr_linear_multiplier':cfg['illumination']['post_hdr_linear_multiplier'],'source_combined_rgb_scales':cfg['illumination']['combined_rgb_scales'],'center_effective_linear_gain':global_gain,'outer_effective_linear_gain':.035*global_gain}
applied['illumination']['post_hdr_linear_multiplier']=cfg['illumination']['post_hdr_linear_multiplier']*global_gain
applied['illumination']['post_hdr_exposure_ev']=float(np.log2(applied['illumination']['post_hdr_linear_multiplier']))
applied['illumination']['combined_rgb_scales']={eye:[float(v*global_gain) for v in scales] for eye,scales in cfg['illumination']['combined_rgb_scales'].items()}
applied['illumination']['provenance']='Base white-reference RGB scales retained; effective response includes source x16 and global x0.75, total x12. These effective scales already include the global adjustment. Not hardware gain or exposure.'
applied['configuration_acceptance']=profile.get('configuration_acceptance')
(out/'config/applied_parameters.json').write_text(json.dumps(applied,ensure_ascii=False,indent=2)+'\n');(out/'config/dark_border_parameters.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps({'samples':m['samples'],'completed':True,'source_run':'../run05_initial','selected_profile':'border_3p5_global075','resolution_per_eye':[1920,1080],'source_renderer':'PathTracing','new_scene_render_performed':False,'global_linear_multiplier':global_gain,'outer_mask_gain':.035,'outer_effective_linear_gain':.035*global_gain},indent=2)+'\n')
print('COMPLETE',out)
