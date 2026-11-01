"""Apply the explicitly chosen field blur/noise to newly rendered HDR images."""
from pathlib import Path
import argparse, json, shutil, hashlib
import numpy as np
import cv2
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
src=a.native;out=a.out;out.mkdir(parents=True,exist_ok=True)
assert not (out/'images').exists(),'Do not overwrite delivered images'
for d in ['images','stages/01_native_dof','stages/02_field_blur','ground_truth','config']:(out/d).mkdir(parents=True,exist_ok=True)
cfg=json.loads((src/'config/initial_parameters.json').read_text());intr=json.loads((src/'config/camera_intrinsics.json').read_text());manifest=json.loads((src/'manifest.json').read_text())
for d in ['ground_truth','config']:
 for f in (src/d).iterdir():
  if f.is_file():shutil.copy2(f,out/d/f.name)
shutil.copy2(src/'manifest.json',out/'config/native_manifest.json')

def srgb(x):
 x=np.maximum(x,0);return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)
def u8(x):return np.clip(np.rint(x),0,255).astype(np.uint8)

def blur_field(x,eye):
 h,w=x.shape[:2];cy=intr[eye]['K'][1][2];cx=intr[eye]['K'][0][2];yy,xx=np.indices((h,w),dtype=np.float32)
 far=max(np.hypot(px-cx,py-cy) for px,py in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)])
 r=np.clip(np.hypot(xx-cx,yy-cy)/far,0,1)
 sigma=np.sqrt(cfg['field_blur']['in_focus_base_sigma_px']**2+(cfg['field_blur']['additional_corner_sigma_px']*r*r)**2)
 levels=np.linspace(float(sigma.min()),float(sigma.max()),17)
 result=np.zeros_like(x,dtype=np.float32)
 step=levels[1]-levels[0]
 for s in levels:
  weight=np.maximum(1-np.abs(sigma-s)/step,0).astype(np.float32)
  filtered=cv2.GaussianBlur(x,(0,0),float(s),borderType=cv2.BORDER_REFLECT_101)
  result+=weight[:,:,None]*filtered
 return result,{'sigma_min_px':float(sigma.min()),'sigma_max_px':float(sigma.max()),'gaussian_bank_levels':len(levels),'interpolation':'linear between adjacent Gaussian levels; filtering in linear RGB'}

report={'native_run':str(src),'initial_profile':cfg['profile_id'],'effects_are_initial_assumptions':True,'physical_camera_matched':False,'motion_blur_applied':False,'native_dof':'Isaac Sim PathTracing camera aperture; see native manifest readback','frames':[]}
for idx,sid in enumerate(manifest['samples']):
 for ei,eye in enumerate(['left','right']):
  linear=np.load(src/f'linear/{sid}_{eye}.npy').astype(np.float32)
  assert linear.shape==(1080,1920,3) and np.isfinite(linear).all()
  scale=np.asarray(cfg['illumination']['combined_rgb_scales'][eye],dtype=np.float32)
  native=u8(srgb(linear*scale)*255)
  blurred,binfo=blur_field(linear,eye)
  encoded=np.clip(srgb(blurred*scale),0,1);dn=encoded*255
  y=encoded@np.array([.2126,.7152,.0722],dtype=np.float32)
  sigma=np.sqrt(1+6*y)
  seed=int(cfg['noise']['seed_base'])+idx*2+ei;rng=np.random.default_rng(seed)
  pixel=rng.standard_normal(dn.shape,dtype=np.float32)*sigma[:,:,None]
  row=rng.normal(0,cfg['noise']['row_offset_sigma_dn'],(1080,1,1)).astype(np.float32)
  final=u8(dn+pixel+row)
  Image.fromarray(native).save(out/f'stages/01_native_dof/{sid}_{eye}.png')
  Image.fromarray(u8(dn)).save(out/f'stages/02_field_blur/{sid}_{eye}.png')
  Image.fromarray(final).save(out/f'images/{sid}_{eye}.png')
  mid=(y>.45)&(y<.55)
  entry={'sample':sid,'eye':eye,'seed':seed,'field_blur':binfo,'added_pixel_noise_midgray_std_dn':float(pixel[mid].std()) if mid.any() else None,'added_row_noise_std_dn':float(row.std()),'mean_rgb_dn':final.mean((0,1)).tolist(),'saturated_channel_fraction':float((final>=255).mean()),'zero_channel_fraction':float((final==0).mean()),'png_sha256':hashlib.sha256((out/f'images/{sid}_{eye}.png').read_bytes()).hexdigest()}
  report['frames'].append(entry)
  print(sid,eye,'noise_midgray_dn',entry['added_pixel_noise_midgray_std_dn'],'mean',round(float(final.mean()),2),flush=True)
report['completed']=True
(out/'effect_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
cfg['status']='rendered_initial_trial_not_fitted_to_target_camera';cfg['timing']['static_preview_motion_blur']=False
(out/'config/applied_parameters.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
print('COMPLETE',out)
