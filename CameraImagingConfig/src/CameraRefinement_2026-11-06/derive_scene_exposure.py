"""A locked render-exposure correction derived from public real tissue frames; not device exposure."""
from pathlib import Path
import json,hashlib
import cv2
import numpy as np
from PIL import Image
root=Path(__file__).parent;refdir=root/'sources/tissue_reference'
source=json.loads((refdir/'source.json').read_text())
roi=[.25,.25,.75,.75]
def luma(im):return np.asarray(im,dtype=np.float32)/255@np.array([.2126,.7152,.0722])
def region(im):
 h,w=im.shape[:2];return im[int(h*.25):int(h*.75),int(w*.25):int(w*.75)]
stats=[]
for eye in ['left','right']:
 cap=cv2.VideoCapture(str(refdir/(eye+'.mp4')))
 count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
 for idx in np.unique(np.linspace(0,count-1,5).astype(int)):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,bgr=cap.read();assert ok
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);y=luma(region(rgb))
  name=f'{eye}_{idx:05d}.png';Image.fromarray(rgb).save(refdir/name)
  stats.append({'eye':eye,'frame_index':int(idx),'video_frames':count,'video_fps':fps,'resolution':[rgb.shape[1],rgb.shape[0]],'median_roi_luma_encoded':float(np.median(y)),'p95_roi_luma_encoded':float(np.quantile(y,.95)),'image':name,'sha256':hashlib.sha256((refdir/name).read_bytes()).hexdigest()})
 cap.release()
target=float(np.median([s['median_roi_luma_encoded'] for s in stats]))
response=json.loads((root/'run02/config/effective_response.json').read_text())
linear=[region(np.load(root/f'run02/linear/0000_{eye}.npy').astype(np.float32))*response['eyes'][eye]['response_scale_rgb'] for eye in ['left','right']]
def statistic(g):
 ys=[]
 for a in linear:
  x=np.maximum(a*g,0.)
  encoded=np.clip(np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055),0,1)
  ys.append(np.median(encoded@np.array([.2126,.7152,.0722])))
 return float(np.median(ys))
lo,hi=0.,1.
while statistic(hi)<target:hi*=2
for _ in range(50):
 mid=(lo+hi)/2
 if statistic(mid)<target:lo=mid
 else:hi=mid
gain=(lo+hi)/2
report={'status':'data_derived_relative_render_exposure_NOT_physical_camera_exposure','source':source,
 'reference_sequence':'DRENDS ExVivo Seq01_Colon_Med','reference_frames':stats,
 'roi_fraction_xyxy':roi,'roi_selection_status':'explicit analysis choice: central half-width/half-height, not manufacturer parameter',
 'objective':'Match median encoded Rec.709-weighted RGB brightness of first simulated stereo pair central ROI to median of 10 public raw-video frame measurements; one common linear gain, frozen for all eyes and samples.',
 'target_median_encoded_luma':target,'before_median_encoded_luma':statistic(1.),'after_median_encoded_luma':statistic(gain),
 'linear_exposure_multiplier':gain,'render_exposure_compensation_ev':float(np.log2(gain)),
 'physical_exposure_seconds':None,'hardware_gain_db':None,'device_light_setting':None,
 'limitations':['Appearance normalization only; materials, geometry, illumination and camera response are not separated.','Reference sequence label Med does not calibrate simulated optical output to a physical 60% control setting.','Does not imply recovered real camera gamma, f-stop or exposure time.','White-reference color ratios retained; absolute white-reference level is intentionally changed by this scene-exposure correction.']}
(root/'scene_exposure.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['target_median_encoded_luma','before_median_encoded_luma','after_median_encoded_luma','linear_exposure_multiplier','render_exposure_compensation_ev']},indent=2))
