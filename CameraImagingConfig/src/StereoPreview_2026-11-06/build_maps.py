"""Map each desired raw pixel to its ideal ray using the published OpenCV K/D."""
from pathlib import Path
import cv2
import numpy as np
import json
import argparse
root=Path(__file__).parent
default_profile=root.parent/'calibration/profiles/drends_exvivo_1080.json'
if not default_profile.exists():
    default_profile=root.parent/'PoseProject/CameraImagingConfig/calibration/profiles/drends_exvivo_1080.json'
parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,default=default_profile);args=parser.parse_args()
p=json.loads(args.profile.read_text())
w,h=1920,1080
y,x=np.indices((h,w),dtype=np.float64)
pixels=np.stack([x,y],-1).reshape(-1,1,2)
rays={}
for eye in ['left','right']:
    K=np.array(p[eye]['K']);D=np.array(p[eye]['D'])
    rays[eye]=cv2.undistortPointsIter(pixels,K,D,None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(h,w,2)
f=1600.
cx=float(np.ceil((max(np.abs(r[:,:,0]).max() for r in rays.values())*f+24)/32)*32)
cy=float(np.ceil((max(np.abs(r[:,:,1]).max() for r in rays.values())*f+24)/32)*32)
maps={eye:(r*np.array([f,f])+np.array([cx,cy])).astype(np.float32) for eye,r in rays.items()}
errors={}
for eye,r in rays.items():
    sample=r[::20,::20].reshape(-1,2)
    xyz=np.c_[sample,np.ones(len(sample))]
    uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),np.array(p[eye]['K']),np.array(p[eye]['D']))
    errors[eye]=float(np.linalg.norm(uv.reshape(-1,2)-pixels.reshape(h,w,2)[::20,::20].reshape(-1,2),axis=1).max())
np.savez_compressed(root/'raw_pixel_to_ideal_maps.npz',**maps)
meta={'ideal_resolution':[int(cx*2),int(cy*2)],'ideal_K':[[f,0,cx],[0,f,cy],[0,0,1]],
      'native_render_K_pixel_centers':[[f,0,cx+.5],[0,f,cy+.5],[0,0,1]],
      'source_profile':p['profile_id'],'opencv_version':cv2.__version__,
      'mapping':'For each raw distorted output pixel, invert published OpenCV K,D, then sample the same ray in the overscanned ideal render.',
      'native_distortion_coefficients':[0.]*12,'max_inverse_mapping_error_px':errors,
      'note':'The 0.5 native principal-point offset reconciles raster pixel centers. These intermediate parameters are numerical rendering parameters, not physical dVRK specifications.'}
(root/'map_config.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
