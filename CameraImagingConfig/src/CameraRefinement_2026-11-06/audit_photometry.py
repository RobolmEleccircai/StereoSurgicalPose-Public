"""Audit author-published calibration images; do not overwrite geometric calibration."""
from pathlib import Path
import json, hashlib
import cv2
import numpy as np
from scipy.optimize import least_squares

root=Path(__file__).parent
source=root/'sources/calibration_images/Calibration'
profile=json.loads((root.parent/'PoseProject/CameraImagingConfig/calibration/profiles/drends_exvivo_1080.json').read_text())
out=root/'photometric_audit';out.mkdir(exist_ok=True)
obj=np.zeros((12,3),np.float32);obj[:,:2]=np.mgrid[0:4,0:3].T.reshape(-1,2)*.015
records=[];samples=[]
for eye in ['left','right']:
 K=np.array(profile[eye]['K']);D=np.array(profile[eye]['D'])
 for file in sorted((source/eye).glob('*.png')):
  im=cv2.imread(str(file));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  ok,corners=cv2.findChessboardCornersSB(gray,(4,3),cv2.CALIB_CB_NORMALIZE_IMAGE)
  if not ok:
   records.append(dict(eye=eye,file=file.name,found=False));continue
  ok,rv,tv=cv2.solvePnP(obj,corners,K,D)
  rot=cv2.Rodrigues(rv)[0];pr=cv2.projectPoints(obj,rv,tv,K,D)[0]
  err=float(np.sqrt(np.mean(np.sum((pr.reshape(-1,2)-corners.reshape(-1,2))**2,axis=-1))))
  if err>2.:
   records.append(dict(eye=eye,file=file.name,found=True,reprojection_rms_px=err,used=False));continue
  frame=[]
  # Internal cell interiors only: known planar surface, far from printed edges.
  for x in range(3):
   for y in range(2):
    pts=np.array([[(x+u)*.015,(y+v)*.015,0] for u in [.3,.5,.7] for v in [.3,.5,.7]])
    uv=cv2.projectPoints(pts,rv,tv,K,D)[0].reshape(-1,2)
    values=[]
    for u,v in uv:
     patch=im[int(v)-3:int(v)+4,int(u)-3:int(u)+4]
     values.append(np.median(patch.reshape(-1,3),axis=0)[::-1]/255.)
    frame.append((pts,np.array(values),uv))
  cell_luma=np.array([np.mean(v) for p,v,u in frame])
  cutoff=(cell_luma.max()+cell_luma.min())/2
  rgb=[]
  for pts,values,uv in frame:
   if values.mean()<cutoff:continue
   xyz=pts@rot.T+tv.reshape(1,3);dist=np.linalg.norm(xyz,axis=1)
   cosine=np.abs(xyz@rot[:,2])/dist
   offaxis=xyz[:,2]/dist
   for j,val in enumerate(values):
    samples.append(dict(eye=eye,file=file.name,rgb=val.tolist(),distance_m=float(dist[j]),cos_normal=float(cosine[j]),cos_axis=float(offaxis[j]),pixel=uv[j].tolist()))
   rgb.extend(values)
  records.append(dict(eye=eye,file=file.name,found=True,used=True,reprojection_rms_px=err,white_rgb_median=np.median(rgb,axis=0).tolist(),normal=rot[:,2].tolist(),translation_m=tv.ravel().tolist()))
  print(eye,file.name,'rms',round(err,3),flush=True)

fits={}
for eye in ['left','right']:
 ss=[s for s in samples if s['eye']==eye]
 keys=sorted(set(s['file'] for s in ss));ids=np.array([keys.index(s['file']) for s in ss])
 rgb=np.array([s['rgb'] for s in ss]);Y=rgb@np.array([.2126,.7152,.0722])
 A=np.log([s['cos_normal']/s['distance_m']**2 for s in ss]);B=np.log([s['cos_axis'] for s in ss])
 logy=np.log(Y)
 # Free per-image intercept absorbs unknown exposure/gain and absolute light scale.
 def residual(p):return p[0]*A+p[1]*B+p[2:][ids]-logy
 initial=np.r_[1/2.2,1.,np.array([np.mean(logy[ids==i]-A[ids==i]/2.2-B[ids==i]) for i in range(len(keys))])]
 fit=least_squares(residual,initial,loss='soft_l1',f_scale=.02,max_nfev=300)
 a,b=fit.x[:2];pred=np.exp(logy+residual(fit.x))
 fits[eye]=dict(gamma=float(1/a),angular_exponent=float(b/a),rmse_gray=float(np.sqrt(np.mean((pred-Y)**2))*255),image_count=len(keys),sample_count=len(ss),white_rgb_median=np.median(rgb,axis=0).tolist(),white_luma_median=float(np.median(Y)),note='Diagnostic fit assumes Lambertian board and coaxial point lighting; not an accepted camera calibration.')
report=dict(source='DRENDS author-linked ExVivo calibration images',archive_sha256=hashlib.sha256((root/'sources/DRENDS_ExVivo_Calibration.zip').read_bytes()).hexdigest(),geometry_unchanged=True,records=records,diagnostic_fits=fits)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'white_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
print(json.dumps(fits,indent=2))
