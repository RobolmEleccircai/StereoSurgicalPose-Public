"""One documented real white patch defines relative display level, not physical exposure."""
from pathlib import Path
import hashlib,json
import cv2
import numpy as np
root=Path(__file__).parent
prof=json.loads((root.parent/'PoseProject/CameraImagingConfig/calibration/profiles/drends_exvivo_1080.json').read_text())
audit=json.loads((root/'photometric_audit/report.json').read_text())
samples=json.loads((root/'photometric_audit/white_samples.json').read_text())
eligible=[s for s in samples if s['eye']=='left' and s['file']=='00000.png']
patch=max(eligible,key=lambda s:s['cos_axis'])
K=np.array(prof['left']['K']);D=np.array(prof['left']['D'])
record=next(s for s in audit['records'] if s['eye']=='left' and s['file']==patch['file'])
ray=cv2.undistortPointsIter(np.array(patch['pixel']).reshape(1,1,2),K,D,None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(2)
ray=np.r_[ray,1];xyz=ray/np.linalg.norm(ray)*patch['distance_m']
rgb={};uvs={};hashes={}
for eye in ['left','right']:
 pt=xyz if eye=='left' else np.array(prof['stereo']['R_right_from_left'])@xyz+np.array(prof['stereo']['t_right_from_left_m'])
 uv=cv2.projectPoints(pt.reshape(1,3),np.zeros(3),np.zeros(3),np.array(prof[eye]['K']),np.array(prof[eye]['D']))[0].reshape(2)
 file=root/'sources/calibration_images/Calibration'/eye/patch['file'];im=cv2.imread(str(file))
 x,y=np.rint(uv).astype(int);roi=im[y-3:y+4,x-3:x+4]
 rgb[eye]=(np.median(roi.reshape(-1,3),axis=0)[::-1]/255).tolist();uvs[eye]=uv.tolist()
 hashes[eye]=hashlib.sha256(file.read_bytes()).hexdigest()
 cv2.imwrite(str(root/'photometric_audit'/f'{eye}_reference_patch.png'),im[y-60:y+61,x-60:x+61])
result={'status':'single_patch_relative_response_reference_NOT_radiometric_calibration',
 'source':'DRENDS author-linked ExVivo calibration archive; same published K,D,R,t used without refitting',
 'source_frame':patch['file'],'source_sha256':hashes,'source_pixels':uvs,
 'archive_sha256':audit['archive_sha256'],'point_in_left_camera_m':xyz.tolist(),
 'board_normal_in_left_camera':record['normal'],'target_encoded_rgb_0_1':rgb,
 'display_transfer':'IEC sRGB encoding convention, NOT an estimated physical camera response',
 'method':'Use a unit-albedo diffuse patch at the recovered board point and normal; choose per-eye RGB multipliers so the patch matches source encoded RGB after sRGB display encoding. Lock multipliers across all captures.',
 'unknowns':{'physical_exposure_s':None,'hardware_gain_db':None,'hardware_white_balance':None,'physical_gamma':None,'angular_falloff_exponent':None},
 'limitations':['Board illumination is not labelled 60% or 80%.','Reference material reflectance and light intensity are not separated.','No physical gamma, vignetting or sensor noise is inferred.','Static board pose is used only for relative display matching, not to change camera extrinsics.']}
(root/'photometric_reference.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
