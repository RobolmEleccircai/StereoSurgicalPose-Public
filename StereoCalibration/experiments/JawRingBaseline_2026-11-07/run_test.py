"""Reproduce the requested single-pair baseline test. Never edits the old project.

Inputs: newly rendered PNG pair, supplied intrinsics/distortion, ROI-assisted
image detections, and metric CAD A-B length. Estimate is saved before loading
the stereo extrinsic truth for evaluation. This is not a successful calibration.
"""
from pathlib import Path
import hashlib
import json
import sys
import cv2
import numpy as np

root = Path(__file__).resolve().parent
old = Path('/home/leorobot/capstone-1/Grasp5-Stereo-SurgPose/pixel_extraction')
sys.path.insert(0,str(old))
import geometry

cams=json.loads((root/'intrinsics.json').read_text())['cameras']
known=json.loads((root/'known_AB.json').read_text())
det=json.loads((root/'points_detected.json').read_text())['eyes']
points={eye:[det[eye]['A'],det[eye]['B']] for eye in ['left','right']}
original_feature_function=geometry.features

def without_legacy_blue_mask(image,sift):
    mask=np.zeros(image.shape[:2],np.uint8)
    mask[65:-55,175:-150]=255
    return sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)

estimates={}
for name,feature_function in [('original',original_feature_function),('blue_filter_removed',without_legacy_blue_mask)]:
    geometry.features=feature_function
    R,t,n,k=geometry.relative_pose(root/'input_images',[0],cams)
    status,residual,baseline=geometry.reconstruct(points,cams,R,t,known['known_length_mm'])
    estimates[name]={'R':R.tolist(),'unit_t':t.tolist(),'matches':n,'inliers':k,
        'status':status,'epipolar_residual_px':residual,'baseline_estimated_mm':baseline}
record={'source_images':'../SingleInstrument_2026-11-07/preview_final/images/0000_{left,right}.png',
    'old_real_images_used':False,'stereo_extrinsic_truth_supplied_to_estimator':False,
    'extraction':'Manually specified image ROIs + color edge midpoint + local ellipse fit; not fully automatic.',
    'known_CAD_length_mm':known['known_length_mm'],'points':points,
    'algorithm_file':str(old/'geometry.py'),
    'algorithm_sha256':hashlib.sha256((old/'geometry.py').read_bytes()).hexdigest(),
    'adaptation':'Remove the original real-blue-marker exclusion only. Same SIFT/matching parameters and unchanged geometric solver.',
    'estimates':estimates}
(root/'estimate_reproducible.json').write_text(json.dumps(record,indent=2))

# Evaluation begins only after the estimate has been written.
truth_file=root.parent/'SingleInstrument_2026-11-07/preview_final/ground_truth/stereo_extrinsics.json'
truth=json.loads(truth_file.read_text())
true_baseline=1000*truth['baseline_m']
for row in estimates.values():
    baseline=row['baseline_estimated_mm']
    row['baseline_true_mm']=true_baseline
    row['abs_error_mm']=None if baseline is None else abs(baseline-true_baseline)
    row['relative_error_percent']=None if baseline is None else 100*abs(baseline/true_baseline-1)
    R=np.array(row['R']);G=np.array(truth['R_right_from_left'])
    row['rotation_error_deg']=float(np.rad2deg(np.arccos(np.clip((np.trace(R@G.T)-1)/2,-1,1))))

T=np.array(truth['t_right_from_left_m'])
status,residual,baseline=geometry.reconstruct(points,cams,np.array(truth['R_right_from_left']),T/np.linalg.norm(T),known['known_length_mm'])
record['diagnostic_known_relative_pose']={'warning':'True R and translation direction are supplied ONLY in this diagnostic. Not an independent extrinsic estimate.',
    'status':status,'baseline_estimated_mm':baseline,'epipolar_residual_px':residual,
    'relative_error_percent':None if baseline is None else 100*abs(baseline/true_baseline-1)}
record['evaluation_truth_file']=str(truth_file)
record['conclusion']='This first image-based jaw-to-ring trial fails accuracy validation. Do not treat the estimated baseline as a calibrated value.'
(root/'result.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
