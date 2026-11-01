"""Bounded stability/parameter controls; diagnostic, never selects by true error."""
from pathlib import Path
import json,sys
import cv2,numpy as np

root=Path(__file__).resolve().parent;out=root/'relative_pose_audit'
sys.path.insert(0,'/home/leorobot/capstone-1/Grasp5-Stereo-SurgPose/pixel_extraction')
from geometry import undistort,reconstruct
data=np.load(out/'diagnostic_inputs.npz')
cams=json.loads((root/'intrinsics.json').read_text())['cameras']
truth=json.loads((root/'separation_check/truth_from_scene.json').read_text())
length=json.loads((root/'known_AB.json').read_text())['known_length_mm']
Rtrue=data['Rtrue'];ttrue=data['ttrue'];B=truth['relative_pose']['baseline_mm']
cv2.setNumThreads(3)

def evaluate(points,method=cv2.USAC_MAGSAC,max_iters=12000,explicit=False):
    l=undistort(points[:,:2],cams['left']);r=undistort(points[:,2:],cams['right'])
    cv2.setRNGSeed(412)
    if explicit:
        params=cv2.UsacParams();params.confidence=.999;params.maxIterations=max_iters
        params.threshold=1.25/1635;params.score=cv2.SCORE_METHOD_MAGSAC
        params.loMethod=cv2.LOCAL_OPTIM_SIGMA;params.loSampleSize=50;params.loIterations=10
        E,mask=cv2.findEssentialMat(l,r,np.eye(3),np.eye(3),None,None,params)
    else:
        E,mask=cv2.findEssentialMat(l,r,np.eye(3),method=method,prob=.999,threshold=1.25/1635,maxIters=max_iters)
    if E is None or E.shape!=(3,3):return {'status':'no_unique_E'}
    n,R,t,pm=cv2.recoverPose(E,l,r,np.eye(3),mask=mask.copy());t=t.ravel()
    status,epi,b=reconstruct(truth['points'],cams,R,t,length)
    return {'status':status,'rotation_error_deg':float(np.degrees(np.arccos(np.clip((np.trace(R@Rtrue.T)-1)/2,-1,1)))),
        'direction_error_deg':float(np.degrees(np.arccos(np.clip(t@ttrue,-1,1)))),
        'E_inliers':int(mask.sum()),'pose_inliers':int(n),'baseline_mm':b,
        'baseline_error_percent':None if b is None else 100*abs(b/B-1),'E':E.tolist(),
        'R':R.tolist(),'unit_t':t.tolist()}

xy=data['actual_matches']
controls={}
for n in [1,1000,12000]:controls[f'legacy_USAC_maxIters_{n}']=evaluate(xy,max_iters=n)
controls['explicit_USAC_maxIterations_12000']=evaluate(xy,explicit=True)
controls['classic_RANSAC_same_input']=evaluate(xy,method=cv2.RANSAC)
controls['LMEDS_same_input']=evaluate(xy,method=cv2.LMEDS)
print('Solver controls complete',flush=True)

# Equal count, the same camera pair and scene, no wrong correspondences. This
# isolates sensitivity to small localization noise and feature distribution.
same=data['perfect_same_layout'];grid=data['perfect_grid']
grid=grid[np.linspace(0,len(grid)-1,len(same)).round().astype(int)]
trials=[]
for label,points in [('same_feature_locations',same),('spread_scene_points',grid)]:
    for sigma in [.25,.5]:
        for seed in range(10):
            rng=np.random.default_rng(62000+seed)
            noisy=points+rng.normal(0,sigma,points.shape)
            v=evaluate(noisy)
            for key in ['E','R','unit_t']:v.pop(key,None)
            trials.append(dict(layout=label,sigma_px=sigma,seed=seed,n=len(points),**v))
        print('Noise batch complete',label,sigma,flush=True)

summaries=[]
for label in ['same_feature_locations','spread_scene_points']:
    for sigma in [.25,.5]:
        rows=[r for r in trials if r['layout']==label and r['sigma_px']==sigma]
        accepted=[r for r in rows if r.get('baseline_mm') is not None]
        summaries.append({'layout':label,'sigma_px':sigma,'trials':len(rows),
            'scale_failed':len(rows)-len(accepted),
            'rotation_error_median_deg':float(np.median([r['rotation_error_deg'] for r in rows])),
            'translation_direction_error_median_deg':float(np.median([r['direction_error_deg'] for r in rows])),
            'baseline_error_median_percent_among_returned':None if not accepted else float(np.median([r['baseline_error_percent'] for r in accepted])),
            'baseline_error_max_percent_among_returned':None if not accepted else float(max(r['baseline_error_percent'] for r in accepted)),
            'within_5_percent_of_truth':sum(r.get('baseline_error_percent') is not None and r['baseline_error_percent']<=5 for r in rows)})

report={'scope':'Diagnostic comparison only. All listed methods/settings/trials are reported; no production solver chosen using ground truth.',
    'opencv_version':cv2.__version__,'controls':controls,'noise_trials':trials,'noise_summary':summaries,
    'known_version_issue':{
        'source_4_5_5':'https://github.com/opencv/opencv/blob/4.5.5/modules/calib3d/src/five-point.cpp#L408-L416',
        'upstream_issue':'https://github.com/opencv/opencv/issues/22696',
        'legacy_maxIters_ignored':True,
        'legacy_1_1000_12000_same_E':bool(np.array_equal(controls['legacy_USAC_maxIters_1']['E'],controls['legacy_USAC_maxIters_12000']['E'])),
        'explicit_12000_same_E':bool(np.array_equal(controls['legacy_USAC_maxIters_12000']['E'],controls['explicit_USAC_maxIterations_12000']['E'])),
        'interpretation':'This dependency defect is real, but bypassing it produced the same result on this pair; it does not explain away the observed error.'}}
(out/'stability.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'controls':{k:{x:y for x,y in v.items() if x not in ['E','R','unit_t']} for k,v in controls.items()},'noise_summary':summaries,'version_issue':report['known_version_issue']},indent=2),flush=True)
