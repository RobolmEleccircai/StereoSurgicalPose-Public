"""Diagnose matching separately from the unchanged relative-pose solver.

Ground-truth pose/depth is used ONLY for diagnostic point-transfer, labels and
synthetic solver controls. No result here is an image-only calibration fix.
"""
from pathlib import Path
import hashlib,json,sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

root=Path(__file__).resolve().parent
out=root/'relative_pose_audit';out.mkdir(exist_ok=True)
native=root.parent/'SingleInstrument_2026-11-07/preview/native'
old=Path('/home/leorobot/capstone-1/Grasp5-Stereo-SurgPose/pixel_extraction')
sys.path.insert(0,str(old))
import geometry
cv2.setNumThreads(3)
prior=json.loads((root/'result.json').read_text())
truth=json.loads((root/'separation_check/truth_from_scene.json').read_text())
cams=json.loads((root/'intrinsics.json').read_text())['cameras']
Rtrue=np.array(truth['relative_pose']['R_right_from_left'])
Ttrue=np.array(truth['relative_pose']['translation_right_from_left_m'])
ttrue=Ttrue/np.linalg.norm(Ttrue)
Btrue=truth['relative_pose']['baseline_mm']
known=json.loads((root/'known_AB.json').read_text())['known_length_mm']
source_hash=hashlib.sha256((old/'geometry.py').read_bytes()).hexdigest()
assert source_hash==prior['algorithm_sha256']

images={e:cv2.imread(str(root/'input_images'/e/'00.png')) for e in ['left','right']}
depth={e:np.load(native/f'depth/0000_{e}.npy') for e in ['left','right']}
def features(image,sift):
    mask=np.zeros(image.shape[:2],np.uint8);mask[65:-55,175:-150]=255
    return sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)

# Reproduce the exact input order, descriptor tests and spatial deduplication.
sift=cv2.SIFT_create(nfeatures=1800);matcher=cv2.BFMatcher()
kl,dl=features(images['left'],sift);kr,dr=features(images['right'],sift)
lr=geometry.ratio_matches(matcher,dl,dr);rl=geometry.ratio_matches(matcher,dr,dl)
rows=[];seen=set();cells={}
for i,j in lr.items():
    if rl.get(j)!=i:continue
    row=np.array(kl[i].pt+kr[j].pt)
    key=tuple((row//12).astype(int));cell=tuple((row[:2]//60).astype(int))
    if key in seen or cells.get(cell,0)>=12:continue
    seen.add(key);cells[cell]=cells.get(cell,0)+1;rows.append(row)
xy=np.array(rows);assert len(xy)==88

def unit_points(p,eye,precise=False):
    if not precise:return geometry.undistort(p,cams[eye])
    return cv2.undistortPointsIter(np.ascontiguousarray(p,dtype=float).reshape(-1,1,2),
        np.array(cams[eye]['K']),np.array(cams[eye]['D_k1_k2_p1_p2_k3']),None,None,
        (cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS,100,1e-14)).reshape(-1,2)

def project(q,eye):
    return cv2.projectPoints(np.asarray(q,dtype=float),np.zeros(3),np.zeros(3),
        np.array(cams[eye]['K']),np.array(cams[eye]['D_k1_k2_p1_p2_k3']))[0].reshape(-1,2)

def solve(points,name):
    if len(points)<8:return {'name':name,'n':len(points),'status':'too_few'},None
    l=unit_points(points[:,:2],'left');r=unit_points(points[:,2:],'right')
    cv2.setRNGSeed(412)
    E,mask=cv2.findEssentialMat(l,r,np.eye(3),method=cv2.USAC_MAGSAC,prob=.999,
        threshold=1.25/1635.,maxIters=12000)
    if E is None or E.shape!=(3,3):return {'name':name,'n':len(points),'status':'no_E'},None
    n,R,t,pose_mask=cv2.recoverPose(E,l,r,np.eye(3),mask=mask.copy());t=t.ravel()
    status,epi,baseline=geometry.reconstruct(truth['points'],cams,R,t,known)
    report={'name':name,'n':len(points),'status':'solved','E_inliers':int(mask.sum()),
        'pose_inliers':int(n),'R':R.tolist(),'unit_t':t.tolist(),
        'rotation_error_deg':float(np.degrees(np.arccos(np.clip((np.trace(R@Rtrue.T)-1)/2,-1,1)))),
        'translation_direction_error_deg':float(np.degrees(np.arccos(np.clip(t@ttrue,-1,1)))),
        'true_AB_scale_status':status,'baseline_with_true_AB_mm':baseline,
        'baseline_relative_error_percent':None if baseline is None else 100*abs(baseline/Btrue-1)}
    return report,{'E':E,'E_mask':mask.ravel()>0,'pose_mask':pose_mask.ravel()>0}

actual,model=solve(xy,'actual_image_matches')
frozen=prior['estimates']['blue_filter_removed']
assert np.allclose(actual['R'],frozen['R'],atol=1e-12)
assert np.allclose(actual['unit_t'],frozen['unit_t'],atol=1e-12)
assert actual['pose_inliers']==frozen['inliers']

def sample_depth(eye,p):
    p=np.asarray(p)
    return cv2.remap(depth[eye],p[:,0].astype(np.float32).reshape(-1,1),
        p[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan')).ravel().astype(float)

def transfer(p,source):
    target='right' if source=='left' else 'left'
    z=sample_depth(source,p)
    q=np.c_[unit_points(p,source,True),np.ones(len(p))]*z[:,None]
    dest=q@Rtrue.T+Ttrue if source=='left' else (q-Ttrue)@Rtrue
    uv=project(dest,target);zt=sample_depth(target,uv)
    inside=(uv[:,0]>=3)&(uv[:,0]<1917)&(uv[:,1]>=3)&(uv[:,1]<1077)
    valid=inside&np.isfinite(z)&np.isfinite(zt)&(z>0)&(zt>0)&(dest[:,2]>0)
    depth_difference=np.abs(zt-dest[:,2])
    visible=valid&(depth_difference<=np.maximum(.001,.01*dest[:,2]))
    # Flag local depth discontinuities, where interpolated depth isn't a precise surface sample.
    samples=np.stack([sample_depth(source,np.asarray(p)+[dx,dy]) for dx,dy in [(0,0),(-2,0),(2,0),(0,-2),(0,2)]])
    spread=np.nanmax(samples,axis=0)-np.nanmin(samples,axis=0)
    stable=np.isfinite(spread)&(spread<=np.maximum(.001,.02*z))
    return uv,q,visible&stable,depth_difference,spread

right_expected,cloud,left_valid,left_dz,left_spread=transfer(xy[:,:2],'left')
left_expected,_,right_valid,right_dz,right_spread=transfer(xy[:,2:],'right')
forward_error=np.linalg.norm(xy[:,2:]-right_expected,axis=1)
reverse_error=np.linalg.norm(xy[:,:2]-left_expected,axis=1)
sym_error=np.maximum(forward_error,reverse_error)
valid=left_valid&right_valid
good=valid&(sym_error<=2.)
bad=valid&(sym_error>2.)

def sampson(points,R,t):
    l=np.c_[unit_points(points[:,:2],'left'),np.ones(len(points))]
    r=np.c_[unit_points(points[:,2:],'right'),np.ones(len(points))]
    x,y,z=t;E=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])@R
    a=l@E.T;b=r@E
    return 1635*np.abs(np.sum(r*a,axis=1))/np.sqrt(np.sum(a[:,:2]**2+b[:,:2]**2,axis=1))

true_epi=sampson(xy,Rtrue,ttrue)
estimated_epi=sampson(xy,np.array(actual['R']),np.array(actual['unit_t']))
cases=[actual]
# Same left feature locations/depths, replace only right matches with precise geometric correspondences.
exact_same=np.c_[xy[:,:2],right_expected]
cases.append(solve(exact_same,'perfect_correspondences_same_left_locations')[0])
cases.append(solve(exact_same[left_valid],'perfect_correspondences_visible_stable_subset')[0])
for threshold in [1.,2.]:
    keep=valid&(sym_error<=threshold)
    cases.append(solve(xy[keep],f'oracle_verified_image_matches_within_{threshold:g}px')[0])

# A spread-out, visible control from the same scene, not an unrelated synthetic setup.
xx,yy=np.meshgrid(np.linspace(225,1675,12),np.linspace(120,940,9))
grid=np.c_[xx.ravel(),yy.ravel()]
grid_r,grid_cloud,grid_valid,_,_=transfer(grid,'left')
grid_pairs=np.c_[grid,grid_r][grid_valid]
cases.append(solve(grid_pairs,'perfect_correspondences_spread_across_scene')[0])

def summary(mask):
    idx=np.flatnonzero(mask)
    m=valid&mask
    return {'count':len(idx),'depth_checked':int(m.sum()),
        'within_1px':int((m&(sym_error<=1)).sum()),'within_2px':int((m&(sym_error<=2)).sum()),
        'over_2px':int((m&(sym_error>2)).sum()),'over_5px':int((m&(sym_error>5)).sum()),
        'over_10px':int((m&(sym_error>10)).sum()),
        'depth_transfer_error_median_px':None if not m.any() else float(np.median(sym_error[m])),
        'true_epipolar_median_px':float(np.median(true_epi[mask])),
        'true_epipolar_over_1p25px':int((mask&(true_epi>1.25)).sum()),
        'estimated_epipolar_median_px':float(np.median(estimated_epi[mask]))}

roundtrip={}
for e,p in [('left',xy[:,:2]),('right',xy[:,2:])]:
    q=np.c_[unit_points(p,e),np.ones(len(p))]
    error=np.linalg.norm(project(q,e)-p,axis=1)
    roundtrip[e]={'default_undistortion_roundtrip_max_px':float(error.max()),'median_px':float(np.median(error))}
instrument_roi=(xy[:,0]>1150)&(xy[:,0]<1720)&(xy[:,1]>580)&(xy[:,1]<850)
_,hom_mask=cv2.findHomography(unit_points(xy[:,:2],'left'),unit_points(xy[:,2:],'right'),
    cv2.RANSAC,1.25/1635,maxIters=10000,confidence=.999)
match_rows=[]
for i in range(len(xy)):
    match_rows.append({'index':i,'left':xy[i,:2].tolist(),'right_detected':xy[i,2:].tolist(),
        'right_from_left_depth':right_expected[i].tolist(),'left_from_right_depth':left_expected[i].tolist(),
        'depth_check_valid':bool(valid[i]),'forward_error_px':float(forward_error[i]),
        'reverse_error_px':float(reverse_error[i]),'true_epipolar_error_px':float(true_epi[i]),
        'estimated_epipolar_error_px':float(estimated_epi[i]),'accepted_by_pose':bool(model['pose_mask'][i]),
        'left_depth_m':float(cloud[i,2]),'left_depth_edge_spread_m':float(left_spread[i]),
        'left_to_right_depth_consistency_m':float(left_dz[i])})
np.savez_compressed(out/'diagnostic_inputs.npz',actual_matches=xy,perfect_same_layout=exact_same,
    perfect_grid=grid_pairs,left_valid=left_valid,depth_checked=valid,depth_error_px=sym_error,
    pose_mask=model['pose_mask'],Rtrue=Rtrue,ttrue=ttrue,cloud=cloud)
report={'scope':'Diagnostic only; uses simulator truth to audit matching and to supply ideal correspondences. No image-only accuracy claim.',
    'opencv_version':cv2.__version__,'original_solver_sha256':source_hash,
    'reproduced_frozen_estimate_exactly':True,'actual_cases':cases,'match_audit':{
        'all':summary(np.ones(len(xy),bool)),'accepted_by_estimated_pose':summary(model['pose_mask']),
        'matches_in_approximate_instrument_ROI':int(instrument_roi.sum()),
        'homography_inliers':int(hom_mask.sum()),'visibility_test':'both depth projections consistent within max(1 mm,1% Z), source local depth spread <= max(1 mm,2% Z)',
        'classification':'2 px bidirectional distance is a diagnostic tolerance; depth resampling and finite blur limit exact feature correspondence at boundaries.'},
    'undistortion_roundtrip':roundtrip,'matches':match_rows,
    'sources':{'opencv':'https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html',
        'depth':'https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.12.27/source/extensions/omni.replicator.core/docs/API.html#distance-to-image-plane'}}
(out/'report.json').write_text(json.dumps(report,indent=2))

# Same raw images and accepted correspondences, colored using independent depth audit.
canvas=Image.new('RGB',(1920,650),'#152033');draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
for k,e in enumerate(['left','right']):
    im=Image.open(root/'input_images'/e/'00.png').convert('RGB').resize((960,540))
    canvas.paste(im,(960*k,70))
for i in np.flatnonzero(model['pose_mask']):
    col='#36df80' if good[i] else '#ff665e' if bad[i] else '#ffd15a'
    a=(xy[i,0]/2,xy[i,1]/2+70);b=(xy[i,2]/2+960,xy[i,3]/2+70)
    draw.line([a,b],fill=col,width=1)
    for x,y in [a,b]:draw.ellipse((x-3,y-3,x+3,y+3),outline=col,width=2)
draw.text((12,10),'Accepted matches: green <=2 px; red >2 px; yellow depth/visibility uncertain',font=font,fill='white')
draw.text((12,38),'Color from independent simulation depth correspondence; no change to image solver.',font=font,fill='white')
canvas.save(out/'accepted_matches_audit.png')
print(json.dumps({k:v for k,v in report.items() if k not in ['matches']},indent=2),flush=True)
