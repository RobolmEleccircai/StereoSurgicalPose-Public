"""Deterministic articulated temporal labels and resolution-aware image formation."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import cv2
from pxr import Usd, UsdGeom, UsdPhysics, Gf

ROBOT = '/SurgicalTools/PSM_Right/Robot'
ROLL = ROBOT + '/psm_tool_roll_link'
ECM = '/Endoscope/ECM'
RIG = ECM + '/Robot/ecm_tip_link/StereoPreviewRig'
AXES = np.diag([1., -1., -1., 1.])
EYES = ('left', 'right')

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def world(stage, path):
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)).T

def set_local(stage, path, matrix):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path)); xf.ClearXformOpOrder()
    a = xf.GetPrim().GetAttribute('xformOp:transform')
    op = UsdGeom.XformOp(a) if a.IsValid() else xf.AddTransformOp()
    op.Set(Gf.Matrix4d(np.asarray(matrix).T.tolist())); xf.SetXformOpOrder([op])

def set_world(stage, path, matrix):
    parent = str(stage.GetPrimAtPath(path).GetParent().GetPath())
    set_local(stage, path, np.linalg.inv(world(stage, parent)) @ matrix)

def rotation(axis, angle):
    v = np.eye(3)['XYZ'.index(axis)]
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    T = np.eye(4); T[:3,:3] = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    return T

def joint_frame(joint, side):
    q = joint.GetAttribute('physics:localRot'+str(side)).Get().GetNormalized()
    T = np.eye(4); T[:3,:3] = np.array(Gf.Matrix3d(q)).T
    T[:3,3] = joint.GetAttribute('physics:localPos'+str(side)).Get()
    return T

class Kinematics:
    """Apply joint-coordinate motions to the official USD rigid-body graph, without dynamics."""
    def __init__(self, stage):
        self.stage = stage; self.robot_world = world(stage, ROBOT)
        inv = np.linalg.inv(self.robot_world)
        self.bind = {}; self.joints = []
        for p in stage.Traverse():
            if not str(p.GetPath()).startswith(ROBOT+'/') or not p.IsA(UsdPhysics.Joint): continue
            a=p.GetRelationship('physics:body0').GetTargets(); b=p.GetRelationship('physics:body1').GetTargets()
            if not a or not b: continue
            parent, child = str(a[0]), str(b[0])
            for path in (parent,child): self.bind[path] = inv@world(stage,path)
            J0,J1 = joint_frame(p,0),joint_frame(p,1)
            rel = np.linalg.inv(self.bind[parent]@J0)@self.bind[child]@J1
            axis=p.GetAttribute('physics:axis').Get() or 'X'; k='XYZ'.index(axis)
            angular=p.IsA(UsdPhysics.RevoluteJoint); linear=p.IsA(UsdPhysics.PrismaticJoint)
            i,j=((k+1)%3,(k+2)%3)
            q0=float(np.arctan2(rel[j,i],rel[i,i])) if angular else float(rel[k,3]) if linear else 0.
            lower=p.GetAttribute('physics:lowerLimit').Get(); upper=p.GetAttribute('physics:upperLimit').Get()
            if angular: lower,upper=np.deg2rad([lower,upper])
            self.joints.append(dict(name=p.GetName(),path=str(p.GetPath()),parent=parent,child=child,
                J0=J0,bind_relative=np.linalg.inv(self.bind[parent])@self.bind[child],axis=axis,
                angular=angular,linear=linear,q0=q0,lower=lower,upper=upper))
        self.by_name={j['name']:j for j in self.joints}
        self.base={k:j['q0'] for k,j in self.by_name.items() if j['angular'] or j['linear']}
        self.base['psm_tool_gripper1_joint']=0.; self.base['psm_tool_gripper2_joint']=0.

    def solve(self, positions, apply=True):
        result={ROBOT+'/psm_base_link':self.bind[ROBOT+'/psm_base_link']}
        pending=list(self.joints)
        while pending:
            progressed=False
            for j in list(pending):
                if j['parent'] not in result: continue
                q=float(positions.get(j['name'],j['q0']))
                if j['angular'] or j['linear']:
                    assert j['lower']-1e-6 <= q <= j['upper']+1e-6,(j['name'],q,j['lower'],j['upper'])
                delta=q-j['q0']; M=np.eye(4)
                if j['angular']: M=rotation(j['axis'],delta)
                elif j['linear']: M['XYZ'.index(j['axis']),3]=delta
                result[j['child']]=result[j['parent']]@j['J0']@M@np.linalg.inv(j['J0'])@j['bind_relative']
                pending.remove(j); progressed=True
            assert progressed, 'Joint graph is not a tree rooted at the PSM base'
        if apply:
            for path,T in result.items(): set_world(self.stage,path,self.robot_world@T)
            for j in self.joints:
                if not (j['angular'] or j['linear']): continue
                suffix='angular' if j['angular'] else 'linear'
                q=positions.get(j['name'],j['q0'])
                attr=self.stage.GetPrimAtPath(j['path']).GetAttribute(f'state:{suffix}:physics:position')
                if attr.IsValid(): attr.Set(float(np.rad2deg(q) if j['angular'] else q))
                v=self.stage.GetPrimAtPath(j['path']).GetAttribute(f'state:{suffix}:physics:velocity')
                if v.IsValid(): v.Set(0.)
        return {path:self.robot_world@T for path,T in result.items()}

    def audit(self):
        return {j['name']:{k:(float(j[k]) if isinstance(j[k],np.floating) else j[k])
            for k in ('path','axis','angular','linear','q0','lower','upper')} for j in self.joints}

def camera_profile(config):
    p=json.loads((Path(config['runtime_root'])/config['calibration']).read_text())
    sx=config['width']/p['image']['width']; sy=config['height']/p['image']['height']
    # Pixel centers: resize uses u_new=(u_old+0.5)*scale-0.5.
    A=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    for eye in EYES: p[eye]['K']=(A@np.array(p[eye]['K'])).tolist()
    p['image'].update(width=config['width'],height=config['height'])
    p['resize_pixel_center_transform']=A.tolist(); p['profile_id']='drends_exvivo_720_pixel_centers'
    return p

def maps(profile):
    h,w=profile['image']['height'],profile['image']['width']
    y,x=np.indices((h,w),dtype=np.float64); pixels=np.stack([x,y],-1).reshape(-1,1,2)
    rays={e:cv2.undistortPointsIter(pixels,np.array(profile[e]['K']),np.array(profile[e]['D']),None,None,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(h,w,2) for e in EYES}
    f=1600.*w/1920
    cx=float(np.ceil((max(abs(r[:,:,0]).max() for r in rays.values())*f+16)/32)*32)
    cy=float(np.ceil((max(abs(r[:,:,1]).max() for r in rays.values())*f+16)/32)*32)
    mapped={e:(r*f+[cx,cy]).astype(np.float32) for e,r in rays.items()}
    return mapped,dict(width=int(cx*2),height=int(cy*2),f=f,cx=cx,cy=cy,
                      focal_length_mm=4.,horizontal_aperture_mm=2*cx/f*4,vertical_aperture_mm=2*cy/f*4)

def project(points,profile,eye):
    p=np.asarray(points,float).reshape(-1,3)
    uv,_=cv2.projectPoints(p,np.zeros(3),np.zeros(3),np.array(profile[eye]['K']),np.array(profile[eye]['D']))
    return uv.reshape(-1,2)

def tip_references(stage):
    refs=[]
    for link,mesh in [('psm_tool_gripper1_link','gripper_right'),('psm_tool_gripper2_link','gripper_left')]:
        path=ROBOT+'/'+link+'/visuals_xform/'+mesh
        q=np.unique(np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get(),float),axis=0)
        p=q[abs(q[:,1]-q[:,1].max())<1e-9].mean(0)
        refs.append((path,p))
    return refs

def labels(stage,profile,refs):
    Taxis=world(stage,ROLL); C=Taxis[:3,3]; d=Taxis[:3,2]/np.linalg.norm(Taxis[:3,2])
    tips=np.array([(world(stage,path)@np.r_[p,1])[:3] for path,p in refs])
    midpoint=tips.mean(0); result={}
    for eye in EYES:
        Tw=world(stage,RIG+'/'+eye.capitalize())@AXES; Tc=np.linalg.inv(Tw)
        pts=(Tc@np.c_[np.vstack([C,midpoint,tips]),np.ones(4)].T).T[:,:3]*10
        dc=Tc[:3,:3]@d; dc/=np.linalg.norm(dc)
        uv=project(pts,profile,eye); line=pts[0]+np.linspace(-9,0,37)[:,None]*dc
        result[eye]=dict(shaft_reference_position_mm=pts[0].tolist(),shaft_direction_unit=dc.tolist(),
            tip_position_mm=pts[1].tolist(),jaw1_tip_position_mm=pts[2].tolist(),jaw2_tip_position_mm=pts[3].tolist(),
            shaft_reference_pixel=uv[0].tolist(),tip_pixel=uv[1].tolist(),jaw_tip_pixels=uv[2:].tolist(),
            shaft_polyline_pixels=project(line,profile,eye).tolist(),
            in_frame=[bool(p[2]>0 and 0<=u[0]<profile['image']['width'] and 0<=u[1]<profile['image']['height']) for p,u in zip(pts,uv)])
    return result

def smooth(a,b,frame,count):
    u=frame/(count-1); s=u*u*(3-2*u)
    return np.asarray(a)*(1-s)+np.asarray(b)*s

def frame_state(seq,index,config):
    n=config['frames_per_sequence']; q={k:float(smooth(a,seq['q_end'][k],index,n)) for k,a in seq['q_start'].items()}
    return dict(joints=q,ecm_motion=smooth(seq['ecm_start'],seq['ecm_end'],index,n).tolist(),
        light_multiplier=float(smooth(seq['light_start'],seq['light_end'],index,n)),
        noise_multiplier=seq['noise_multiplier'],blur_multiplier=seq['blur_multiplier'])

def apply_frame(stage,kin,seq,state,base_ecm,base_camera,base_rig):
    kin.solve(state['joints'])
    xyz=np.array(state['ecm_motion'][:3])/10; angles=np.deg2rad(state['ecm_motion'][3:])
    R=rotation('Z',angles[2])@rotation('Y',angles[1])@rotation('X',angles[0])
    T=base_camera@R@np.linalg.inv(base_camera); T[:3,3]+=base_camera[:3,:3]@xyz
    set_world(stage,ECM,T@base_ecm)
    rel=base_rig.copy(); rel[:3,3]*=seq['baseline_scale']
    jitter=rotation('Y',np.deg2rad(seq['relative_yaw_delta_deg']))
    rel[:3,:3]=rel[:3,:3]@jitter[:3,:3]
    set_local(stage,RIG+'/Right',rel@AXES)
    light=AXES.copy(); light[:3,3]=rel[:3,3]/2
    set_local(stage,RIG+'/EndoscopePointLight',light)

def image_effects(linear,eye,profile,initial,state,seed,config):
    h,w=linear.shape[:2]; sx=w/1920; sy=h/1080
    y,x=np.indices((h,w),dtype=np.float32)
    cx,cy=profile[eye]['K'][0][2],profile[eye]['K'][1][2]
    far=max(np.hypot(px-cx,py-cy) for px,py in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)])
    r=np.clip(np.hypot(x-cx,y-cy)/far,0,1)
    sigma=np.sqrt(.3**2+(.8*r*r)**2)*sx*state['blur_multiplier']
    levels=np.linspace(float(sigma.min()),float(sigma.max()),17); step=levels[1]-levels[0]
    blurred=np.zeros_like(linear,dtype=np.float32)
    for s in levels:
        weight=np.maximum(1-abs(sigma-s)/step,0)
        blurred+=weight[:,:,None]*cv2.GaussianBlur(linear,(0,0),float(s),borderType=cv2.BORDER_REFLECT_101)
    # Preserve the accepted ellipse in normalized image coordinates.
    r=np.sqrt(((x-(959.5+.5)*sx+.5)/(840*sx))**2+((y-(539.5+.5)*sy+.5)/(470*sy))**2)
    t=np.clip((r-.3)/.8,0,1); gain=1-(1-config['vignette_minimum_linear_gain'])*t*t*(3-2*t)
    rgb=blurred*np.asarray(initial['illumination']['combined_rgb_scales'][eye])*gain[:,:,None]*config['global_linear_multiplier']
    rgb=np.maximum(rgb,0); rgb=np.clip(np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055),0,1)
    rng=np.random.default_rng(seed); luminance=rgb@np.array([.2126,.7152,.0722]); factor=state['noise_multiplier']
    noise=rng.standard_normal(rgb.shape)*np.sqrt(1+6*luminance)[:,:,None]*factor
    row=rng.normal(0,.3*factor,(h,1,1))
    return np.uint8(np.clip(np.rint(rgb*255+noise+row),0,255))
