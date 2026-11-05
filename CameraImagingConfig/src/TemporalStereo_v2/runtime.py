"""Scene inputs, explicit effective metadata and independent optical-frame labels."""
from pathlib import Path
import argparse,copy,json
import numpy as np
from pxr import Usd,UsdGeom,UsdUtils,Sdf
from io_utils import read,write,sha,canonical_hash,source_hash,load_machine,assert_mode,validate_settings
from kinematics import Kinematics,AXES,usd_world,rigid_mm,set_local,physical_landmarks,axis_rotation
from imaging import project

def arguments():
    p=argparse.ArgumentParser();p.add_argument('--machine',type=Path,required=True)
    p.add_argument('--settings',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['plan','validate','generate','shard','merge'],required=True)
    a=p.parse_args();a.machine_data=load_machine(a.machine);a.settings_data=read(a.settings)
    validate_settings(a.settings_data)
    assert_mode(a.mode,a.machine_data)
    return a

def source_paths(m,c):
    return {key:(Path(m['asset_root'])/c[key]).resolve() for key in ('source_scene','calibration','appearance')}

def dependency_audit(m,c):
    paths=source_paths(m,c)
    for p in paths.values():
        if not p.is_file():raise FileNotFoundError(p)
    from assets import inspect_assets
    stage=Usd.Stage.Open(str(paths['source_scene']))
    files,records,overrides=inspect_assets(stage,m['isaac_root'],Path(m['output_root'])/'asset_cache')
    files.update(paths.values())
    inventory=[]
    for p in sorted(files):
        if not p.is_file():raise FileNotFoundError('Missing asset: '+str(p))
        inventory.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','trimesh','python-fcl','Pillow')}
    return {'files':inventory,'unresolved':[],'active_asset_attributes':records,'asset_overrides':overrides,'python_packages':versions,
            'scope':'Composed active shader graph, used USD layers, calibration and appearance. Editor icons and shadowed authoring history are excluded.',
            'isaac_version':(Path(m['isaac_root'])/'VERSION').read_text().strip()}

class Scene:
    def __init__(self,m,c,stage=None):
        self.m=m;self.c=c;self.paths=source_paths(m,c)
        self.stage=stage if stage is not None else Usd.Stage.Open(str(self.paths['source_scene']))
        if self.stage is None:raise ValueError('Cannot open source scene')
        # Every authored change goes to a new anonymous overlay, never a source layer.
        if stage is None:
            overlay=Sdf.Layer.CreateAnonymous('temporal_v2');overlay.subLayerPaths=[str(self.paths['source_scene'])]
            self.stage=Usd.Stage.Open(overlay)
        if abs(UsdGeom.GetStageMetersPerUnit(self.stage)-.01)>1e-12:raise ValueError('Expected centimeter USD stage')
        self.stage.SetEditTarget(self.stage.GetRootLayer())
        unused_left=self.stage.GetPrimAtPath('/SurgicalTools/PSM_Left')
        if unused_left.IsValid() and unused_left.IsActive():raise ValueError('Expected a single active PSM')
        self.psm=Kinematics(self.stage,c['psm_root']);self.ecm=Kinematics(self.stage,c['ecm_root'])
        self.refs=physical_landmarks(self.stage,self.psm);self.roll=c['psm_root']+'/psm_tool_roll_link'
        self.ecm_tip=c['ecm_root']+'/ecm_tip_link';self.rig=c['rig_path']
        left=rigid_mm(usd_world(self.stage,self.rig+'/Left'))@AXES
        right=rigid_mm(usd_world(self.stage,self.rig+'/Right'))@AXES
        self.mount=np.linalg.inv(self.ecm.fk(self.ecm.base)[self.ecm_tip])@left
        self.relative=np.linalg.inv(left)@right
        self.original_profile=read(self.paths['calibration']);self.appearance=read(self.paths['appearance'])
        p=self.original_profile;sx=c['width']/p['image']['width'];sy=c['height']/p['image']['height']
        A=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
        self.profile={eye:{'K':(A@np.array(p[eye]['K'])).tolist(),'D':p[eye]['D']} for eye in ('left','right')}
        self.profile.update(image={'width':c['width'],'height':c['height'],'space':'raw_distorted'},
            distortion_model='opencv_5_parameter',distortion_order=['k1','k2','p1','p2','k3'],
            source=p['source'],resize_pixel_center_transform=A.tolist(),
            effective={'fps':c['fps'],'frame_interval_s':1/c['fps'],'synchronized':True,
                       'exposure_time_s':None,'motion_blur':False,'record_type':'requested_capture_configuration'},
            mounting={'status':'existing_preview_convention_not_measured_hand_eye','T_ecm_tip_from_left_mm':self.mount.tolist()})

    def rig_relative(self,spec):
        rel=self.relative.copy();rel[:3,3]*=spec['baseline_scale']
        rel[:3,:3]=rel[:3,:3]@axis_rotation('Y',np.deg2rad(spec['relative_yaw_delta_deg']))[:3,:3]
        return rel

    def cameras(self,qe,spec):
        left=self.ecm.fk(qe)[self.ecm_tip]@self.mount
        return {'left':left,'right':left@self.rig_relative(spec)}

    def points(self,qp):
        transforms=self.psm.fk(qp);R=transforms[self.roll]
        tips=np.array([(transforms[b]@np.r_[v,1])[:3] for b,v in self.refs])
        return R[:3,3],R[:3,2],tips

    def labels(self,qp,qe,spec):
        C,d,tips=self.points(qp);points=np.vstack([C,tips.mean(0),tips]);result={}
        cameras=self.cameras(qe,spec)
        for eye,Tw in cameras.items():
            Tc=np.linalg.inv(Tw);p=(Tc@np.c_[points,np.ones(4)].T).T[:,:3]
            dc=Tc[:3,:3]@d;dc/=np.linalg.norm(dc);uv=project(p,self.profile,eye)
            result[eye]={'shaft_reference_position_mm':p[0].tolist(),'shaft_direction_unit':dc.tolist(),
                'tip_position_mm':p[1].tolist(),'jaw1_tip_position_mm':p[2].tolist(),'jaw2_tip_position_mm':p[3].tolist(),
                'shaft_reference_pixel':uv[0].tolist(),'tip_pixel':uv[1].tolist(),'jaw_tip_pixels':uv[2:].tolist(),
                'shaft_polyline_pixels':project(p[0]+np.linspace(-9,0,37)[:,None]*dc,self.profile,eye).tolist(),
                'in_frame':[bool(x[2]>0 and 0<=u[0]<self.c['width'] and 0<=u[1]<self.c['height']) for x,u in zip(p,uv)],
                'reference_surface_status':{'C':'internal_axis_reference','T':'virtual_midpoint'}}
        return result

    def apply(self,qp,qe,spec,vp=None,ve=None):
        self.psm.apply(qp,vp);self.ecm.apply(qe,ve)
        # Existing left mount is retained; only the right optical transform changes per sequence.
        L=usd_world(self.stage,self.rig+'/Left')@AXES
        rel=self.rig_relative(spec);relcm=rel.copy();relcm[:3,3]/=10
        right=L@relcm@AXES;parent=usd_world(self.stage,self.rig)
        set_local(self.stage,self.rig+'/Right',np.linalg.inv(parent)@right)
        light=L@AXES;light[:3,3]=((L@np.r_[relcm[:3,3]/2,1])[:3])
        set_local(self.stage,self.rig+'/EndoscopePointLight',np.linalg.inv(parent)@light)

def initialize_run(a):
    m,c=a.machine_data,a.settings_data;run=a.run;run.mkdir(parents=True,exist_ok=True)
    inv=dependency_audit(m,c)
    fingerprint=canonical_hash({'code':source_hash(),'settings':c,'dependencies':inv})
    path=run/'config/manifest.json'
    if path.exists() and read(path)['fingerprint']!=fingerprint:
        raise ValueError('Run fingerprint changed; choose a NEW run-id. Existing outputs are preserved.')
    write(path,{'fingerprint':fingerprint,'code_sha256':source_hash(),'dependencies':inv,'mode':a.mode,
                'machine_role':m['role'],'settings':c,'source_paths':{k:str(v) for k,v in source_paths(m,c).items()}})
    write(run/'config/effective_settings.json',c)
    return fingerprint
