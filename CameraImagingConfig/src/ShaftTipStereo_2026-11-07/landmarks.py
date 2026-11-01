"""CAD-defined shaft reference and mesh-tip labels; no stereo pose estimator."""
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf

ROBOT='/SurgicalTools/PSM_Right/Robot'
ROLL=ROBOT+'/psm_tool_roll_link'
JAWS=[ROBOT+'/psm_tool_gripper1_link/visuals_xform/gripper_right',
      ROBOT+'/psm_tool_gripper2_link/visuals_xform/gripper_left']

def world(stage,path):
    return np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

def put_frame(stage,path,xyz):
    xf=UsdGeom.Xform.Define(stage,path)
    xf.ClearXformOpOrder()
    attr=xf.GetPrim().GetAttribute('xformOp:translate')
    op=UsdGeom.XformOp(attr) if attr.IsValid() else xf.AddTranslateOp()
    op.Set(Gf.Vec3d(*map(float,xyz)));xf.SetXformOpOrder([op])
    return xf

def define(stage):
    # Reference-frame origin is the model's shaft/wrist joint center.
    frame=put_frame(stage,ROLL+'/ShaftAxisReference',[0,0,0])
    frame.GetPrim().CreateAttribute('label:axisDirectionLocal',Sdf.ValueTypeNames.Double3,custom=True).Set(Gf.Vec3d(0,0,1))
    frame.GetPrim().SetDocumentation('Shaft-axis reference: psm_tool_roll_link origin; +Z points toward the distal wrist. Non-rendering ground-truth frame.')
    ends=[]
    for mesh_path in JAWS:
        mesh=UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
        q=np.unique(np.asarray(mesh.GetPointsAttr().Get(),float),axis=0)
        # The inspected official meshes extend along local +Y to the jaw tip.
        selected=q[np.abs(q[:,1]-q[:,1].max())<1e-9]
        point=selected.mean(0)
        path=mesh_path+'/DistalTipReference'
        put_frame(stage,path,point)
        ends.append({'mesh_path':mesh_path,'reference_path':path,'mesh_local_point_m':point.tolist(),'distal_vertex_count':len(selected)})
    # Verify the shaft frame against the circular section of the existing mesh.
    path=ROLL+'/visuals_xform/visuals'
    q=np.unique(np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get(),float),axis=0)
    T=np.linalg.inv(world(stage,ROLL))@world(stage,path)
    q=(T@np.c_[q,np.ones(len(q))].T).T[:,:3]
    ring=q[abs(q[:,2]+.0036)<.0000006]
    xy=ring[:,:2]
    fit=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],(xy**2).sum(1),rcond=None)[0]
    radius=np.sqrt(fit[2]+sum(fit[:2]**2))
    residual=abs(np.linalg.norm(xy-fit[:2],axis=1)-radius).max()*1000
    assert np.linalg.norm(fit[:2])*1000<.002 and residual<.002
    definition={
        'shaft_reference_prim':ROLL+'/ShaftAxisReference',
        'shaft_reference_local_m':[0,0,0],
        'shaft_reference_definition':'Official psm_tool_roll_link origin, at the shaft/wrist joint center; a CAD-defined axis anchor, not an image feature.',
        'shaft_direction_local_unit':[0,0,1],
        'shaft_direction_sign':'Proximal shaft toward distal wrist',
        'shaft_axis_mesh_check':{'section_z_mm':-3.6,'fitted_section_center_xy_mm':(fit[:2]*1000).tolist(),'radius_mm':float(radius*1000),'circle_max_residual_mm':float(residual)},
        'tip_definition':'Midpoint of the two closed-jaw distal mesh end centers. Each end center is the mean of unique mesh vertices at maximal local Y. This is distinct from the nominal tool control-point frame.',
        'tip_endpoints':ends,
        'jaw_state':'Existing closed-jaw state retained in every sample',
        'units':'Positions in mm; directions dimensionless and normalized',
        'optical_axes':{'x':'right','y':'down','z':'forward'},
        'projection':'OpenCV 5-coefficient Brown-Conrady distortion, K and D for each eye; raw distorted image pixels',
        'shaft_draw_segment_mm':[-9,0],
        'shaft_long_mesh_visibility':'Inherited from current official asset/scene; this task does not change mesh visibility.',
        'simulation_scope':'Three static rendering poses; no physical joint/RCM trajectory validation; no camera-relation estimation.',
    }
    return definition

def physical_tip(stage,definition):
    return np.mean([world(stage,p['reference_path'])[:3,3] for p in definition['tip_endpoints']],axis=0)

def project(xyz,cam):
    xyz=np.asarray(xyz,float);x=xyz[...,0]/xyz[...,2];y=xyz[...,1]/xyz[...,2]
    k1,k2,p1,p2,k3=cam['D'];r=x*x+y*y;radial=1+k1*r+k2*r*r+k3*r*r*r
    K=np.asarray(cam['K'])
    return np.stack([K[0,0]*(x*radial+2*p1*x*y+p2*(r+2*x*x))+K[0,2],K[1,1]*(y*radial+p1*(r+2*y*y)+2*p2*x*y)+K[1,2]],axis=-1)

def labels(stage,definition,Tworld_camera,profile):
    # All USD positions are still in stage centimeters here.
    Taxis=world(stage,definition['shaft_reference_prim'])
    anchor=Taxis[:3,3];direction=Taxis[:3,2]/np.linalg.norm(Taxis[:3,2])
    tip=physical_tip(stage,definition)
    robot_T=world(stage,ROBOT)
    put_frame(stage,ROBOT+'/ClosedJawTipReference',(np.linalg.inv(robot_T)@np.r_[tip,1])[:3])
    info={}
    for eye,T in Tworld_camera.items():
        Tc=np.linalg.inv(T)
        c=(Tc@np.r_[anchor,1])[:3]*10
        p=(Tc@np.r_[tip,1])[:3]*10
        d=Tc[:3,:3]@direction;d/=np.linalg.norm(d)
        s=np.linspace(-9,0,91)
        line=c+s[:,None]*d
        uv_c=project(c,profile[eye]);uv_p=project(p,profile[eye]);uv_line=project(line,profile[eye])
        assert c[2]>0 and p[2]>0
        assert (0<uv_c[0]<1920 and 0<uv_c[1]<1080 and 0<uv_p[0]<1920 and 0<uv_p[1]<1080)
        info[eye]={'shaft_reference_position_mm':c.tolist(),'shaft_direction_unit':d.tolist(),'tip_position_mm':p.tolist(),
                   'shaft_reference_pixel_raw':uv_c.tolist(),'tip_pixel_raw':uv_p.tolist(),
                   'shaft_polyline_s_mm':s.tolist(),'shaft_polyline_pixels_raw':uv_line.tolist(),
                   'tip_distance_to_shaft_axis_mm':float(np.linalg.norm(np.cross(p-c,d))),
                   'points_inside_image':True,'occlusion_visibility':'Not inferred from in-frame status; centerline is inside the solid shaft.'}
    return info

VARIANTS=[{'id':'0000','tool_translation_left_camera_mm':[0,0,0],'rotation_about_left_optical_z_deg':0},
          {'id':'0001','tool_translation_left_camera_mm':[-12,-4,0],'rotation_about_left_optical_z_deg':15},
          {'id':'0002','tool_translation_left_camera_mm':[-6,5,3],'rotation_about_left_optical_z_deg':-12}]

def variant_transform(base_root,base_tip,Rworld_left,variant):
    a=np.deg2rad(variant['rotation_about_left_optical_z_deg']);c,s=np.cos(a),np.sin(a)
    R=Rworld_left@np.array([[c,-s,0],[s,c,0],[0,0,1]])@Rworld_left.T
    G=np.eye(4);G[:3,:3]=R
    G[:3,3]=base_tip-R@base_tip+Rworld_left@np.array(variant['tool_translation_left_camera_mm'])/10
    return G@base_root
