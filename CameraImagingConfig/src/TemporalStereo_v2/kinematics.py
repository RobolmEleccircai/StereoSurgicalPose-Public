"""Official USD joint frames, generic PSM/ECM FK, and physical mesh landmarks."""
import numpy as np
from pxr import UsdGeom,UsdPhysics,Gf

AXES=np.diag([1.,-1.,-1.,1.])

def usd_world(stage,path):
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)).T

def rigid_mm(W):
    T=W.copy();scale=np.linalg.norm(W[:3,:3],axis=0)
    if not np.allclose(scale,scale[0],atol=1e-5):raise ValueError('Nonuniform rigid-body scale')
    T[:3,:3]/=scale;T[:3,3]*=10
    return T

def set_local(stage,path,matrix):
    x=UsdGeom.Xformable(stage.GetPrimAtPath(path));x.ClearXformOpOrder()
    a=x.GetPrim().GetAttribute('xformOp:transform')
    op=UsdGeom.XformOp(a) if a.IsValid() else x.AddTransformOp()
    op.Set(Gf.Matrix4d(np.asarray(matrix).T.tolist()));x.SetXformOpOrder([op])

def axis_rotation(axis,a):
    v=np.eye(3)['XYZ'.index(axis)]
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    T=np.eye(4);T[:3,:3]=np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
    return T

def joint_frame(p,side):
    q=p.GetAttribute('physics:localRot'+str(side)).Get().GetNormalized()
    T=np.eye(4);T[:3,:3]=np.array(Gf.Matrix3d(q)).T
    T[:3,3]=p.GetAttribute('physics:localPos'+str(side)).Get();return T

class Kinematics:
    def __init__(self,stage,root):
        self.stage=stage;self.root=root;self.W=usd_world(stage,root);inv=np.linalg.inv(self.W)
        self.bind={};self.joints=[]
        for p in stage.Traverse():
            if not str(p.GetPath()).startswith(root+'/') or not p.IsA(UsdPhysics.Joint):continue
            a=p.GetRelationship('physics:body0').GetTargets();b=p.GetRelationship('physics:body1').GetTargets()
            if not a or not b:continue
            a,b=str(a[0]),str(b[0])
            for path in (a,b):self.bind[path]=inv@usd_world(stage,path)
            J0,J1=joint_frame(p,0),joint_frame(p,1)
            rel=np.linalg.inv(self.bind[a]@J0)@self.bind[b]@J1
            axis=p.GetAttribute('physics:axis').Get() or 'X';k='XYZ'.index(axis);i,j=(k+1)%3,(k+2)%3
            angular=p.IsA(UsdPhysics.RevoluteJoint);linear=p.IsA(UsdPhysics.PrismaticJoint)
            q0=float(np.arctan2(rel[j,i],rel[i,i])) if angular else float(rel[k,3]) if linear else 0.
            lo=p.GetAttribute('physics:lowerLimit').Get();hi=p.GetAttribute('physics:upperLimit').Get()
            if angular:lo,hi=np.deg2rad([lo,hi])
            self.joints.append(dict(name=p.GetName(),path=str(p.GetPath()),parent=a,child=b,J0=J0,
                invJ0=np.linalg.inv(J0),relative=np.linalg.inv(self.bind[a])@self.bind[b],axis=axis,
                angular=angular,linear=linear,q0=q0,lower=lo,upper=hi))
        self.by_name={j['name']:j for j in self.joints}
        self.names=[j['name'] for j in self.joints if j['angular'] or j['linear']]
        self.base=np.array([self.by_name[n]['q0'] for n in self.names])
        self.lower=np.array([self.by_name[n]['lower'] for n in self.names]);self.upper=np.array([self.by_name[n]['upper'] for n in self.names])
        roots=set(self.bind)-{j['child'] for j in self.joints};self.roots=roots
        pending=list(self.joints);ordered=[];known=set(roots)
        while pending:
            ready=[j for j in pending if j['parent'] in known]
            if not ready:raise ValueError('USD joint graph has an unresolved cycle')
            for j in ready:ordered.append(j);known.add(j['child']);pending.remove(j)
        self.joints=ordered
        self.ancestors={b:[] for b in roots}
        for j in ordered:self.ancestors[j['child']]=self.ancestors[j['parent']]+([j['name']] if j['name'] in self.names else [])

    def positions(self,q):return dict(zip(self.names,map(float,q)))

    def fk(self,q,raw=False):
        q=np.asarray(q,float)
        if q.shape!=self.base.shape or not np.isfinite(q).all():raise ValueError('Invalid joint vector')
        if np.any(q<self.lower-1e-7) or np.any(q>self.upper+1e-7):raise ValueError('Joint limit exceeded')
        pos=self.positions(q);out={b:self.bind[b] for b in self.roots}
        for j in self.joints:
            d=pos.get(j['name'],j['q0'])-j['q0'];M=np.eye(4)
            if j['angular']:M=axis_rotation(j['axis'],d)
            elif j['linear']:M['XYZ'.index(j['axis']),3]=d
            out[j['child']]=out[j['parent']]@j['J0']@M@j['invJ0']@j['relative']
        return {b:(self.W@T if raw else rigid_mm(self.W@T)) for b,T in out.items()}

    def apply(self,q,velocity=None):
        for path,W in self.fk(q,raw=True).items():
            parent=str(self.stage.GetPrimAtPath(path).GetParent().GetPath())
            set_local(self.stage,path,np.linalg.inv(usd_world(self.stage,parent))@W)
        for i,name in enumerate(self.names):
            j=self.by_name[name];p=self.stage.GetPrimAtPath(j['path']);suffix='angular' if j['angular'] else 'linear'
            for field,value in [('position',q[i]),('velocity',0. if velocity is None else velocity[i])]:
                attr=p.GetAttribute(f'state:{suffix}:physics:{field}')
                if attr.IsValid():attr.Set(float(np.rad2deg(value) if j['angular'] else value))

    def speed_limits(self,c):
        return np.array([c['jaw_speed_rad_s'] if 'gripper' in n else c['insertion_speed_m_s'] if self.by_name[n]['linear'] else c['joint_speed_rad_s'] for n in self.names])

    def audit(self):
        return {j['name']:{k:j[k] for k in ('path','parent','child','angular','linear','q0','lower','upper')} for j in self.joints}

def mesh_triangles(prim):
    mesh=UsdGeom.Mesh(prim);v=np.array(mesh.GetPointsAttr().Get(),float)
    inds=np.array(mesh.GetFaceVertexIndicesAttr().Get(),np.int32);counts=np.array(mesh.GetFaceVertexCountsAttr().Get())
    if np.all(counts==3):return v,inds.reshape(-1,3)
    faces=[];off=0
    for n in counts:
        ring=inds[off:off+n];off+=n
        faces.extend((ring[0],ring[i],ring[i+1]) for i in range(1,n-1))
    return v,np.array(faces,np.int32)

def physical_landmarks(stage,psm):
    import trimesh
    result=[];bind=psm.fk(psm.base)
    for link,mesh in [('psm_tool_gripper1_link','gripper_right'),('psm_tool_gripper2_link','gripper_left')]:
        body=psm.root+'/'+link;path=body+'/visuals_xform/'+mesh
        vertices,faces=mesh_triangles(stage.GetPrimAtPath(path))
        v=np.unique(vertices,axis=0)
        tip=v[abs(v[:,1]-v[:,1].max())<1e-9].mean(0)
        # The extremal-ring centroid can lie just inside a curved cap. Anchor to
        # the actual CAD surface once, then keep that fixed material point.
        tip=trimesh.Trimesh(vertices,faces,process=False).nearest.on_surface([tip])[0][0]
        w=(usd_world(stage,path)@np.r_[tip,1])[:3]*10
        local=(np.linalg.inv(bind[body])@np.r_[w,1])[:3]
        result.append((body,local))
    return result
