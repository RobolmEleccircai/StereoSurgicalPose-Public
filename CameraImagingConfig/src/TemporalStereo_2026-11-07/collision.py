"""Triangle-mesh collision checks for visible distal PSM geometry and nearby anatomy."""
import numpy as np
import fcl
from pxr import UsdGeom
from common import world, ROBOT

def triangles(mesh):
    vertices=np.array(mesh.GetPointsAttr().Get(),dtype=float)
    indices=np.array(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32)
    counts=np.array(mesh.GetFaceVertexCountsAttr().Get(),dtype=np.int32)
    if np.all(counts==3): return vertices,indices.reshape(-1,3)
    faces=[]; offset=0
    for count in counts:
        ring=indices[offset:offset+count]; offset+=count
        faces.extend((ring[0],ring[i],ring[i+1]) for i in range(1,count-1))
    return vertices,np.array(faces,dtype=np.int32)

def geometry(vertices,faces):
    model=fcl.BVHModel(); model.beginModel(len(vertices),len(faces))
    model.addSubModel(np.ascontiguousarray(vertices),np.ascontiguousarray(faces))
    model.endModel(); return model

class CollisionCheck:
    def __init__(self,stage):
        self.stage=stage; self.static=[]; self.moving=[]; self.models=[]
        center=world(stage,ROBOT+'/psm_tool_tip_link')[:3,3]*10
        for p in stage.Traverse():
            if not p.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(p).ComputeVisibility()=='invisible': continue
            path=str(p.GetPath()); tool=path.startswith(ROBOT+'/psm_tool_')
            if not tool and not path.startswith('/World/'): continue
            v,f=triangles(UsdGeom.Mesh(p)); W=world(stage,path)
            pts=(W@np.c_[v,np.ones(len(v))].T).T[:,:3]*10
            if not tool:
                # Retain all triangles whose bounds overlap the entire trajectory workspace.
                tri=pts[f]; selected=np.all(tri.max(1)>=center-180,axis=1)&np.all(tri.min(1)<=center+180,axis=1)
                f=f[selected]
                if not len(f): continue
                model=geometry(pts,f); obj=fcl.CollisionObject(model)
                self.models.append(model); self.static.append((path,obj))
            else:
                scale=np.linalg.norm(W[:3,:3],axis=0); R=W[:3,:3]/scale
                model=geometry(v*scale*10,f); obj=fcl.CollisionObject(model,fcl.Transform(R,W[:3,3]*10))
                self.models.append(model); self.moving.append((path,obj,scale))
        self.manager=fcl.DynamicAABBTreeCollisionManager()
        self.manager.registerObjects([x[1] for x in self.static]); self.manager.setup()
        self.audit={'static_anatomy_meshes':len(self.static),'moving_visible_distal_meshes':len(self.moving),
            'algorithm':'FCL triangle-mesh intersection; official angular limits and symmetric jaw coupling additionally enforced',
            'scope':'Distal tool versus nearby visible anatomy. Adjacent jaw/hinge contact is not treated as self-collision; no deformable-tissue or force simulation.'}

    def check(self):
        for path,obj,scale in self.moving:
            W=world(self.stage,path); obj.setTransform(fcl.Transform(W[:3,:3]/scale,W[:3,3]*10))
            data=fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=1,enable_contact=False))
            self.manager.collide(obj,data,fcl.defaultCollisionCallback)
            if data.result.is_collision: return False,path
        return True,None
