"""Mesh distance, containment, anatomical visibility and conservative swept checks."""
from dataclasses import dataclass
import numpy as np
import trimesh
import fcl
from pxr import UsdGeom
from kinematics import mesh_triangles,usd_world

def bvh(v,f):
    model=fcl.BVHModel();model.beginModel(len(v),len(f));model.addSubModel(np.ascontiguousarray(v),np.ascontiguousarray(f,dtype=np.int32));model.endModel();return model

def separation(a,b):
    return float(np.linalg.norm(np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0)))

@dataclass
class Shape:
    name:str
    body:str
    group:str
    mesh:object
    model:object
    obj:object
    probes:object
    T:object
    bounds:object
    levers:object

    def move(self,T):
        self.T=T;self.obj.setTransform(fcl.Transform(T[:3,:3],T[:3,3]))
        corners=trimesh.bounds.corners(self.mesh.bounds)
        p=corners@T[:3,:3].T+T[:3,3];self.bounds=np.array([p.min(0),p.max(0)])

def make_shape(name,body,group,v,f,levers=None):
    mesh=trimesh.Trimesh(v,f,process=True)
    if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume<0:mesh.invert()
    model=bvh(np.asarray(mesh.vertices),np.asarray(mesh.faces));obj=fcl.CollisionObject(model)
    comps=trimesh.graph.connected_components(mesh.edges_unique,nodes=np.arange(len(mesh.vertices)),min_len=1)
    probes=np.array([mesh.vertices[x[0]] for x in comps])
    return Shape(name,body,group,mesh,model,obj,probes,np.eye(4),mesh.bounds.copy(),levers)

class Safety:
    def __init__(self,scene):
        self.scene=scene;self.static=[];self.moving=[];self.ignored_pairs=set();stage=scene.stage;c=scene.c
        root=stage.GetPrimAtPath(c['anatomy_root'])
        if not root.IsValid():raise ValueError('Anatomy root missing: '+c['anatomy_root'])
        models={'psm':scene.psm,'ecm':scene.ecm}
        bind={g:k.fk(k.base) for g,k in models.items()}
        for p in stage.Traverse():
            if not p.IsA(UsdGeom.Mesh):continue
            name=str(p.GetPath());anatomy=name.startswith(c['anatomy_root']+'/')
            group=next((g for g,k in models.items() if name.startswith(k.root+'/')),None)
            if not anatomy and group is None:continue
            if anatomy and UsdGeom.Imageable(p).ComputeVisibility()=='invisible':continue
            if group=='psm' and UsdGeom.Imageable(p).ComputeVisibility()=='invisible':continue
            # Visual meshes define the anatomy and instrument surfaces; collision proxies are not mixed in.
            if group and ('collision' in name.lower() or '/visuals' not in name):continue
            v,f=mesh_triangles(p);W=usd_world(stage,name);worldv=(W@np.c_[v,np.ones(len(v))].T).T[:,:3]*10
            if anatomy:self.static.append(make_shape(name,'','anatomy',worldv,f));continue
            k=models[group];body=next((b for b in bind[group] if name.startswith(b+'/')),None)
            if body is None:raise ValueError('Mesh is not attached to an official rigid body: '+name)
            T=bind[group][body];local=(worldv-T[:3,3])@T[:3,:3]
            levers=self._levers(k,body,float(np.linalg.norm(local,axis=1).max()))
            shape=make_shape(name,body,group,local,f,levers);shape.move(T);self.moving.append(shape)
        if not self.static or not self.moving:raise ValueError('Empty anatomy or moving mesh set: refusing a vacuous safety pass')
        self.contact_surfaces=[x for x in self.static if x.mesh.is_watertight and x.mesh.is_winding_consistent and x.name.split('/')[-1] not in c['contact_excluded_names']]
        if not self.contact_surfaces:raise ValueError('No closed oriented contact surface')
        # Fixed/adjacent bodies, including the two jaw bodies sharing a physical hinge, are structural contacts.
        self.adjacent=set()
        for k in models.values():
            fixed={b:{b} for b in k.bind}
            for _ in k.bind:
                for j in k.joints:
                    if not j['angular'] and not j['linear']:
                        union=fixed[j['parent']]|fixed[j['child']]
                        for b in union:fixed[b]=union
            for j in k.joints:
                for a in fixed[j['parent']]:
                    for b in fixed[j['child']]:self.adjacent.add(frozenset((a,b)))
            for b,group in fixed.items():
                for other in group:self.adjacent.add(frozenset((b,other)))
        self.adjacent.add(frozenset((c['psm_root']+'/psm_tool_gripper1_link',c['psm_root']+'/psm_tool_gripper2_link')))
        # Telescoping shaft through its housing; wrist hinge cheeks around the two jaws.
        # These are explicit structural mating surfaces, not an automatic ignore-on-collision list.
        self.structural_pairs=[('psm_pitch_end_link','psm_tool_roll_link'),
                               ('psm_tool_pitch_link','psm_tool_gripper1_link'),
                               ('psm_tool_pitch_link','psm_tool_gripper2_link')]
        for a,b in self.structural_pairs:self.adjacent.add(frozenset((c['psm_root']+'/'+a,c['psm_root']+'/'+b)))
        self.audit={'anatomy_root':c['anatomy_root'],'anatomy_count':len(self.static),'contact_surface_count':len(self.contact_surfaces),
            'moving_meshes':[{'name':x.name,'body':x.body,'group':x.group} for x in self.moving],
            'anatomy':[{'name':x.name,'watertight':bool(x.mesh.is_watertight),'winding_consistent':bool(x.mesh.is_winding_consistent)} for x in self.static],
            'excluded_structural_body_pairs':[sorted(x) for x in self.adjacent],
            'explicit_mating_pairs':self.structural_pairs}
        self.check_count=0;self.pair_cache={};self.pair_dofs={};self.q_state={}

    @staticmethod
    def _levers(k,body,radius):
        ancestors=k.ancestors[body];result=np.zeros(len(k.names))
        chain=[];cur=body
        while cur not in k.roots:
            j=next(j for j in k.joints if j['child']==cur);chain.insert(0,j);cur=j['parent']
        for n in ancestors:
            at=next(i for i,j in enumerate(chain) if j['name']==n);j=chain[at];i=k.names.index(n)
            if j['linear']:result[i]=1000.;continue
            pivot=k.bind[j['parent']]@j['J0']
            length=np.linalg.norm(k.bind[j['child']][:3,3]-pivot[:3,3])*1000+radius
            for descendant in chain[at+1:]:
                length+=np.linalg.norm(descendant['relative'][:3,3])*1000
                if descendant['linear']:length+=max(abs(descendant['lower']-descendant['q0']),abs(descendant['upper']-descendant['q0']))*1000
            result[i]=length
        return result

    def move(self,qp,qe):
        self.q_state={'psm':np.asarray(qp),'ecm':np.asarray(qe)}
        transforms={'psm':self.scene.psm.fk(qp),'ecm':self.scene.ecm.fk(qe)}
        for shape in self.moving:shape.move(transforms[shape.group][shape.body])

    def cached_pair(self,a,b):
        """Exact joint-state cache for relative geometry, not rounded world poses."""
        pair=(a.name,b.name)
        if pair not in self.pair_dofs:
            involved={}
            for shape in (a,b):
                if shape.group=='anatomy':continue
                k=getattr(self.scene,shape.group)
                involved.setdefault(shape.group,set()).update(k.ancestors[shape.body])
            if a.group==b.group:
                k=getattr(self.scene,a.group)
                involved[a.group]-=set(k.ancestors[a.body])&set(k.ancestors[b.body])
            self.pair_dofs[pair]={g:[getattr(self.scene,g).names.index(n) for n in sorted(names)] for g,names in involved.items()}
        key=pair+tuple((g,tuple(self.q_state[g][indices])) for g,indices in self.pair_dofs[pair].items())
        if key not in self.pair_cache:
            if len(self.pair_cache)>20000:self.pair_cache.clear()
            # Nearest world coordinates are intentionally not cached: common rigid
            # motion preserves distance/containment but moves those coordinates.
            self.pair_cache[key]={k:v for k,v in self.pair(a,b).items() if k!='nearest_points_mm'}
        return self.pair_cache[key]

    @staticmethod
    def pair(a,b):
        collision=fcl.CollisionResult();fcl.collide(a.obj,b.obj,fcl.CollisionRequest(),collision)
        if collision.is_collision:return {'safe':False,'reason':'surface_intersection','pair':[a.name,b.name],'distance_mm':0.}
        # If a connected component is wholly inside the closed organ, surface tests alone miss it.
        for inner,outer in ((a,b),(b,a)):
            if not outer.mesh.is_watertight:continue
            world=inner.probes@inner.T[:3,:3].T+inner.T[:3,3];local=(world-outer.T[:3,3])@outer.T[:3,:3]
            selected=np.all(local>=outer.mesh.bounds[0],axis=1)&np.all(local<=outer.mesh.bounds[1],axis=1)
            if selected.any() and outer.mesh.contains(local[selected]).any():
                return {'safe':False,'reason':'contained_inside','pair':[a.name,b.name],'distance_mm':0.}
        result=fcl.DistanceResult();d=fcl.distance(a.obj,b.obj,fcl.DistanceRequest(enable_nearest_points=True),result)
        if not np.isfinite(d) or d<0: return {'safe':False,'reason':'invalid_distance','pair':[a.name,b.name],'distance_mm':0.}
        return {'safe':True,'distance_mm':float(d),'pair':[a.name,b.name],
                'nearest_points_mm':[np.asarray(x).tolist() for x in result.nearest_points]}

    def check(self,qp,qe,clearance=0.,self_check=True):
        self.check_count+=1;self.move(qp,qe);pairs=[]
        for a in self.moving:
            for b in self.static:pairs.append((separation(a.bounds,b.bounds),a,b))
        if self_check:
            for i,a in enumerate(self.moving):
                for b in self.moving[i+1:]:
                    if a.body==b.body or frozenset((a.body,b.body)) in self.adjacent:continue
                    pairs.append((separation(a.bounds,b.bounds),a,b))
        best={'safe':True,'distance_mm':float('inf'),'pair':None};organ_best=None
        for lower,a,b in sorted(pairs,key=lambda x:x[0]):
            threshold=max(best['distance_mm'],organ_best['distance_mm'] if organ_best else float('inf'))
            if lower>threshold:break
            result=self.cached_pair(a,b)
            if not result['safe']:return result
            if result['distance_mm']<best['distance_mm']:best=result
            if b.group=='anatomy' and (organ_best is None or result['distance_mm']<organ_best['distance_mm']):organ_best=result
        if not np.isfinite(best['distance_mm']):raise ValueError('No finite mesh distance')
        if organ_best is None:raise ValueError('No anatomical distance was evaluated')
        best=dict(best,organ_distance_mm=organ_best['distance_mm'],organ_pair=organ_best['pair'],required_tissue_clearance_mm=float(clearance))
        if organ_best['distance_mm']+1e-8<clearance:
            best=dict(best,safe=False,reason='insufficient_clearance',pair=organ_best['pair'])
        return best

    def organ_distance(self,qp,organ):
        self.move(qp,self.scene.ecm.base);best=None
        for a in self.moving:
            if a.group!='psm':continue
            if best and separation(a.bounds,organ.bounds)>best['distance_mm']:continue
            r=self.cached_pair(a,organ)
            if not r['safe']:return r
            if best is None or r['distance_mm']<best['distance_mm']:best=r
        return best

    def visible(self,origin,point):
        """Geometric visibility of a physical endpoint; C and T do not use this test."""
        delta=point-origin;length=float(np.linalg.norm(delta));direction=delta/length;nearest=length;blocker=None
        for obj in self.static+self.moving:
            # The existing scene deliberately excludes ECM housing from the camera render.
            if obj.group=='ecm':continue
            invR=obj.T[:3,:3].T;o=invR@(origin-obj.T[:3,3]);d=invR@direction
            hits,_,_=obj.mesh.ray.intersects_location([o],[d],multiple_hits=True)
            if len(hits):
                along=(hits-o)@d;valid=along[(along>1e-5)&(along<nearest-.05)]
                if len(valid):nearest=float(valid.min());blocker=obj.name
        return {'visible':blocker is None,'occluder':blocker,'occluder_kind':None if blocker is None else
                'tissue' if blocker.startswith(self.scene.c['anatomy_root']+'/') else 'instrument',
                'occluder_distance_mm':None if blocker is None else nearest}

    def sweep(self,path,a,b,check_state,depth=0,cache=None):
        """Clearance lower bound minus conservative articulated surface displacement."""
        cache={} if cache is None else cache
        def at(t):
            if t not in cache:cache[t]=check_state(t)
            return cache[t]
        ra,rb=at(a),at(b)
        if not ra['safe']:return ra
        if not rb['safe']:return rb
        speeds=path.speed_bounds(a,b)
        self.move(*path.at(a));velocity={x.name:float(x.levers@speeds[x.group]) for x in self.moving}
        pairs=[(x,y) for x in self.moving for y in self.static]
        pairs.extend((x,y) for i,x in enumerate(self.moving) for y in self.moving[i+1:]
                     if x.body!=y.body and frozenset((x.body,y.body)) not in self.adjacent)
        certified=True
        for x,y in pairs:
            speed=velocity[x.name]+velocity.get(y.name,0.)
            if x.group==y.group:
                # A common ancestor applies one rigid transform to both meshes:
                # it cannot change their separation. Bound only relative joints.
                k=getattr(self.scene,x.group);common=set(k.ancestors[x.body])&set(k.ancestors[y.body])
                relative=speeds[x.group].copy()
                for name in common:relative[k.names.index(name)]=0.
                speed=float((x.levers+y.levers)@relative)
            excursion=speed*(b-a)
            if excursion<1e-12:continue # a stationary housing gap cannot limit a moving jaw elsewhere
            clearance=max(ra.get('required_tissue_clearance_mm',0.),rb.get('required_tissue_clearance_mm',0.)) if y.group=='anatomy' else 0.
            if separation(x.bounds,y.bounds)>clearance+excursion+1e-7:continue
            key=('pair',a,x.name,y.name)
            if key not in cache:cache[key]=getattr(self,'cached_pair',self.pair)(x,y)
            r=cache[key]
            if not r['safe']:return r
            if r['distance_mm']<=clearance+excursion+1e-7:certified=False;break
        if certified:
            return {'safe':True,'certified_intervals':1}
        if depth>=self.scene.c['max_subdivision_depth']:
            return {'safe':False,'reason':'swept_interval_not_certified','interval_s':[a,b]}
        mid=(a+b)/2;rm=at(mid)
        if not rm['safe']:return dict(rm,time_s=mid)
        x=self.sweep(path,a,mid,check_state,depth+1,cache)
        if not x['safe']:return x
        y=self.sweep(path,mid,b,check_state,depth+1,cache)
        if not y['safe']:return y
        return {'safe':True,'certified_intervals':x['certified_intervals']+y['certified_intervals']}
