"""Separate physical-point visibility, sampled distal-region visibility and RGB quality."""
from collections import Counter
import numpy as np
from io_utils import canonical_hash
from imaging import project

DEFAULT_REGION = {'distal_length_mm': 1.0, 'surface_samples': 128,
                  'ray_endpoint_tolerance_mm': 0.05, 'mask_support_tolerance_px': 2.0}
DEFAULT_EXPOSURE = {'channel_threshold': 250, 'frame_fraction_threshold': 0.01}


def distal_surface_triangles(mesh, length_mm):
    """Clip actual jaw triangles to the distal 1 mm along native jaw-local +Y."""
    cutoff=float(mesh.vertices[:,1].max()-length_mm);result=[]
    for triangle in mesh.triangles[np.max(mesh.triangles[:,:,1],axis=1)>=cutoff]:
        polygon=[]
        for p,q in zip(triangle,np.roll(triangle,-1,axis=0)):
            pin=p[1]>=cutoff;qin=q[1]>=cutoff
            if pin:polygon.append(p)
            if pin!=qin:polygon.append(p+(q-p)*((cutoff-p[1])/(q[1]-p[1])))
        for i in range(1,len(polygon)-1):result.append([polygon[0],polygon[i],polygon[i+1]])
    triangles=np.asarray(result,float)
    if not len(triangles):raise ValueError('Empty distal jaw region')
    areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    keep=areas>1e-12
    return triangles[keep],areas[keep],cutoff


def surface_samples(triangles,areas,count):
    """Fixed deterministic equal-area strata; independent of camera and frame."""
    selected=np.searchsorted(np.cumsum(areas),(np.arange(count)+.5)/count*areas.sum())
    ids=np.arange(count)+.5
    u=np.sqrt((ids*0.6180339887498949)%1);v=(ids*0.4142135623730951)%1
    bary=np.c_[1-u,u*(1-v),u*v]
    return np.sum(triangles[selected]*bary[:,:,None],axis=1)


def classify_rays(safety,origin,points,tolerance):
    """Batch geometric rays, excluding only hits within tolerance of each target."""
    origin=np.asarray(origin);points=np.asarray(points);delta=points-origin
    lengths=np.linalg.norm(delta,axis=1)
    if np.any(lengths<=tolerance):raise ValueError('Region sample too close to optical center')
    directions=delta/lengths[:,None];nearest=np.full(len(points),np.inf);blockers=[None]*len(points)
    for obj in safety.static+safety.moving:
        if obj.group=='ecm':continue  # Same intentionally nonrendered housing as fixed-point test.
        R=obj.T[:3,:3];o=(origin-obj.T[:3,3])@R;d=directions@R
        hits,indices,_=obj.mesh.ray.intersects_location(np.repeat(o[None],len(points),axis=0),d,multiple_hits=True)
        if not len(hits):continue
        distances=np.einsum('ij,ij->i',hits-o,d[indices])
        for index,distance in zip(indices,distances):
            if 1e-5<distance<lengths[index]-tolerance and distance<nearest[index]:
                nearest[index]=distance;blockers[index]=obj.name
    return [{'visible':b is None,'occluder':b,
             'occluder_kind':None if b is None else 'tissue' if b.startswith(safety.scene.c['anatomy_root']+'/') else 'instrument'}
            for b in blockers]


def image_quality(rgb,mask,settings=None):
    cfg=dict(DEFAULT_EXPOSURE,**(settings or {}));rgb=np.asarray(rgb);mask=np.asarray(mask)>0
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected saved 8-bit RGB')
    white=np.all(rgb>=cfg['channel_threshold'],axis=2);fraction=float(white.mean())
    return {'overexposed':fraction>=cfg['frame_fraction_threshold'],
            'method':'near_white_RGB_fraction_heuristic_not_sensor_calibration',
            'near_white_fraction':fraction,'near_white_pixel_count':int(white.sum()),
            'tool_near_white_fraction':float(white[mask].mean()) if mask.any() else None,
            'channel_threshold':cfg['channel_threshold'],'frame_fraction_threshold':cfg['frame_fraction_threshold'],
            'action':'label_and_keep_no_filtering'}


class TipRegions:
    def __init__(self,safety,settings=None):
        self.safety=safety;self.cfg=dict(DEFAULT_REGION,**(settings or {}));self.regions=[]
        count=int(self.cfg['surface_samples']);length=float(self.cfg['distal_length_mm'])
        if count<16 or length<=0:raise ValueError('Invalid tip-region sampling configuration')
        for jaw in (1,2):
            shapes=[s for s in safety.moving if s.group=='psm' and s.body.endswith(f'/psm_tool_gripper{jaw}_link')]
            if len(shapes)!=1:raise ValueError('Expected one visible mesh per physical jaw')
            shape=shapes[0];triangles,areas,cutoff=distal_surface_triangles(shape.mesh,length)
            points=surface_samples(triangles,areas,count)
            definition={'jaw_index':jaw,'body':shape.body,'mesh':shape.name,
                        'definition':'native_jaw_local_Y >= maximum_mesh_Y - distal_length_mm',
                        'distal_length_mm':length,'local_Y_cutoff_mm':cutoff,'surface_area_mm2':float(areas.sum()),
                        'sample_count':count,'sampling':'deterministic_equal_area_stratified',
                        'sample_points_body_mm':points.tolist(),'source':'explicit_dataset_label_design_not_manufacturer_spec'}
            definition['sha256']=canonical_hash(definition)
            self.regions.append((shape,points,definition))

    def definitions(self):
        return {'schema_version':1,'parameters':self.cfg,'jaws':[d for _,_,d in self.regions]}

    def evaluate(self,camera,profile,eye,mask):
        import cv2
        camera=np.asarray(camera);inv=np.linalg.inv(camera);h,w=mask.shape
        distance=cv2.distanceTransform((np.asarray(mask)==0).astype(np.uint8),cv2.DIST_L2,5)
        result=[]
        for shape,local,definition in self.regions:
            world=local@shape.T[:3,:3].T+shape.T[:3,3]
            cam=(inv@np.c_[world,np.ones(len(world))].T).T[:,:3]
            uv=project(cam,profile,eye)
            inside=(cam[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            rays=classify_rays(self.safety,camera[:3,3],world,self.cfg['ray_endpoint_tolerance_mm'])
            statuses=[];support=[]
            for i,(valid,ray) in enumerate(zip(inside,rays)):
                state='out_of_frame' if not valid else 'visible' if ray['visible'] else 'tissue_occluded' if ray['occluder_kind']=='tissue' else 'self_occluded'
                statuses.append(state)
                if state=='visible':
                    x,y=np.rint(uv[i]).astype(int);x=np.clip(x,0,w-1);y=np.clip(y,0,h-1)
                    support.append(float(distance[y,x])<=self.cfg['mask_support_tolerance_px'])
            counts={key:statuses.count(key) for key in ('visible','self_occluded','tissue_occluded','out_of_frame')}
            result.append({'jaw_index':definition['jaw_index'],'region_definition_sha256':definition['sha256'],
                'region_visible':counts['visible']>0,'any_in_frame':bool(inside.any()),
                'surface_sample_count':len(local),'sample_counts':counts,
                'visible_surface_sample_fraction':counts['visible']/len(local),
                'fraction_scope':'whole_3D_patch_surface_samples_not_image_pixel_area',
                'mask_supported_visible_sample_count':sum(support),'visible_samples_mask_support_fraction':float(np.mean(support)) if support else None,
                'sample_projection_pixels':uv.tolist(),'sample_states':statuses,
                'distal_length_mm':self.cfg['distal_length_mm']})
        return result


def validate_quality_fields(record):
    for eye,e in record['eyes'].items():
        if e['fixed_tip_visibility']!=e['jaw_visibility']:raise ValueError('Fixed-tip compatibility fields disagree')
        for j,r in enumerate(e['jaw_tip_region_visibility'],1):
            n=r['surface_sample_count'];counts=r['sample_counts']
            if r['jaw_index']!=j or n!=sum(counts.values()) or len(r['sample_states'])!=n:raise ValueError('Invalid region sample counts')
            if Counter(r['sample_states'])!=Counter({k:v for k,v in counts.items() if v}):raise ValueError('Region states/counts disagree')
            if r['region_visible']!=(counts['visible']>0) or abs(r['visible_surface_sample_fraction']-counts['visible']/n)>1e-12:raise ValueError('Invalid region visibility fraction')
        q=e['image_quality']
        if q['overexposed']!=(q['near_white_fraction']>=q['frame_fraction_threshold']):raise ValueError('Overexposure tag disagrees')
    if record['quality_flags']['overexposed_any_eye']!=any(e['image_quality']['overexposed'] for e in record['eyes'].values()):raise ValueError('Pair overexposure tag disagrees')
