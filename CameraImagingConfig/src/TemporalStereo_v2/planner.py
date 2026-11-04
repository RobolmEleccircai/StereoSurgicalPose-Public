"""Plan complete, deterministic, bounded trajectories before any rendering."""
import time,collections
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import trimesh
from runtime import arguments,initialize_run,Scene
from safety import Safety
from paths import Trajectory,jaw_positions,smoother,phase,contact_offset
from io_utils import write,read,sha,canonical_hash,seal,require_seal

CASES=[('pose_motion','return'),('pose_motion','arc'),('lighting','hover'),('lighting','contact'),
       ('sensor_quality','return'),('sensor_quality','slide'),('occlusion','occlusion'),('occlusion','occlusion'),
       ('ecm_motion','arc'),('ecm_motion','slide'),('stereo_rig','return'),('stereo_rig','slide'),
       ('combined_pose_jaw','slide'),('combined_pose_jaw','occlusion')]

class Planner:
    def __init__(self,scene,safety):
        self.s=scene;self.safe=safety;self.c=scene.c;self.k=scene.psm
        self.active=[i for i,n in enumerate(self.k.names) if 'gripper' not in n]
        self.end=self.k.root+'/psm_tool_yaw_link'
        self.R0=self.k.fk(self.k.base)[self.end][:3,:3]
        self.duration=(self.c['frames_per_sequence']-1)/self.c['fps']

    def ik(self,point,R,alpha,seed):
        k=self.k;q=jaw_positions(seed,k,alpha);indices=self.active
        def residual(x):
            qq=q.copy();qq[indices]=x;fk=k.fk(qq)
            body,local=self.s.refs[0];tip=(fk[body]@np.r_[local,1])[:3]
            angle=Rotation.from_matrix(R.T@fk[self.end][:3,:3]).as_rotvec()
            return np.r_[(tip-point)/self.c['ik_position_tolerance_mm'],angle/np.deg2rad(self.c['ik_orientation_tolerance_deg'])]
        lo=k.lower[indices]+1e-8;hi=k.upper[indices]-1e-8
        x=np.minimum(np.maximum(q[indices],lo+1e-10),hi-1e-10) # bounded solver seed, not sampled labels
        result=least_squares(residual,x,bounds=(lo,hi),max_nfev=160,xtol=1e-9,ftol=1e-9,gtol=1e-9,x_scale='jac')
        q[indices]=result.x
        if np.max(abs(residual(result.x)))>1.:raise ValueError('IK target not reached')
        return q

    def camera_aim(self,point,screen_x=.57):
        k=self.s.ecm;profile=self.s.profile['left'];K=np.array(profile['K']);desired=np.array([self.c['width']*screen_x,self.c['height']*.52])
        xy=(desired-K[:2,2])/np.diag(K)[:2]
        def fun(q):
            cam=self.s.cameras(q,{'baseline_scale':1.,'relative_yaw_delta_deg':0.})['left']
            p=cam[:3,:3].T@(point-cam[:3,3]);return np.r_[(p[:2]-xy*p[2])/2.,(p[2]-100.)/2.,q[-1]*3]
        lo=k.lower+1e-8;hi=k.upper-1e-8;seed=np.minimum(np.maximum(k.base,lo+1e-10),hi-1e-10)
        r=least_squares(fun,seed,bounds=(lo,hi),max_nfev=180)
        if np.max(abs(fun(r.x)[:3]))>4:raise ValueError('ECM cannot frame contact target')
        return r.x

    def sample_free(self,rng):
        spans={'psm_yaw_joint':np.deg2rad(2),'psm_pitch_end_joint':np.deg2rad(2),
               'psm_main_insertion_joint':.012,'psm_tool_roll_joint':np.deg2rad(25),
               'psm_tool_pitch_joint':np.deg2rad(12),'psm_tool_yaw_joint':np.deg2rad(8)}
        q=self.k.base.copy()
        for i,n in enumerate(self.k.names):
            if n not in spans:continue
            lo=max(self.k.lower[i]+1e-5,self.k.base[i]-spans[n]);hi=min(self.k.upper[i]-1e-5,self.k.base[i]+spans[n])
            if self.k.by_name[n]['linear']:lo+=.003 # room for both directions, without clipping sampled trajectories
            q[i]=rng.uniform(lo,hi)
        return q

    def free_path(self,rng,spec,alpha):
        q=self.sample_free(rng);base=q.copy();knots=[];times=np.linspace(0,self.duration,5)
        delta=np.array([.0025 if self.k.by_name[n]['linear'] else np.deg2rad(3) for n in self.k.names])
        for u in np.linspace(0,1,5):
            scale=np.sin(2*np.pi*u) if spec['kind']=='arc' else np.sin(np.pi*u)**2
            qq=base+delta*scale
            if spec['kind']=='arc':qq[0]+=np.deg2rad(1)*(1-np.cos(2*np.pi*u))
            qq=jaw_positions(qq,self.k,np.deg2rad(alpha[0]+(alpha[1]-alpha[0])*smoother(u)))
            knots.append(qq)
        ebase=self.s.ecm.base.copy();ecms=np.repeat(ebase[None,:],5,axis=0)
        if spec['category']=='ecm_motion':
            ecms[:,0]+=np.deg2rad(.6)*np.sin(np.linspace(0,np.pi,5))
            ecms[:,1]+=np.deg2rad(.3)*np.sin(np.linspace(0,np.pi,5))
            ecms[:,2]+=np.linspace(.001,.003,5)
        return Trajectory.create(times,knots,ecms),None

    def surface_seed(self,rng):
        _,_,tips=self.s.points(self.k.base);query=tips[0]+rng.uniform(-18,18,3)
        candidates=[]
        for organ in self.safe.contact_surfaces:
            p,d,idx=organ.mesh.nearest.on_surface([query]);candidates.append((float(d[0])+rng.uniform(0,10),organ,p[0],int(idx[0])))
        _,organ,point,index=min(candidates,key=lambda v:v[0]);n=organ.mesh.face_normals[index]
        return organ,point,n

    def contact_pose(self,organ,point,normal,goal,R,alpha,seed,offset,require_jaw=True):
        """Adjust actual mesh clearance, not the virtual T midpoint."""
        for _ in range(30):
            q=self.ik(point+normal*offset,R,alpha,seed);r=self.safe.organ_distance(q,organ)
            if not r['safe']:offset+=.5;seed=q;continue
            error=goal-r['distance_mm']
            if abs(error)<.002:
                if require_jaw and 'gripper' not in r['pair'][0]:raise ValueError('Contact is on wrist/shaft instead of jaw')
                return q,offset
            offset+=np.clip(error,-2.,2.);seed=q
            if abs(offset)>25:break
        raise ValueError('Contact clearance did not converge')

    def contact_path(self,rng,spec,alpha):
        organ,center,n=self.surface_seed(rng)
        camera=self.camera_aim(center);camera_world=self.s.cameras(camera,spec)['left']
        if n@(camera_world[:3,3]-center)<0:raise ValueError('Contact surface faces away from camera')
        tangent=camera_world[:3,0]-n*(n@camera_world[:3,0]);tangent/=np.linalg.norm(tangent)
        R=self.R0@Rotation.from_euler('xyz',rng.uniform(-10,10,3),degrees=True).as_matrix()
        length=rng.uniform(*self.c['slide_length_mm']);times=np.linspace(0,self.duration,41)
        qp=[];qe=[];seed=self.k.base.copy();offset=self.c['contact_target_mm']+self.c['approach_mm'];previous_goal=offset
        for t in times:
            u=t/self.duration;slide=smoother((u-.4)/.4)*length if spec['kind']=='slide' else 0.
            point,_,idx=organ.mesh.nearest.on_surface([center+tangent*slide]);point=point[0];face=int(idx[0])
            bary=trimesh.triangles.points_to_barycentric(organ.mesh.triangles[[face]],[point])[0]
            normal=bary@organ.mesh.vertex_normals[organ.mesh.faces[face]];normal/=np.linalg.norm(normal)
            goal=contact_offset(u,self.c)
            if spec['kind']=='hover':goal+=self.c['free_clearance_mm']
            angle=np.deg2rad(alpha[0]+(alpha[1]-alpha[0])*smoother(u))
            q,offset=self.contact_pose(organ,point,normal,goal,R,angle,seed,offset+goal-previous_goal)
            seed=q;previous_goal=goal;qp.append(q)
            e=camera.copy()
            if spec['category'] in ('ecm_motion','combined_pose_jaw'):
                e[0]+=np.deg2rad(.3)*np.sin(np.pi*u);e[1]+=np.deg2rad(.2)*np.sin(np.pi*u)
            qe.append(e)
        spec['surface']={'prim':organ.name,'center_mm':center.tolist(),'initial_normal':n.tolist(),'slide_length_mm':length if spec['kind']=='slide' else 0.}
        return Trajectory.create(times,qp,qe,continuous=True).with_jaw_curve(self.k,alpha),organ

    def occlusion_path(self,rng,spec,alpha):
        """Find a safe joint-space excursion around a real anatomical silhouette.

        The seeds are recorded scene-specific design poses, never copied labels.
        Every candidate and the entire interpolated path is checked anew.
        """
        design=self.c['occlusion_search'];names=design['joint_names']
        center=np.array(design['seed_positions'][int(rng.integers(len(design['seed_positions'])))])
        half=np.array(design['seed_half_ranges']);q=self.k.base.copy()
        for n,value,span in zip(names,center,half):
            i=self.k.names.index(n);lo=max(self.k.lower[i]+1e-6,value-span);hi=min(self.k.upper[i]-1e-6,value+span)
            q[i]=rng.uniform(lo,hi)
        mid_angle=np.deg2rad(sum(alpha)/2);q=jaw_positions(q,self.k,mid_angle)
        _,_,tips=self.s.points(q);qe=self.camera_aim(tips.mean(0),screen_x=.48)
        if not self.safe.check(q,qe,self.c['free_clearance_mm'])['safe']:raise ValueError('Occlusion seed is not collision-free')
        cameras=self.s.cameras(qe,spec)
        def visibility(qp):
            self.safe.move(qp,qe);_,_,points=self.s.points(qp)
            return [self.safe.visible(cam[:3,3],point) for cam in cameras.values() for point in points]
        def tissue(v):return bool(v['occluder'] and v['occluder'].startswith(self.c['anatomy_root']+'/'))
        mid=visibility(q);hidden=[i for i,v in enumerate(mid) if tissue(v)]
        if not hidden:raise ValueError('Occlusion seed has no tissue-hidden endpoint')
        indices=[self.k.names.index(n) for n in names];excursion=np.array(design['excursion_half_ranges'])
        for _ in range(design['visible_candidates']):
            visible=q.copy();visible[indices]+=rng.uniform(-1,1,len(names))*excursion
            first=jaw_positions(visible,self.k,np.deg2rad(alpha[0]));last=jaw_positions(visible,self.k,np.deg2rad(alpha[1]))
            if np.any(first<self.k.lower) or np.any(first>self.k.upper):continue
            if not self.safe.check(first,qe,self.c['free_clearance_mm'])['safe']:continue
            if not self.safe.check(last,qe,self.c['free_clearance_mm'])['safe']:continue
            v0,v1=visibility(first),visibility(last)
            targets=[i for i in hidden if not tissue(v0[i]) and not tissue(v1[i])]
            if not targets:continue
            chosen=targets[0];spec['occlusion_target']={'eye':('left','right')[chosen//2],'jaw_index':chosen%2,
                'transition':'tissue_clear_to_tissue_occluded_to_tissue_clear',
                'surface_visibility_is_separately_labeled':True}
            spec['occluding_surface']=mid[chosen]['occluder']
            return Trajectory.create([0,self.duration/2,self.duration],[first,q,last],[qe,qe,qe]),None
        raise ValueError('No collision-free reveal near the occlusion seed')

    def inspect(self,path,spec):
        path.validate_bounds(self.s);count=self.c['frames_per_sequence'];records=[];cache={}
        print('CHECKING_FRAMES',spec.get('sequence_id','candidate'),count,flush=True)
        iscontact=spec['kind'] in ('contact','slide')
        def check(t):
            qp,qe=path.at(t);ph=phase(spec['kind'],t/self.duration)
            free=spec['kind'] not in ('contact','slide')
            r=self.safe.check(qp,qe,self.c['free_clearance_mm'] if free else 0.)
            if r['safe'] and iscontact and ph.startswith('contact') and r['organ_distance_mm']>self.c['contact_max_mm']:
                r=dict(r,safe=False,reason='contact_lost')
            if r['safe'] and iscontact and ph.startswith('contact') and r['organ_pair'][1]!=spec['surface']['prim']:
                r=dict(r,safe=False,reason='contact_switched_tissue')
            return r
        for i in range(count):
            t=i/self.c['fps'];qp,qe=path.at(t);r=check(t);cache[t]=r
            if not r['safe']:raise ValueError(r.get('reason','unsafe')+' '+str(r.get('pair','')))
            labels=self.s.labels(qp,qe,spec)
            if not all(all(x['in_frame']) for x in labels.values()):raise ValueError('Landmark outside stereo frame')
            if min(np.linalg.norm(np.array(x['tip_pixel'])-x['shaft_reference_pixel']) for x in labels.values())<50:raise ValueError('Instrument projected span too small')
            cams=self.s.cameras(qe,spec);_,_,tips=self.s.points(qp)
            vis={eye:[self.safe.visible(cam[:3,3],p) for p in tips] for eye,cam in cams.items()}
            records.append({'frame_index':i,'time_s':t,'timestamp_ns':round(t*1e9),'phase':phase(spec['kind'],t/self.duration),
                'q_psm':qp.tolist(),'q_ecm':qe.tolist(),'contact':{'geometric_contact':bool(r['organ_distance_mm']<=self.c['contact_max_mm']),
                'organ_distance_mm':r['organ_distance_mm'],'closest_pair':r['organ_pair'],'model':'near_zero_gap_not_force_simulation'},
                'jaw_visibility':vis})
        if spec['kind']=='occlusion':
            target=spec['occlusion_target'];visibility=[r['jaw_visibility'][target['eye']][target['jaw_index']] for r in records]
            hidden=[bool(v['occluder'] and v['occluder'].startswith(self.c['anatomy_root']+'/')) for v in visibility]
            longest=current=0
            for value in hidden:current=current+1 if value else 0;longest=max(longest,current)
            if longest<15 or hidden[0] or hidden[-1]:raise ValueError('No visible-occluded-visible transition')
        print('CHECKING_INTERVALS',spec.get('sequence_id','candidate'),count-1,flush=True)
        for i in range(count-1):
            r=self.safe.sweep(path,i/self.c['fps'],(i+1)/self.c['fps'],check,cache=cache)
            if not r['safe']:raise ValueError('Unsafe interval: '+r.get('reason','unknown'))
        return records,sum(isinstance(k,(float,int)) for k in cache)

    def sequence(self,sid):
        category,kind=CASES[sid%len(CASES)];rng=np.random.default_rng(self.c['seed']+sid);failures=collections.Counter()
        for attempt in range(self.c['max_candidates']):
            spec={'sequence_id':f'seq_{sid:03d}','category':category,'kind':kind,'seed':int(self.c['seed']+sid),
                'split':'test' if sid%14 in (11,13) else 'validation' if sid%14 in (5,9) else 'train',
                'baseline_scale':float(rng.uniform(.95,1.05)) if category=='stereo_rig' else 1.,
                'relative_yaw_delta_deg':float(rng.uniform(-.5,.5)) if category=='stereo_rig' else 0.,
                'light_start':float(rng.uniform(.7,1.3)) if category in ('lighting','combined_pose_jaw') else 1.,
                'light_end':float(rng.uniform(.7,1.3)) if category in ('lighting','combined_pose_jaw') else 1.,
                'noise_multiplier':float(rng.uniform(.5,1.5)) if category=='sensor_quality' else 1.,
                'blur_multiplier':float(rng.uniform(.75,1.25)) if category=='sensor_quality' else 1.}
            a0=rng.uniform(3,27);a1=rng.uniform(max(0,a0-8),min(30,a0+8));alpha=(a0,a1)
            if sid%14==0:alpha=(30.,22.)
            if sid%14==6:alpha=(12.,16.)
            if sid%14==7:alpha=(20.,16.)
            if sid%14==12:alpha=(0.,8.)
            if sid%14==13:alpha=(14.,20.)
            try:
                if kind in ('slide','contact','hover'):path,_=self.contact_path(rng,spec,alpha)
                elif kind=='occlusion':path,_=self.occlusion_path(rng,spec,alpha)
                else:path,_=self.free_path(rng,spec,alpha)
                records,checked=self.inspect(path,spec)
                spec.update(path=path.record,frames=records,preflight={'passed':True,'states_checked':checked,'attempt':attempt+1,'rejections':dict(failures)})
                print('PLANNED',spec['sequence_id'],category,kind,'attempt',attempt+1,'checked',checked,flush=True)
                return spec
            except ValueError as e:
                reason=str(e).split(' /')[0];failures[reason]+=1
                if attempt%10==0:print('REPLAN',sid,attempt,reason[:160],flush=True)
        raise RuntimeError(f'No valid trajectory for {category}/{kind}: {dict(failures)}')

def select_jobs(sequences,c):
    jobs=[];selected={}
    small=c.get('validation_selection','standard_32')=='categories_14'
    for seq in sequences:
        if seq['category'] in selected:continue
        selected[seq['category']]=seq
        start=c['frames_per_sequence']//2
        for i in range(start,start+(2 if small else 4)):jobs.append({'sequence_id':seq['sequence_id'],'frame_index':i,'reason':'continuous_category_clip'})
    if small:
        if len(selected)!=7 or len(jobs)!=14:raise ValueError('Small validation must include exactly seven two-frame categories')
        for i,j in enumerate(jobs):j['job_id']=f'validation_{i:03d}'
        if len(jobs)>min(32,c['validation_render_limit']):raise ValueError('Validation selection exceeds render cap')
        return jobs
    free=sequences[0];contact=next(s for s in sequences if s['kind'] in ('contact','slide'))
    jobs.extend([{'sequence_id':free['sequence_id'],'frame_index':0,'jaw_override_deg':0.,'reason':'closed_jaw'},
                 {'sequence_id':free['sequence_id'],'frame_index':0,'jaw_override_deg':30.,'reason':'maximum_opening'},
                 {'sequence_id':contact['sequence_id'],'frame_index':c['frames_per_sequence']//2,'reason':'contact_keyframe'},
                 {'sequence_id':contact['sequence_id'],'frame_index':c['frames_per_sequence']-1,'reason':'withdraw_keyframe'}])
    jobs=jobs[-4:-2]+jobs[:-4]+jobs[-2:] # Check both visible jaw limits before the longer category run.
    for i,j in enumerate(jobs):j['job_id']=f'validation_{i:03d}'
    if len(jobs)>min(32,c['validation_render_limit']):raise ValueError('Validation selection exceeds render cap')
    return jobs

def jaw_keyframes(scene,safety,sequences,jobs):
    """Use one safe wrist view for both jaw limits, with a visible surface tip.

    These are isolated validation states, not samples inserted into a sequence.
    """
    if not any('jaw_override_deg' in j for j in jobs):return
    free=sequences[0];frame=free['frames'][0];base=np.array(frame['q_psm']);qe=np.array(frame['q_ecm'])
    pitch=scene.psm.names.index('psm_tool_pitch_joint');roll=scene.psm.names.index('psm_tool_roll_joint')
    for angle in (base[pitch],np.deg2rad(60),np.deg2rad(-60),np.deg2rad(75),np.deg2rad(-75)):
        for twist in (base[roll],base[roll]+np.pi/2,base[roll]-np.pi/2):
            pose=base.copy();pose[pitch]=angle;pose[roll]=twist;states=[];visible=0
            if np.any(pose<scene.psm.lower) or np.any(pose>scene.psm.upper):continue
            for jaw in (0.,30.):
                q=jaw_positions(pose,scene.psm,np.deg2rad(jaw))
                if not safety.check(q,qe,scene.c['free_clearance_mm'])['safe']:break
                labels=scene.labels(q,qe,free)
                if not all(all(x['in_frame']) for x in labels.values()):break
                _,_,tips=scene.points(q)
                visible+=sum(safety.visible(cam[:3,3],point)['visible'] for cam in scene.cameras(qe,free).values() for point in tips)
                states.append(q)
            if len(states)==2 and visible:
                for job,q in zip([j for j in jobs if 'jaw_override_deg' in j],states):job['q_psm_override']=q.tolist()
                return
    raise ValueError('Cannot find a safe jaw-limit validation view with a visible physical endpoint')

def main():
    a=arguments();fp=initialize_run(a);s=Scene(a.machine_data,a.settings_data);safety=Safety(s);planner=Planner(s,safety)
    if (a.run/'trajectories/index.json').exists():require_seal(a.run/'trajectories/index.json',fp)
    write(a.run/'config/camera_effective.json',s.profile);write(a.run/'config/camera_source.json',s.original_profile)
    write(a.run/'config/appearance_source.json',s.appearance);write(a.run/'validation/geometry_inventory.json',safety.audit)
    write(a.run/'config/joints.json',{'psm':s.psm.audit(),'ecm':s.ecm.audit(),'order':{'psm':s.psm.names,'ecm':s.ecm.names}})
    sequences=[]
    priority=[12,13]+[i for i in range(s.c['sequence_count']) if i not in (12,13)]
    for sid in priority:
        p=a.run/'trajectories'/f'seq_{sid:03d}.json'
        if p.exists():
            require_seal(p,fp)
            seq=read(p)
            if seq['fingerprint']!=fp:raise ValueError('Trajectory cache fingerprint mismatch')
        else:seq=planner.sequence(sid);seq['fingerprint']=fp;write(p,seq);seal(p,fp)
        sequences.append(seq)
    sequences.sort(key=lambda seq:seq['sequence_id'])
    jobs=select_jobs(sequences,s.c)
    jaw_keyframes(s,safety,sequences,jobs)
    # Keyframe overrides receive the same geometry and limits checks before rendering.
    for job in jobs:
        seq=next(sq for sq in sequences if sq['sequence_id']==job['sequence_id']);frame=seq['frames'][job['frame_index']]
        if 'jaw_override_deg' in job:
            qp=np.array(job['q_psm_override'])
            check=safety.check(qp,frame['q_ecm'],s.c['free_clearance_mm'])
            if not check['safe']:raise ValueError('Unsafe jaw endpoint keyframe: '+str(check))
    write(a.run/'trajectories/index.json',{'fingerprint':fp,'sequences':[s['sequence_id'] for s in sequences],
        'total_pairs':sum(len(s['frames']) for s in sequences),'validation_jobs':jobs,
        'trajectory_hashes':{s['sequence_id']:sha(a.run/'trajectories'/f"{s['sequence_id']}.json") for s in sequences}})
    seal(a.run/'trajectories/index.json',fp)
    write(a.run/'status.json',{'state':'planned','mode':a.mode,'complete':False,'sequences':len(sequences),'validation_pairs':len(jobs)})
    print('PLAN_COMPLETE',len(sequences),'sequences;',len(jobs),'validation pairs; NO full rendering on laptop',flush=True)

if __name__=='__main__':main()
