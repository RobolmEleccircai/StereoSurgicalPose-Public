"""C2 quintic joint trajectories; polynomial coefficient bounds are conservative."""
import numpy as np
from scipy.interpolate import BPoly,PPoly

def smoother(u):
    u=np.clip(u,0.,1.);return u*u*u*(10+u*(-15+6*u))

class Trajectory:
    def __init__(self,record):
        self.record=record;self.times=np.array(record['knot_times_s']);self.polys={}
        self.segment_speeds={}
        for group in ('psm','ecm'):
            q=np.array(record['q_'+group+'_knots']);v=np.array(record['v_'+group+'_knots']);acc=np.array(record['a_'+group+'_knots'])
            self.polys[group]=BPoly.from_derivatives(self.times,[[a,b,c] for a,b,c in zip(q,v,acc)])
            power=PPoly.from_bernstein_basis(self.polys[group].derivative());bounds=[]
            for i,dt in enumerate(np.diff(self.times)):
                row=[]
                for j in range(q.shape[1]):
                    coeff=power.c[:,i,j];roots=np.roots(np.polyder(coeff))
                    candidates=[0.,dt]+[float(x.real) for x in roots if abs(x.imag)<1e-10 and 0<x.real<dt]
                    row.append(max(abs(np.polyval(coeff,x)) for x in candidates))
                bounds.append(row)
            self.segment_speeds[group]=np.array(bounds)

    @classmethod
    def create(cls,times,psm,ecm,continuous=False):
        times=np.array(times,float);r={'knot_times_s':times.tolist()}
        for group,q in [('psm',psm),('ecm',ecm)]:
            q=np.array(q,float)
            v=np.gradient(q,times,axis=0,edge_order=2) if continuous else np.zeros_like(q)
            acc=np.gradient(v,times,axis=0,edge_order=2) if continuous else np.zeros_like(q)
            v[[0,-1]]=0.;acc[[0,-1]]=0.
            r['q_'+group+'_knots']=q.tolist();r['v_'+group+'_knots']=v.tolist();r['a_'+group+'_knots']=acc.tolist()
        return cls(r)

    def at(self,t):return tuple(self.polys[g](t) for g in ('psm','ecm'))
    def velocity(self,t):return tuple(self.polys[g].derivative()(t) for g in ('psm','ecm'))

    def with_jaw_curve(self,model,angles_deg,hold_start_s=0.):
        """Exact C2 opening curve, including zero-angle hard-stop endpoints."""
        import copy
        record=copy.deepcopy(self.record);duration=self.times[-1]-self.times[0]-hold_start_s
        if duration<=0 or hold_start_s<0:raise ValueError('Invalid jaw hold duration')
        times=self.times
        if hold_start_s:
            times=np.unique(np.r_[times,self.times[0]+hold_start_s])
            record['knot_times_s']=times.tolist()
            for group in ('psm','ecm'):
                for field,order in (('q',0),('v',1),('a',2)):
                    record[field+'_'+group+'_knots']=self.polys[group].derivative(order)(times).tolist()
        u=np.clip((times-self.times[0]-hold_start_s)/duration,0.,1.);start,end=np.deg2rad(angles_deg);delta=end-start
        values=[start+delta*smoother(u),delta*30*u*u*(1-u)**2/duration,
                delta*(60*u-180*u*u+120*u*u*u)/duration**2]
        for field,values_at_knots in zip(('q','v','a'),values):
            data=np.array(record[field+'_psm_knots'])
            for name,sign in [('psm_tool_gripper1_joint',-1.),('psm_tool_gripper2_joint',1.)]:
                data[:,model.names.index(name)]=sign*values_at_knots
            record[field+'_psm_knots']=data.tolist()
        return Trajectory(record)

    def speed_bounds(self,a=None,b=None):
        a=self.times[0] if a is None else a;b=self.times[-1] if b is None else b
        intervals=(self.times[1:]>=a)&(self.times[:-1]<=b)
        return {g:np.max(s[intervals],axis=0) for g,s in self.segment_speeds.items()}

    def validate_bounds(self,scene):
        for group,k in [('psm',scene.psm),('ecm',scene.ecm)]:
            coeff=self.polys[group].c
            if np.any(coeff<k.lower[None,None,:]-1e-7) or np.any(coeff>k.upper[None,None,:]+1e-7):raise ValueError('Continuous joint limit exceeded: '+group)
            if np.any(self.speed_bounds()[group]>k.speed_limits(scene.c)+1e-8):raise ValueError('Continuous joint speed exceeded: '+group)
        return True

def jaw_positions(q,k,alpha):
    q=np.array(q,copy=True);q[k.names.index('psm_tool_gripper1_joint')]=-alpha;q[k.names.index('psm_tool_gripper2_joint')]=alpha
    return q

def phase(kind,u):
    if kind in ('contact','slide'):
        return 'approach' if u<.2 else 'contact_hold' if u<.4 else 'contact_slide' if u<.8 and kind=='slide' else 'contact_hold' if u<.8 else 'withdraw'
    if kind=='hover':return 'approach' if u<.4 else 'hover' if u<.6 else 'withdraw'
    if kind=='occlusion':return 'occlusion_transition'
    return 'free_motion'

def contact_offset(u,c):
    if u<.2:return c['contact_target_mm']+c['approach_mm']*(1-smoother(u/.2))
    if u>.8:return c['contact_target_mm']+c['approach_mm']*smoother((u-.8)/.2)
    return c['contact_target_mm']
