"""Policy failure injection without Isaac Sim; geometry tests remain test_suite.py."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dataset_policy as p
from paths import Trajectory,phase
from io_utils import validate_settings


def configuration():return json.loads((ROOT/'settings.balanced21.json').read_text())


def sequence(row,offset):
    s=p.sample_conditions(row,np.random.default_rng(row['seed']));n=s['parent_frame_count']
    target={'eye':'left','jaw_index':0};s['occlusion_target']=target
    parent=[]
    for i in range(n):
        angle=s['jaw_angles_deg'][0]+(s['jaw_angles_deg'][1]-s['jaw_angles_deg'][0])*max(0,i-15)/(n-16)
        hidden=s['kind']=='occlusion' and .25*n<=i<.7*n
        v={'visible':not hidden,'occluder_kind':'tissue' if hidden else None,'occluder':'/models_topo_blender/organ' if hidden else None}
        parent.append(dict(frame_index=i,time_s=i/60,timestamp_ns=round(i*1e9/60),
            q_psm=[offset,i*.001,angle],q_ecm=[0],v_psm=[0,.06,0],v_ecm=[0],
            jaw_half_angle_deg=angle,appearance=p.effects_at(s,i,60),
            jaw_visibility={'left':[v.copy(),v.copy()],'right':[v.copy(),v.copy()]},phase=phase(s['kind'],i/(n-1))))
    s['parent_frames']=parent
    x=np.linspace(0,4,33)+offset
    d={'C_mm':np.c_[x,np.zeros((33,2))].tolist(),'R_world':np.tile(np.eye(3),(33,1,1)).tolist(),
       'q_psm':np.c_[x,np.zeros((33,2))].tolist(),'joint_names':['base','psm_tool_pitch_joint','psm_tool_gripper2_joint']}
    s['trajectory_descriptor']=d;s['trajectory_group_id']=p.group_id(d)
    return s


def fixture():
    c=configuration();out=[]
    for split in p.SPLITS:
        for row in c['sequence_manifest']:
            if row['split']!=split:continue
            s=sequence(row,int(row['sequence_id'][-3:])*10)
            train=next((x for x in out if x['category']==s['category'] and x['split']=='train'),None)
            s['frames']=p.choose_clip(s,s['parent_frames'],60,train);out.append(s)
    return c,out


class PolicyTests(unittest.TestCase):
    def test_exact_manifest_and_legacy(self):
        c=configuration();validate_settings(c)
        validate_settings(json.loads((ROOT/'settings.json').read_text()))
        self.assertEqual({s:sum(r['frame_count'] for r in c['sequence_manifest'] if r['split']==s) for s in p.SPLITS},p.TOTALS)
        bad=copy.deepcopy(c);bad['sequence_manifest'][1]['frame_count']=33
        with self.assertRaises(ValueError):validate_settings(bad)

    def test_all_21_clips_and_32_jobs(self):
        c,seqs=fixture();report=p.audit(seqs,c)
        self.assertEqual(report['planned_pairs'],2100)
        jobs=p.diagnostic_jobs(seqs)
        self.assertEqual(len(jobs),32);self.assertEqual(len({j['sequence_id'] for j in jobs}),21)
        self.assertEqual(sum(j['reason']=='balanced_adjacent_sample' for j in jobs),7)
        for j in jobs:
            if j['reason']=='balanced_adjacent_sample':
                self.assertIn((j['sequence_id'],j['frame_index']-1),[(x['sequence_id'],x['frame_index']) for x in jobs])

    def test_source_time_and_effects_not_retimed(self):
        c,seqs=fixture();s=next(x for x in seqs if x['category']=='combined_pose_jaw' and x['split']=='test')
        self.assertGreater(s['clip_start_frame'],0)
        f=s['frames'][0];self.assertEqual(f['time_s'],0)
        self.assertEqual(f['appearance'],p.effects_at(s,f['source_frame_index'],60))
        self.assertNotEqual(f['appearance'],p.effects_at(s,0,60))
        for field,value in [('source_frame_index',-1),('v_psm',[99,0,0]),('appearance',{})]:
            bad=copy.deepcopy(seqs);b=next(x for x in bad if x['sequence_id']==s['sequence_id']);b['frames'][0][field]=value
            with self.assertRaises(ValueError):p.audit(bad,c)

    def test_missing_train_combination_and_wrong_bin_rejected(self):
        c,seqs=fixture()
        for field,value in [('phase','untrained_state'),('jaw_half_angle_deg',30.)]:
            bad=copy.deepcopy(seqs);s=next(x for x in bad if x['category']=='sensor_quality' and x['split']=='test')
            for f in s['frames']:f[field]=value
            with self.assertRaises(ValueError):p.audit(bad,c)
        bad=copy.deepcopy(seqs);bad[0]['noise_multiplier']=9
        with self.assertRaises(ValueError):p.audit(bad,c)

    def test_occlusion_requires_clear_hidden_recovered(self):
        _,seqs=fixture();s=next(x for x in seqs if x['category']=='occlusion' and x['split']=='test')
        self.assertTrue(p.occlusion_coverage(s['frames'],s['occlusion_target']))
        hidden=[f for f in s['frames'] if p.target_hidden(f,s['occlusion_target'])]
        self.assertFalse(p.occlusion_coverage(hidden,s['occlusion_target']))

    def test_short_clip_cannot_claim_a_phase_from_one_boundary_frame(self):
        _,seqs=fixture()
        for category,required,minimum in [('sensor_quality',('approach','contact_hold'),10),
                                           ('combined_pose_jaw',('contact_slide','withdraw'),16)]:
            s=next(x for x in seqs if x['category']==category and x['split']=='test')
            for phase_name in required:self.assertGreaterEqual(sum(f['phase']==phase_name for f in s['frames']),minimum)
            bad=copy.deepcopy(s['frames'])
            for f in bad:f['phase']=required[0]
            bad[-1]['phase']=required[1]
            self.assertFalse(p.adequate_clip_phases(s,bad))

    def test_noise_camera_reversal_and_nearcopy_do_not_create_new_paths(self):
        c,seqs=fixture();a=seqs[0];b=copy.deepcopy(a);b['split']='test';b['noise_multiplier']=1.5;b['baseline_scale']=1.05
        with self.assertRaises(ValueError):p.check_against(b,[a])
        d=copy.deepcopy(a['trajectory_descriptor'])
        for k in ('C_mm','R_world','q_psm'):d[k].reverse()
        self.assertEqual(p.group_id(d),a['trajectory_group_id']);self.assertTrue(p.near_duplicate(d,a['trajectory_descriptor']))
        d['C_mm']=(np.array(d['C_mm'])+[.5,0,0]).tolist();self.assertTrue(p.near_duplicate(d,a['trajectory_descriptor']))
        d['C_mm']=(np.array(d['C_mm'])+[4.,0,0]).tolist();self.assertFalse(p.near_duplicate(d,a['trajectory_descriptor']))

    def test_joint_curve_hold_preserves_other_joints_and_limit(self):
        model=SimpleNamespace(names=['psm_tool_roll_joint','psm_tool_gripper1_joint','psm_tool_gripper2_joint'])
        path=Trajectory.create([0,1.5,3],[[0,0,0],[.1,0,0],[.05,0,0]],[[0],[.01],[0]])
        curve=path.with_jaw_curve(model,[30,24],.25)
        for t in np.linspace(0,3,61):
            self.assertAlmostEqual(curve.at(t)[0][0],path.at(t)[0][0],places=12)
            self.assertAlmostEqual(sum(curve.at(t)[0][1:]),0,places=12)
        self.assertAlmostEqual(np.rad2deg(curve.at(.2)[0][2]),30,places=12)
        self.assertLessEqual(max(curve.speed_bounds()['psm'][1:]),.2)

    def test_indexes_weights_and_incomplete_status(self):
        c,seqs=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p.write_split_indexes(root,seqs)
            info=json.loads((root/'indices/manifest.json').read_text());self.assertFalse(info['dataset_complete'])
            for split,count in p.TOTALS.items():
                rows=[json.loads(x) for x in (root/f'indices/{split}.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),count);self.assertAlmostEqual(sum(x['category_equal_weight'] for x in rows),1.)
                self.assertFalse(any(x['rendered_and_verified'] for x in rows))
            full=[{'sequence_id':s['sequence_id'],'frame_index':i,'reason':'dataset_frame'}
                  for s in seqs for i in range(len(s['frames']))]
            p.write_split_indexes(root,seqs,full)
            self.assertFalse(json.loads((root/'indices/manifest.json').read_text())['dataset_complete'])
            p.write_split_indexes(root,seqs,full,acceptance_passed=True)
            self.assertTrue(json.loads((root/'indices/manifest.json').read_text())['dataset_complete'])
            with self.assertRaises(ValueError):p.write_split_indexes(root,seqs,full[:-1],acceptance_passed=True)

    def test_exposure_visibility_retained_and_coverage_checked(self):
        base={'category':'pose_motion','phase':'free_motion','split':'train',
              'eyes':{'left':{'fixed_tip_visibility':[{'visible':False,'occluder_kind':'instrument'}],
                             'jaw_tip_region_visibility':[{'region_visible':True}]}},
              'quality_flags':{'overexposed_any_eye':False}}
        held=copy.deepcopy(base);held['split']='test'
        self.assertTrue(p.rendered_coverage([base,held])['passed'])
        held['quality_flags']['overexposed_any_eye']=True
        self.assertFalse(p.rendered_coverage([base,held])['passed'])
        trained=copy.deepcopy(base);trained['quality_flags']['overexposed_any_eye']=True
        self.assertTrue(p.rendered_coverage([base,trained,held])['passed'])

if __name__=='__main__':unittest.main()
