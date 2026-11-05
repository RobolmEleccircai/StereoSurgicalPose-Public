"""No-render distributed scheduling, provenance, resume and merge tests.

Tiny file fixtures test transport/receipts only, never claim RTX image acceptance.
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dispatch as d
import generate
from io_utils import read, write, sha, canonical_hash, source_hash, seal, commit_label, assert_mode, reserve_render, lock


def config():
    return read(ROOT/'settings.balanced21.json')


def deps(root='/old/assets'):
    return {'unresolved': [], 'files': [{'path':root+'/scene.usdc','sha256':'scene','bytes':9},
                {'path':root+'/texture.png','sha256':'texture','bytes':12}],
            'active_asset_attributes':[{'attribute':'/mat/file','resolved_files':[root+'/texture.png']}],
            'isaac_version':'5.1.0', 'python_packages':{'numpy':'1.26.0'}}


def plan_fixture(root):
    c=config(); fp='fixture-common-fingerprint'
    write(root/'config/manifest.json',{'settings':c,'code_sha256':source_hash(),
          'fingerprint':fp,'dependencies':deps()})
    for name in d.CONFIG_FILES:
        if name!='manifest.json': write(root/'config'/name,c if name=='effective_settings.json' else {})
    index={'fingerprint':fp,'total_pairs':2100,'sequences':[],'sequence_frame_counts':{},
           'trajectory_hashes':{},'validation_jobs':[]}
    for row in c['sequence_manifest']:
        sid=row['sequence_id'];path=root/'trajectories'/f'{sid}.json'
        write(path,{'fingerprint':fp,'sequence_id':sid,'frames':[{'frame_index':i} for i in range(row['frame_count'])]})
        seal(path,fp); index['sequences'].append(sid);index['sequence_frame_counts'][sid]=row['frame_count']
        index['trajectory_hashes'][sid]=sha(path)
    write(root/'trajectories/index.json',index);seal(root/'trajectories/index.json',fp)
    return c,index


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.c=config();self.workers=d.assignments(self.c)

    def bundle(self):
        plan=self.root/'plan'; plan_fixture(plan)
        # Geometry/policy numerical acceptance has independent real-plan tests.
        with patch.dict(sys.modules,{'dataset_policy':SimpleNamespace(audit=lambda seqs,c:{'passed':True})}):
            manifest=d.prepare(plan,self.root/'dispatch')
        return self.root/'dispatch',manifest

    def runtime(self):
        return patch.dict(sys.modules,{'runtime':SimpleNamespace(
            dependency_audit=lambda m,c:deps('/relocated/assets'),
            source_paths=lambda m,c:{'source_scene':Path('/relocated/assets/scene.usdc')})})

    def make_worker(self,folder,manifest,name,images=True):
        run=self.root/name;m={'role':self.workers[name]['role'],'worker_id':name,'output_root':str(self.root)}
        with self.runtime(): ctx=d.prepare_worker(folder,run,m,self.c,name)
        if not images:return run,m,ctx
        for filename in d.RENDER_CONFIG_FILES:write(run/'config'/filename,{'fixture':True})
        index=read(run/'trajectories/index.json');jobs=d.jobs_for(index,ctx)
        for job in jobs:
            sid,ident=job['sequence_id'],job['job_id'];base=run/'sequences'/sid
            rgb=base/'images'/(ident+'.png');rgb.parent.mkdir(parents=True,exist_ok=True);rgb.write_bytes(b'fixture-not-rendered')
            truth=base/'camera_truth'/(ident+'.json');write(truth,{})
            record={'schema_version':4,'fingerprint':manifest['fingerprint'],**job,'render_worker':ctx,
                'eyes':{eye:{'image':'../images/'+rgb.name,'image_sha256':sha(rgb),
                             'mask':'../images/'+rgb.name,'mask_sha256':sha(rgb)} for eye in ('left','right')},
                'camera_truth':'../camera_truth/'+truth.name,'camera_truth_sha256':sha(truth),
                'tip_region_definition':'../../../config/tip_regions.json',
                'tip_region_definition_sha256':sha(run/'config/tip_regions.json')}
            commit_label(base/'labels'/(ident+'.json'),record)
        (run/'previews').mkdir()
        write(run/'validation/render_inventory.json',{'jobs':jobs,'pairs':len(jobs),'complete':True,'worker':ctx})
        write(run/'validation/report.json',{'passed':True,'rendered_pairs':len(jobs)})
        write(run/'status.json',d.completion_status('shard',len(jobs),index,ctx))
        return run,m,ctx

    def test_exact_disjoint_work_and_split_counts(self):
        _,index=plan_fixture(self.root/'p')
        a=d.jobs_for(index,self.workers['rtx4060']);b=d.jobs_for(index,self.workers['rtx4080super'])
        self.assertEqual((len(a),len(b)),(352,1748))
        keys=lambda jobs:{(j['sequence_id'],j['frame_index']) for j in jobs}
        self.assertFalse(keys(a)&keys(b));self.assertEqual(len(keys(a)|keys(b)),2100)
        rows={r['sequence_id']:r for r in self.c['sequence_manifest']}
        self.assertEqual({s:sum(rows[j['sequence_id']]['split']==s for j in a+b) for s in ('train','validation','test')},
                         {'train':1400,'validation':350,'test':350})

    def test_overlap_missing_unknown_count_and_laptop_overallocation_rejected(self):
        for change in ('overlap','missing','unknown','count','overallocation'):
            workers=copy.deepcopy(self.workers)
            if change=='overlap':workers['rtx4080super']['sequence_ids'][0]='seq_003'
            if change=='missing':workers['rtx4080super']['sequence_ids'].pop()
            if change=='unknown':workers['rtx4080super']['sequence_ids'][0]='seq_999'
            if change=='count':workers['rtx4060']['pair_count']=351
            if change=='overallocation':
                workers['rtx4060']['sequence_ids'].append('seq_000');workers['rtx4060']['pair_count']+=180
                workers['rtx4080super']['sequence_ids'].remove('seq_000');workers['rtx4080super']['pair_count']-=180
            with self.assertRaises(ValueError,msg=change):d.validate_assignments(workers,self.c)

    def test_machine_binding_and_old_laptop_full_guard(self):
        _,manifest=self.bundle()
        m={'role':'laptop','worker_id':'rtx4060'}
        assert_mode('shard',m)
        with self.assertRaises(ValueError):assert_mode('generate',m)
        with self.assertRaises(ValueError):d.worker_context(manifest,'rtx4080super',m)
        with self.assertRaises(ValueError):d.worker_context(manifest,'rtx4060',{'role':'server','worker_id':'rtx4060'})

    def test_relocation_preserves_identity_but_changed_assets_do_not(self):
        self.assertEqual(d.portable_dependencies(deps('/a')),d.portable_dependencies(deps('/b')))
        changed=deps('/a');changed['files'][1]['sha256']='changed'
        self.assertNotEqual(d.portable_dependencies(deps('/a')),d.portable_dependencies(changed))

    def test_frozen_dispatch_changes_and_code_version_rejected(self):
        folder,manifest=self.bundle();d.load_dispatch(folder,self.c)
        bad=copy.deepcopy(manifest);bad['workers']['rtx4060']['sequence_ids'].append('seq_000')
        with self.assertRaises(ValueError):d.validate_dispatch(bad,self.c)
        with patch.object(d,'source_hash',return_value='new-code'):
            with self.assertRaises(ValueError):d.load_dispatch(folder,self.c)
        p=folder/'trajectories/seq_003.json';p.write_text(p.read_text()+' ')
        with self.assertRaises(ValueError):d.load_dispatch(folder,self.c)

    def test_worker_preparation_is_portable_resumable_and_assignment_is_immutable(self):
        folder,manifest=self.bundle();run,m,ctx=self.make_worker(folder,manifest,'rtx4060',images=False)
        before=sha(run/'trajectories/seq_003.json')
        with self.runtime():d.prepare_worker(folder,run,m,self.c,'rtx4060')
        self.assertEqual(sha(run/'trajectories/seq_003.json'),before)
        self.assertEqual(d.run_context(run,m,self.c),ctx)
        self.assertEqual(read(run/'config/manifest.json')['source_paths']['source_scene'],'/relocated/assets/scene.usdc')
        with self.runtime():
            with self.assertRaises(ValueError):d.prepare_worker(folder,run,{'role':'server','worker_id':'rtx4080super'},self.c,'rtx4080super')

    def test_interrupted_preparation_resumes_without_replanning(self):
        folder,manifest=self.bundle();run=self.root/'interrupted'
        machine={'role':'laptop','worker_id':'rtx4060'}
        with self.runtime(),patch.object(d,'copy_atomic',side_effect=OSError('injected interruption')):
            with self.assertRaises(OSError):d.prepare_worker(folder,run,machine,self.c,'rtx4060')
        self.assertTrue((run/'.dispatch_preparing.json').exists())
        with self.runtime():d.prepare_worker(folder,run,machine,self.c,'rtx4060')
        self.assertFalse((run/'.dispatch_preparing.json').exists())
        self.assertEqual(sha(run/'trajectories/seq_003.json'),sha(folder/'trajectories/seq_003.json'))

    def test_only_assigned_jobs_can_render_and_corrupt_pair_can_retry(self):
        folder,manifest=self.bundle();run,m,ctx=self.make_worker(folder,manifest,'rtx4060',images=False)
        for _ in range(2):reserve_render(m,self.c,run.name,'seq_003/000000',worker=ctx)
        for key in ('seq_000/000000','seq_003/000180','seq_003/-00001'):
            with self.assertRaises(ValueError):reserve_render(m,self.c,run.name,key,worker=ctx)
        ledger=read(next((self.root/'.worker_claims').glob('*.budget.json')))
        self.assertEqual(len(ledger['attempts']),2)
        with self.assertRaises(ValueError):reserve_render(m,self.c,'other-run','seq_003/000000',worker=ctx)

    def test_inventory_missing_duplicate_and_wrong_owner_rejected(self):
        _,index=plan_fixture(self.root/'p');ctx=self.workers['rtx4060'];jobs=d.jobs_for(index,ctx)
        inv={'jobs':jobs,'pairs':352,'complete':True,'worker':ctx};d.validate_inventory(inv,index,ctx)
        for bad in ({**inv,'jobs':jobs[:-1]}, {**inv,'jobs':jobs[:-1]+[jobs[0]]}, {**inv,'worker':{}}, {**inv,'complete':False}):
            with self.assertRaises(ValueError):d.validate_inventory(bad,index,ctx)

    def test_shard_cannot_claim_complete_dataset(self):
        _,index=plan_fixture(self.root/'p')
        status=d.completion_status('shard',352,index,self.workers['rtx4060'])
        self.assertTrue(status['shard_complete']);self.assertFalse(status['dataset_complete'])
        with self.assertRaises(ValueError):d.completion_status('merge',352,index)
        self.assertTrue(d.completion_status('merge',2100,index)['dataset_complete'])

    def test_merge_all_2100_files_without_rewriting_labels_and_detect_corruption(self):
        folder,manifest=self.bundle()
        a,_,_=self.make_worker(folder,manifest,'rtx4060');b,m,_=self.make_worker(folder,manifest,'rtx4080super')
        before=sha(a/'sequences/seq_003/labels/000000.json')
        out=self.root/'merged'
        with self.runtime():d.merge([a,b],out,m,self.c)
        self.assertEqual(sha(out/'sequences/seq_003/labels/000000.json'),before)
        self.assertEqual(len(list((out/'sequences').glob('*/labels/*.json'))),2100)
        self.assertFalse(read(out/'status.json')['dataset_complete'])
        self.assertEqual(read(out/'validation/render_inventory.json')['pairs'],2100)
        with self.assertRaises(ValueError):d.inspect_shards([a,a],self.c)
        path=a/'sequences/seq_003/images/000000.png';path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):d.inspect_shards([a,b],self.c)

    def test_prepare_only_launches_no_capture_and_worker_cannot_use_second_run(self):
        folder,manifest=self.bundle();isaac=self.root/'isaac';isaac.mkdir();(isaac/'VERSION').write_text('5.1.0')
        (isaac/'extscache/omni.usd.libs-test').mkdir(parents=True)
        machine=self.root/'machine.json';settings=self.root/'settings.json'
        write(machine,{'role':'laptop','worker_id':'rtx4060','isaac_root':str(isaac),'asset_root':str(self.root),
            'deps_root':str(self.root),'output_root':str(self.root/'runs'),'expected_isaac_version':'5.1.0'})
        write(settings,self.c)
        cmd=['generate.py','--mode','shard','--machine',str(machine),'--settings',str(settings),
             '--dispatch',str(folder),'--worker','rtx4060','--prepare-only','--run-id','assigned']
        with patch.object(sys,'argv',cmd),patch.object(generate.subprocess,'run') as subprocess:
            generate.main();self.assertEqual(subprocess.call_count,1)
            args=subprocess.call_args.args[0];self.assertIn(str(ROOT/'dispatch.py'),args);self.assertNotIn(str(ROOT/'capture.py'),args)
        with patch.object(sys,'argv',cmd[:-1]+['another']),patch.object(generate.subprocess,'run'):
            with self.assertRaises(ValueError):generate.main()

    def test_same_worker_lock_excludes_parallel_process(self):
        with lock(self.root/'worker.lock'):
            with self.assertRaises(RuntimeError):
                with lock(self.root/'worker.lock'):pass


if __name__=='__main__':unittest.main()
