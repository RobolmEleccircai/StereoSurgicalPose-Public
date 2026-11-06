"""Receipt migration and whole-sequence allocation; no synthetic render claims."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import dispatch as ds
import redistribute as rd
from io_utils import read,write,sha,canonical_hash,commit_label,frame_valid
from test_dispatch import plan_fixture,deps


class RedistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.plan=self.root/'plan'
        self.c,self.index=plan_fixture(self.plan)
        self.audit=patch.dict(sys.modules,{'dataset_policy':SimpleNamespace(audit=lambda *args:{'passed':True})})
        self.audit.start();self.addCleanup(self.audit.stop)
        self.old_dir=self.root/'old_dispatch';self.old=ds.prepare(self.plan,self.old_dir)
        self.source=self.root/'stopped';(self.source/'config').mkdir(parents=True)
        manifest=read(self.plan/'config/manifest.json');manifest['dispatch_id']=self.old['dispatch_id']
        write(self.source/'config/manifest.json',manifest)
        for filename in ds.RENDER_CONFIG_FILES:write(self.source/'config'/filename,{'fixture':True})
        self.ctx={'dispatch_id':self.old['dispatch_id'],'worker_id':'rtx4060',**self.old['workers']['rtx4060']}
        base=self.source/'sequences/seq_003'
        image=base/'images/000000.png';image.parent.mkdir(parents=True);image.write_bytes(b'receipt-fixture-not-a-render')
        truth=base/'camera_truth/000000.json';write(truth,{})
        self.label=base/'labels/000000.json'
        record={'schema_version':4,'fingerprint':self.old['fingerprint'],'sequence_id':'seq_003','job_id':'000000',
                'render_worker':self.ctx,'eyes':{eye:{'image':'../images/000000.png','mask':'../images/000000.png',
                  'image_sha256':sha(image),'mask_sha256':sha(image)} for eye in ('left','right')},
                'camera_truth':'../camera_truth/000000.json','camera_truth_sha256':sha(truth),
                'tip_region_definition':'../../../config/tip_regions.json',
                'tip_region_definition_sha256':sha(self.source/'config/tip_regions.json')}
        commit_label(self.label,record)
        write(self.source/'paused_for_redistribution.json',{'renderer_stopped':True,'committed_pairs':1})

    def create(self):
        return rd.create(self.old_dir,self.root/'new_dispatch',ROOT,[self.source],deps(),33.,18.,['seq_000'])

    def test_allocation_is_disjoint_and_speed_weighted(self):
        workers,remaining=rd.choose_workers(self.c,{},33.,18.,['seq_000'],['seq_003','seq_011','seq_013'])
        self.assertIn('seq_000',workers['rtx4080super']['sequence_ids'])
        self.assertTrue({'seq_003','seq_011','seq_013'} <= set(workers['rtx5090']['sequence_ids']))
        with self.assertRaises(ValueError):
            rd.choose_workers(self.c,{},33.,18.,['seq_000'],['seq_000'])
        self.assertGreater(workers['rtx5090']['pair_count'],workers['rtx4080super']['pair_count'])
        self.assertEqual(sum(w['pair_count'] for w in workers.values()),2100)
        self.assertFalse(set(workers['rtx5090']['sequence_ids'])&set(workers['rtx4080super']['sequence_ids']))
        self.assertLess(max(workers['rtx5090']['pair_count']*18,workers['rtx4080super']['pair_count']*33),26000)

    def test_import_keeps_labels_images_and_renderer_identity(self):
        original=sha(self.label);d=self.create()
        self.assertEqual(d['fingerprint'],self.old['fingerprint'])
        self.assertNotEqual(d['dispatch_id'],self.old['dispatch_id'])
        for sid in self.index['sequences']:
            self.assertEqual(sha(self.plan/'trajectories'/f'{sid}.json'),sha(self.root/'new_dispatch/trajectories'/f'{sid}.json'))
        owner=next(name for name,w in d['workers'].items() if 'seq_003' in w['sequence_ids'])
        run=self.root/'destination';m={'role':'server','worker_id':owner}
        fake=SimpleNamespace(dependency_audit=lambda *args:deps(),source_paths=lambda *args:{})
        with patch.dict(sys.modules,{'runtime':fake}):ds.prepare_worker(self.root/'new_dispatch',run,m,self.c,owner)
        self.assertEqual(rd.import_frames([self.source],run,owner),1)
        label=run/'sequences/seq_003/labels/000000.json'
        self.assertEqual(sha(label),original)
        self.assertEqual(read(label)['render_worker']['worker_id'],'rtx4060')
        self.assertTrue(frame_valid(label,d['fingerprint']))
        ctx=read(run/'config/worker.json');ds.validate_frame_owner(label,ctx,d)
        changed=read(label);changed['render_worker']=ctx;write(label,changed)
        with self.assertRaises(ValueError):ds.validate_frame_owner(label,ctx,d)

    def test_runtime_definition_archive_keeps_original_receipt(self):
        definition=self.source/'config/tip_regions.json'
        digest=sha(definition)
        archived=self.source/'config/definition_versions'/f'{digest}.json'
        ds.copy_atomic(definition,archived)
        write(definition,{'fixture':'different runtime serialization'})
        self.assertTrue(frame_valid(self.label,self.old['fingerprint']))
        d=self.create()
        owner=next(name for name,w in d['workers'].items() if 'seq_003' in w['sequence_ids'])
        run=self.root/'destination';m={'role':'server','worker_id':owner}
        fake=SimpleNamespace(dependency_audit=lambda *args:deps(),source_paths=lambda *args:{})
        with patch.dict(sys.modules,{'runtime':fake}):ds.prepare_worker(self.root/'new_dispatch',run,m,self.c,owner)
        rd.import_frames([self.source],run,owner)
        label=run/'sequences/seq_003/labels/000000.json'
        self.assertTrue(frame_valid(label,d['fingerprint']))
        write(run/'paused_for_redistribution.json',{'renderer_stopped':True,'committed_pairs':1})
        newer=rd.create(self.root/'new_dispatch',self.root/'third_dispatch',ROOT,[run],deps(),33,18,['seq_000'])
        ds.validate_dispatch(newer,self.c)
        self.assertEqual(newer['imported_frames']['seq_003/000000']['render_worker'],self.ctx)
        self.assertEqual(sha(label),sha(self.label))
        (run/'config/definition_versions'/f'{digest}.json').write_text('corrupt')
        self.assertFalse(frame_valid(label,d['fingerprint']))

    def test_running_or_corrupt_source_rejected(self):
        write(self.source/'paused_for_redistribution.json',{'renderer_stopped':False,'committed_pairs':1})
        with self.assertRaises(ValueError):self.create()
        write(self.source/'paused_for_redistribution.json',{'renderer_stopped':True,'committed_pairs':1})
        self.label.write_text(self.label.read_text()+' ')
        with self.assertRaises(ValueError):self.create()

    def test_changed_plan_and_missing_original_provenance_rejected(self):
        d=self.create();ds.validate_dispatch(d,self.c)
        bad=copy.deepcopy(d);bad['origin_manifests']={}
        bad['dispatch_id']=canonical_hash({k:v for k,v in bad.items() if k!='dispatch_id'})
        with self.assertRaises(ValueError):ds.validate_dispatch(bad,self.c)
        p=self.old_dir/'trajectories/seq_003.json';p.write_text(p.read_text()+' ')
        with self.assertRaises(ValueError):rd.create(self.old_dir,self.root/'bad',ROOT,[self.source],deps(),33,18,['seq_000'])


if __name__=='__main__':unittest.main()
