"""Relocation must preserve experimental identities and reject damaged inputs."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from reproduction import manifest_digest, path_maps, relocated_manifest, verify_files


def example(tmp_path):
    paths = {}
    for name in ('label', 'left', 'right', 'left_mask', 'right_mask'):
        p = tmp_path / name
        p.write_bytes(name.encode())
        paths[name] = (str(p), hashlib.sha256(p.read_bytes()).hexdigest())
    record = {'split': 'test', 'sequence_id': 'seq_a', 'trajectory_group_id': 'group_a',
              'frame_index': 19, 'timestamp_ns': 316666667,
              'label_path': paths['label'][0], 'label_sha256': paths['label'][1],
              'image_paths': {eye: paths[eye][0] for eye in ('left', 'right')},
              'image_sha256': {eye: paths[eye][1] for eye in ('left', 'right')},
              'mask_paths': {eye: paths[eye+'_mask'][0] for eye in ('left', 'right')},
              'mask_sha256': {eye: paths[eye+'_mask'][1] for eye in ('left', 'right')}}
    manifest = {'records': [record], 'source_scene_sha256': 'asset-identity'}
    manifest['manifest_sha256'] = manifest_digest(manifest)
    return manifest


def test_relocation_preserves_split_order_labels_and_file_content(tmp_path):
    original = tmp_path / 'old'; original.mkdir()
    manifest = example(original)
    before = copy.deepcopy(manifest)
    new = tmp_path / 'new'; new.mkdir()
    for source in original.iterdir():
        (new / source.name).write_bytes(source.read_bytes())
    actual = relocated_manifest(manifest, path_maps([str(original)+'='+str(new)]))
    assert manifest == before
    for key in ('split','sequence_id','trajectory_group_id','frame_index','timestamp_ns','label_sha256'):
        assert actual['records'][0][key] == before['records'][0][key]
    assert actual['manifest_sha256'] != manifest['manifest_sha256']
    assert verify_files(actual) == 5
    (new/'right').write_bytes(b'damaged')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_files(actual)


def test_reject_changed_manifest_and_conflicting_file_hash(tmp_path):
    manifest = example(tmp_path)
    manifest['records'][0]['split'] = 'train'
    with pytest.raises(ValueError, match='manifest hash'):
        relocated_manifest(manifest, [])
    manifest = example(tmp_path)
    other = copy.deepcopy(manifest['records'][0])
    other['label_sha256'] = 'different'
    manifest['records'].append(other)
    with pytest.raises(ValueError, match='Conflicting hashes'):
        verify_files(manifest)


def test_path_mapping_is_bounded_and_longest_prefix_first():
    from reproduction import remap
    maps = path_maps(['/data=/new', '/data/specific=/special'])
    assert remap('/data/specific/a', maps) == '/special/a'
    assert remap('/database/a', maps) == '/database/a'
    assert remap('notes about /data/a', maps) == 'notes about /data/a'
    with pytest.raises(ValueError):
        path_maps(['relative=/absolute'])


def test_archived_run_relocation_preserves_every_training_state(tmp_path, monkeypatch):
    import sys
    import torch
    import relocate_run
    from stereo_pose.runtime import identity
    from reproduction import sha256

    root = Path(__file__).resolve().parents[1]
    instrument = json.loads((root / 'assets/instrument_model.json').read_text())
    manifest = example(tmp_path)
    manifest['source_scene_sha256'] = instrument['source_scene_sha256']
    manifest['manifest_sha256'] = manifest_digest(manifest)
    source = tmp_path / 'source'; source.mkdir()
    (source / 'manifest.json').write_text(json.dumps(manifest))
    config = {'model': {'unchanged': True}, 'comparison': str(source / 'baseline.json')}
    saved = {'identity': identity(config, instrument, manifest), 'config': config,
             'instrument_model': instrument, 'epoch': 9,
             'model': {'weight': torch.arange(6.).reshape(2,3)},
             'optimizer': {'state': {0: {'exp_avg': torch.tensor([.3,.7]), 'step': torch.tensor(123)}}},
             'scheduler': {'last_epoch': 10}, 'rng': torch.get_rng_state(),
             'best_training_loss_epoch': 9, 'best_validation_loss_epoch': 8}
    for name in ('last.pt','best_train.pt','best_val.pt'):
        torch.save(saved, source / name)
    before = {p.name:sha256(p) for p in source.iterdir()}
    output = tmp_path / 'relocated'
    monkeypatch.setattr(sys,'argv',['relocate_run','--source-run',str(source),'--output',str(output)])
    relocate_run.main()
    actual=torch.load(output/'last.pt',weights_only=True)
    for key in ('model','optimizer'):
        def equal(a,b):
            if isinstance(a,torch.Tensor): return torch.equal(a,b)
            if isinstance(a,dict): return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
            return a==b
        assert equal(saved[key],actual[key])
    assert actual['epoch']==9 and actual['scheduler']==saved['scheduler']
    assert torch.equal(actual['rng'],saved['rng'])
    assert actual['config']['comparison']==str(output/'baseline.json')
    assert actual['identity']['code']==saved['identity']['code']
    assert actual['identity']['instrument']==saved['identity']['instrument']
    assert before=={p.name:sha256(p) for p in source.iterdir()}
    assert (output/'best_train.pt').exists() and (output/'best_val.pt').exists()
    assert json.loads((output/'relocation.json').read_text())['training_started'] is False

    (source/'best_val.pt').unlink()
    missing=tmp_path/'missing_best'
    monkeypatch.setattr(sys,'argv',['relocate_run','--source-run',str(source),'--output',str(missing)])
    with pytest.raises(ValueError,match='best_val.pt'):
        relocate_run.main()
    assert not missing.exists()
