"""Explicit architecture transition; never load old optimizer slots by position.

Creates a separate run from a completed baseline checkpoint. Shared parameters
and their Adam states are preserved by name. New heads inherit compatible
weights, but start their own optimizer moments. The source is read-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stereo_pose.model import StereoPoseNet, joint_bounds_from_model
from stereo_pose.runtime import identity, read_json, write_json, verify_manifest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_baseline_class(root):
    name = '_unified_c_frozen_baseline'
    spec = importlib.util.spec_from_file_location(name, root/'stereo_pose/__init__.py',
        submodule_search_locations=[str(root/'stereo_pose')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name+'.model').StereoPoseNet


def parameter_groups(model):
    named = list(model.named_parameters())
    return [[(name, p) for name, p in named if name.startswith('backbone.')],
            [(name, p) for name, p in named if not name.startswith('backbone.')]]


def convert_weights(source, model):
    target = model.state_dict()
    shared, converted = [], []
    for name, value in target.items():
        if name.startswith(('depth_heads.', 'rotation_heads.')):
            kind, eye, layer, field = name.split('.')
            old = source[f'pose_heads.{eye}.{layer}.{field}']
            if kind == 'rotation_heads':
                transferred = old if layer == '0' else old[3:9]
            elif layer == '2':
                transferred = old[2:3]
            elif field == 'weight':
                transferred = torch.zeros_like(value)
                if old.shape[0] != value.shape[0] or old.shape[1] >= value.shape[1]:
                    raise ValueError('Unexpected C-depth input dimension')
                transferred[:, :old.shape[1]] = old
            else:
                transferred = old
            if transferred.shape != value.shape:
                raise ValueError('Converted head shape differs: '+name)
            target[name] = transferred.clone()
            converted.append(name)
        else:
            if name not in source or source[name].shape != value.shape:
                raise ValueError('Unexpected non-head architecture change: '+name)
            target[name] = source[name].clone()
            shared.append(name)
    removed = sorted(set(source)-set(target))
    allowed = ('pose_heads.', 'position.', 'pool_score.', 'joint_head.',
               'guided_joint_pool.', 'guided_joint_temporal.')
    if any(not name.startswith(allowed) for name in removed):
        raise ValueError('Unexpected removed model tensors')
    model.load_state_dict(target, strict=True)
    assert all(torch.equal(model.state_dict()[n], source[n]) for n in shared)
    return shared, converted, removed


def convert_optimizer(saved_optimizer, baseline_model, model):
    old_groups = parameter_groups(baseline_model)
    new_groups = parameter_groups(model)
    if len(saved_optimizer['param_groups']) != len(old_groups):
        raise ValueError('Unexpected optimizer grouping')
    old_ids = {}
    for named, group in zip(old_groups, saved_optimizer['param_groups']):
        if len(named) != len(group['params']):
            raise ValueError('Source parameter order is not the original training grouping')
        old_ids.update((name, slot) for (name, _), slot in zip(named, group['params']))
    output = {'state': {}, 'param_groups': []}
    preserved, reset = [], []
    next_slot = 0
    for named, old_group in zip(new_groups, saved_optimizer['param_groups']):
        group = copy.deepcopy(old_group)
        group['params'] = []
        for name, param in named:
            slot = next_slot; next_slot += 1
            group['params'].append(slot)
            if name.startswith(('depth_heads.', 'rotation_heads.')):
                reset.append(name)
                continue
            if name not in old_ids:
                raise ValueError('No source optimizer mapping: '+name)
            state = saved_optimizer['state'].get(old_ids[name])
            if state is not None:
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if key in state and state[key].shape != param.shape:
                        raise ValueError('Optimizer moment shape mismatch: '+name)
                output['state'][slot] = copy.deepcopy(state)
                preserved.append(name)
        output['param_groups'].append(group)
    # Validate the remapped structure with the actual optimizer implementation.
    probe = torch.optim.AdamW([{'params':[p for _,p in g]} for g in new_groups], fused=False)
    probe.load_state_dict(output)
    for group, named in zip(probe.param_groups, new_groups):
        for param, (name, _) in zip(group['params'], named):
            if name in reset:
                assert param not in probe.state or not probe.state[param]
            elif name in preserved:
                old = saved_optimizer['state'][old_ids[name]]
                for key, value in old.items():
                    actual = probe.state[param][key]
                    if torch.is_tensor(value):
                        assert torch.equal(actual.cpu(), value.cpu()), (name,key)
                    else:
                        assert actual == value
    return output, preserved, reset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-code', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    source_run = args.source_run.resolve(); output = args.output.resolve()
    if output == source_run or (output.exists() and any(output.iterdir())):
        raise ValueError('Architecture conversion requires a separate empty run')
    report = read_json(source_run/'report.json')
    if report.get('status') != 'stopped_after_checkpoint':
        raise ValueError('Source must finish its epoch and save before transition')
    checkpoint_path = source_run/'last.pt'; source_hash = digest(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('diagnostic') or saved.get('trial'):
        raise ValueError('Expected full-data training weights')
    if report['completed_epochs'] != saved['epoch']+1:
        raise ValueError('Source checkpoint and completion report disagree')
    for name, expected in saved['identity']['code'].items():
        if digest(args.source_code/name) != expected:
            raise ValueError('Baseline source identity mismatch: '+name)
    manifest = read_json(source_run/'manifest.json')
    instrument = saved['instrument_model']
    verify_manifest(manifest, instrument)
    if manifest['manifest_sha256'] != saved['identity']['manifest']:
        raise ValueError('Dataset differs from source checkpoint')
    config = copy.deepcopy(saved['config'])
    config['schema_version'] = 'stereo_pose_unified_c_v1'
    config['model']['c_translation_head'] = 'landmark_ray_depth_v1'
    if config['loss_weights'].get('landmark_consistency', 0) != 0:
        raise ValueError('The new architecture does not use a consistency-loss correction')
    baseline_file = output/'baseline_source_validation.json'
    config['training']['comparison_report'] = str(baseline_file)
    config['training']['comparison_label'] = f'Original C head; frozen epoch {saved["epoch"]+1}'
    config['notes'] += (' Architecture transition: one landmark-derived C ray and learned Z per eye; '
        'no independent rayXY regression, no postprocessing correction. '
        'Shared weights and optimizer moments preserved by name; new heads get fresh moments. '
        'Global epoch/scheduler/RNG continue; model-specific best scores restart.')
    old_class = load_baseline_class(args.source_code.resolve())
    baseline_model = old_class(saved['config']['model'], joint_bounds_from_model(instrument), pretrained=False)
    baseline_model.load_state_dict(saved['model'], strict=True)
    model = StereoPoseNet(config['model'], joint_bounds_from_model(instrument), pretrained=False)
    common, converted, removed = convert_weights(saved['model'], model)
    optimizer, preserved_states, reset_states = convert_optimizer(saved['optimizer'], baseline_model, model)
    signature = identity(config, instrument, manifest)
    transition = {'type':'explicit_architecture_transition', 'source_checkpoint':str(checkpoint_path),
        'source_checkpoint_sha256':source_hash, 'source_completed_epoch':saved['epoch']+1,
        'first_new_architecture_epoch':saved['epoch']+2, 'source_identity':saved['identity'],
        'new_identity':signature, 'common_weights_bit_identical':True,
        'shared_tensor_count':len(common), 'converted_head_tensors':converted, 'removed_old_tensors':removed,
        'optimizer_mapping':'by named parameter; new output-head slots reset',
        'optimizer_shared_states_bit_identical':True, 'preserved_optimizer_state_count':len(preserved_states),
        'reset_optimizer_parameter_names':reset_states,
        'scheduler_and_rng_preserved':True, 'old_best_checkpoints_kept_in_source_run':True,
        'dataset_and_split_unchanged':True, 'loss_weights_and_scales_unchanged':True,
        'not_an_ordinary_old_architecture_resume':True}
    converted_checkpoint = dict(saved, model=model.state_dict(), optimizer=optimizer,
        identity=signature, config=config, best=float('inf'), stale=0, best_tip=float('inf'),
        best_training_loss=float('inf'), best_training_loss_epoch=None,
        best_validation_loss=float('inf'), best_validation_loss_epoch=None,
        architecture_transition=transition)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'config.json', config)
    shutil.copy2(source_run/'manifest.json', output/'manifest.json')
    shutil.copy2(source_run/'validation_latest.json', baseline_file)
    history = [json.loads(line) for line in (source_run/'history.jsonl').read_text().splitlines() if line.strip()]
    baseline = next(row for row in history if row['epoch'] == saved['epoch'])
    write_json(output/'baseline_source_epoch.json', baseline)
    write_json(output/'baseline_validation_loss.json', {'source_epoch':saved['epoch']+1,
        'validation_loss':baseline['validation_loss'], 'new_objective':True,
        'meaning':'frozen original architecture; unchanged validation objective'})
    temporary = output/'checkpoint.tmp'; torch.save(converted_checkpoint, temporary)
    temporary.replace(output/'last.pt')
    if digest(checkpoint_path) != source_hash:
        raise RuntimeError('Source checkpoint changed during transition')
    write_json(output/'architecture_transition.json', transition)
    print(json.dumps({'output':str(output),'source_epoch':saved['epoch']+1,
        'next_epoch':saved['epoch']+2,'shared_states':len(preserved_states),
        'new_head_states_reset':len(reset_states),'source_unchanged':True}))


if __name__ == '__main__':
    main()
