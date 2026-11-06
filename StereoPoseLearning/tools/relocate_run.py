"""Rebase an archived run for another machine, retaining learned and optimizer state.

Only explicit path prefixes and the identities derived from those paths change.
The source is read-only. This does not authorize a code/model/data change.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from reproduction import (empty_destination, path_maps, remap, relocated_manifest,
                          sha256, verify_files, write_json)
from stereo_pose.runtime import identity, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--path-map', action='append', default=[])
    args = parser.parse_args()
    source, output = args.source_run.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError('Use a separate destination outside the source run')
    maps = path_maps([str(source) + '=' + str(output), *args.path_map])
    original = json.loads((source / 'manifest.json').read_text())
    manifest = relocated_manifest(original, maps)
    saved = torch.load(source / 'last.pt', map_location='cpu', weights_only=True)
    verify_manifest(original, saved['instrument_model'])
    if saved['identity']['manifest'] != original['manifest_sha256']:
        raise ValueError('Checkpoint and archived manifest differ')
    config = remap(copy.deepcopy(saved['config']), maps)
    actual_instrument = json.loads((ROOT / 'assets/instrument_model.json').read_text())
    if actual_instrument != saved['instrument_model']:
        raise ValueError('Repository instrument definition differs from the checkpoint')
    signature = identity(config, saved['instrument_model'], manifest)
    if saved['identity']['code'] != signature['code']:
        raise ValueError('Source code differs: check out the recorded code version first')
    if saved['identity']['instrument'] != signature['instrument']:
        raise ValueError('Instrument geometry differs')
    verified = verify_files(manifest)
    checkpoints = [p for p in (source / n for n in ('last.pt', 'best_train.pt', 'best_val.pt')) if p.exists()]
    if saved.get('best_training_loss_epoch') and not (source / 'best_train.pt').exists():
        raise ValueError('Archived best_train.pt is required to preserve the best checkpoint')
    if saved.get('best_validation_loss_epoch') and not (source / 'best_val.pt').exists():
        raise ValueError('Archived best_val.pt is required to preserve the best checkpoint')
    output = empty_destination(output)
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', manifest)
    evidence = []
    try:
        for path in checkpoints:
            before_hash = sha256(path)
            item = torch.load(path, map_location='cpu', weights_only=True)
            if item['identity'] != saved['identity']:
                raise ValueError('Best checkpoint belongs to a different run: ' + path.name)
            old_identity = copy.deepcopy(item['identity'])
            item['config'], item['identity'] = config, signature
            torch.save(item, output / path.name)
            if sha256(path) != before_hash:
                raise RuntimeError('Source checkpoint changed during relocation')
            evidence.append({'file': path.name, 'source_sha256': before_hash,
                             'destination_sha256': sha256(output / path.name), 'epoch': item['epoch'] + 1})
        for name in ('history.jsonl', 'progress.jsonl', 'validation_latest.json',
                     'training_loss_best.json', 'validation_loss_best.json',
                     'baseline_source_validation.json', 'baseline_source_epoch.json',
                     'baseline_validation_loss.json', 'architecture_transition.json'):
            if (source / name).exists():
                shutil.copy2(source / name, output / name)
        write_json(output / 'relocation.json', {
            'status': 'complete', 'original_identity': old_identity, 'identity': signature,
            'verified_files': verified, 'path_maps': maps, 'checkpoints': evidence,
            'model_optimizer_scheduler_rng_unchanged': True,
            'record_order_and_split_labels_unchanged': True, 'training_started': False,
            'note': 'Historical logs retain their original identity fields; relocation.json records the mapping.'})
    except Exception as error:
        write_json(output / 'relocation_failed.json', {'error': repr(error)})
        raise
    print(json.dumps({'relocated': str(output), 'epoch': saved['epoch'] + 1, 'verified_files': verified}))


if __name__ == '__main__':
    main()
