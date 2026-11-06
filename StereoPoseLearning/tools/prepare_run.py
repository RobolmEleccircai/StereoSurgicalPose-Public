"""Prepare a fresh run from an existing verified manifest, without changing splits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduction import empty_destination, path_maps, relocated_manifest, verify_files, write_json
from stereo_pose.runtime import verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--path-map', action='append', default=[])
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    manifest = relocated_manifest(source, path_maps(args.path_map))
    instrument = json.loads((ROOT / 'assets/instrument_model.json').read_text())
    verify_manifest(manifest, instrument)
    count = verify_files(manifest)
    config = json.loads(args.config.read_text())
    config['expected_pairs'] = manifest['trajectory_split_counts']
    if any(record.get('isolated_diagnostic') for record in manifest['records']):
        raise ValueError('Use production sequence data for a full training run')
    output = empty_destination(args.output)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'config.json', config)
    write_json(output / 'preparation.json', {
        'source_manifest_sha256': source['manifest_sha256'],
        'manifest_sha256': manifest['manifest_sha256'], 'verified_files': count,
        'path_maps': args.path_map, 'record_order_and_split_labels_unchanged': True,
        'training_started': False})
    print(json.dumps({'prepared': str(output), 'pairs': config['expected_pairs'], 'verified_files': count}))


if __name__ == '__main__':
    main()
