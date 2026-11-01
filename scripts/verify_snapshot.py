#!/usr/bin/env python3
"""Verify the preserved capture snapshot without importing simulator packages."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inventory = json.loads((ROOT / 'docs/source_snapshot.json').read_text())
    for entry in inventory['files']:
        path = ROOT / entry['repository_path']
        require(path.is_file(), f'Missing snapshot file: {path}')
        require(path.stat().st_size == entry['bytes'], f'Size mismatch: {path}')
        require(digest(path) == entry['sha256'], f'Source hash mismatch: {path}')

    # Parse source without executing any simulator or experiment code.
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts]
    for path in files:
        if path.suffix == '.json':
            json.loads(path.read_text())
        elif path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)

    camera = ROOT / 'CameraImagingConfig'
    for line in (camera / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        require(digest(camera / relative) == expected, f'Module hash mismatch: {relative}')

    native = camera / 'src/ShaftTipStereo_2026-11-07/render/native'
    manifest = json.loads((native / 'manifest.json').read_text())
    config = manifest['instrument_configuration']
    require(manifest['completed'] and len(manifest['samples']) == 3, 'Incomplete capture')
    require(config['count'] == 1 and config['active_tools'] == ['right_tool'], 'Not a single instrument')
    require(config['active_prim'] == '/SurgicalTools/PSM_Right', 'Unexpected active tool')
    require(config['inactive_prim'] == '/SurgicalTools/PSM_Left', 'Unexpected inactive tool')
    require(config['ecm_retained'] and config['stereo_camera_count'] == 2, 'Missing ECM/stereo rig')
    for sample_id in manifest['samples']:
        scene = (native / 'scene' / f'sample_{sample_id}.usda').read_text()
        require(re.search(r'over "PSM_Left"\s*\(\s*active = false', scene), 'Left tool not disabled')

    dataset = ROOT / 'SurgicalPoseEstimation/examples/shaft_tip_stereo'
    labels = sorted((dataset / 'ground_truth').glob('*.json'))
    require(len(labels) == 3, 'Expected three label records')
    images_checked = 0
    for path in labels:
        sample = json.loads(path.read_text())
        require(set(sample['eyes']) == {'left', 'right'}, 'Missing eye labels')
        scalar_count = 0
        for eye, record in sample['eyes'].items():
            image = (path.parent / record['image']).resolve()
            require(digest(image) == record['image_sha256'], f'Image hash mismatch: {image}')
            data = image.read_bytes()
            require(data[:8] == b'\x89PNG\r\n\x1a\n', f'Invalid PNG: {image}')
            require(struct.unpack('>II', data[16:24]) == (1920, 1080), 'Unexpected dimensions')
            images_checked += 1
            tool = record['tool']
            for field in ('shaft_reference_position_mm', 'shaft_direction_unit', 'tip_position_mm'):
                vector = tool[field]
                require(len(vector) == 3 and all(math.isfinite(x) for x in vector), f'Invalid {field}')
                scalar_count += len(vector)
            direction_norm = math.sqrt(sum(x*x for x in tool['shaft_direction_unit']))
            require(abs(direction_norm - 1) < 1e-8, f'Non-unit direction: {path.name}/{eye}')
            require(tool['shaft_reference_position_mm'][2] > 0 and tool['tip_position_mm'][2] > 0,
                    'Reference point is behind the camera')
        require(scalar_count == 18, 'Expected 18 target scalars per pair')

    checks = json.loads((dataset / 'checks.json').read_text())
    require(checks['passed'], 'Recorded capture checks failed')
    require(len(checks['opencv_projection_independent_check']) == 6, 'Missing projection checks')
    for check in checks['opencv_projection_independent_check']:
        require(check['point_projection_max_error_px'] < 1e-9 and
                check['axis_projection_max_error_px'] < 1e-9, 'Recorded projection disagreement')
    print(f"PASS: {len(inventory['files'])} source/sample hashes; syntax; module checksums; "
          f"single instrument; {images_checked} images; 3 x 18 target values; recorded projection checks.")


if __name__ == '__main__':
    main()
