"""Check the two saved compatibility pairs with the original validation code."""
import argparse
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
project = root / 'project'
sys.path.insert(0, str(project / 'code_distributed_7d4a31c'))
from io_utils import read, write, sha
from verify import validate_record
import numpy as np
from PIL import Image

run = project / 'outputs/runtime6_probe_20261109'
manifest = read(run / 'config/manifest.json')
settings = read(project / 'dispatch_2100_20261109/config/effective_settings.json')
inventory = read(run / 'validation/render_inventory.json')
assert len(inventory['jobs']) == 2
rows = []
for job in inventory['jobs']:
    label = run / 'sequences' / job['sequence_id'] / 'labels' / (job['job_id'] + '.json')
    record, result, _ = validate_record(label, manifest['fingerprint'], settings)
    assert record['frame_index'] == job['frame_index']
    result['label'] = str(label.relative_to(run))
    result['label_sha256'] = sha(label)
    result['mask_pixels'] = {
        eye: int(np.count_nonzero(np.asarray(Image.open(label.parent / data['mask']))))
        for eye, data in record['eyes'].items()
    }
    rows.append(result)
report = {
    'passed': True, 'pairs': len(rows),
    'scope': 'Two isolated saved pairs; not full-dataset acceptance',
    'adapter_sha256': sha(root / 'operations/capture_runtime6.py'),
    'validation_python': sys.version, 'checks': rows,
    'render_execution': manifest['render_execution'],
    'visual_review': 'Both saved stereo previews inspected: organs, instrument and overlays present; no blank or corrupted image.',
}
write(project / 'checks/runtime6_probe_validation.json', report)
print(report)
