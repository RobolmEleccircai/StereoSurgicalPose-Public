"""Compare the same frozen scene states, retaining both runtime identities."""
import json
import math
import sys
from pathlib import Path

old, new = [json.loads(Path(p).read_text()) for p in sys.argv[1:3]]
max_error = 0.0
numeric_count = 0

def compare(a, b, path):
    global max_error, numeric_count
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            compare(a[key], b[key], path + '/' + key)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + '/' + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert math.isfinite(a) and math.isfinite(b), path
        error = abs(a - b)
        max_error = max(max_error, error)
        numeric_count += 1
        assert error <= 1e-8, (path, a, b)
    else:
        assert a == b, (path, a, b)

for key in ('anatomy_meshes', 'moving_meshes', 'contact_surfaces', 'jaw_definitions', 'truth'):
    compare(old[key], new[key], key)
report = {
    'passed': True, 'scope': 'same 63 frozen states; numerical geometry and labels only',
    'old_runtime': old['dependencies']['isaac_version'],
    'new_runtime': new['dependencies']['isaac_version'],
    'numeric_values_checked': numeric_count, 'maximum_absolute_error': max_error,
    'does_not_assert_identical_rendered_pixels': True,
}
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
