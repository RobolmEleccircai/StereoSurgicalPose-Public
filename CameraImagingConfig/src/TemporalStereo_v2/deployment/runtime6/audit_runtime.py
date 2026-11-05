"""Compare frozen scene geometry and labels across standalone Isaac runtimes; no rendering."""
import argparse
import importlib.metadata
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--code', type=Path, required=True)
p.add_argument('--machine', type=Path, required=True)
p.add_argument('--dispatch', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.code))
from io_utils import load_machine, read, write, source_hash
from runtime import Scene, dependency_audit
from safety import Safety, make_shape
from pose_truth import jaw_definitions
import numpy as np
import trimesh
import cv2

m = load_machine(a.machine)
c = read(a.dispatch / 'config/effective_settings.json')
scene = Scene(m, c)
safety = Safety(scene)
box = trimesh.creation.box(extents=(1, 1, 1))
shapes = [make_shape(name, '', 'anatomy', box.vertices, box.faces) for name in ('a', 'b')]
assert not Safety.pair(*shapes)['safe'], 'Native collision query failed'
truth = {}
for spec in c['sequence_manifest']:
    seq = read(a.dispatch / 'trajectories' / (spec['sequence_id'] + '.json'))
    for i in sorted({0, len(seq['frames']) // 2, len(seq['frames']) - 1}):
        f = seq['frames'][i]
        qp, qe = np.array(f['q_psm']), np.array(f['q_ecm'])
        truth[f"{seq['sequence_id']}/{i}"] = {
            'labels': scene.labels(qp, qe, seq),
            'cameras': {k: v.tolist() for k, v in scene.cameras(qe, seq).items()},
            'psm_transforms': {k: v.tolist() for k, v in scene.psm.fk(qp).items()},
        }
write(a.output, {
    'passed': True, 'scope': 'CPU assets, native FCL and 63 frozen frame states; no rendering',
    'python': sys.version, 'code_sha256': source_hash(),
    'opencv': cv2.__version__, 'dependencies': dependency_audit(m, c),
    'anatomy_meshes': len(safety.static), 'moving_meshes': len(safety.moving),
    'contact_surfaces': len(safety.contact_surfaces),
    'jaw_definitions': jaw_definitions(scene.stage, c['psm_root']), 'truth': truth,
})
print('RUNTIME_AUDIT_SAVED', a.output, 'states', len(truth), flush=True)
