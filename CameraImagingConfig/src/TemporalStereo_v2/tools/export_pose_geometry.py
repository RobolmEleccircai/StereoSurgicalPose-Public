"""Read saved USD geometry and joint truth without creating a SimulationApp."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import load_machine, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--machine', type=Path, required=True)
    p.add_argument('--settings', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    m = load_machine(a.machine)
    if not a.worker:
        isaac = Path(m['isaac_root'])
        usd = list((isaac/'extscache').glob('omni.usd.libs-*'))
        if len(usd) != 1:
            raise ValueError('Expected one USD runtime')
        env = os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[name] = '1'
        env['PYTHONPATH'] = os.pathsep.join([str(usd[0]), m['deps_root'], str(ROOT), env.get('PYTHONPATH', '')])
        env['LD_LIBRARY_PATH'] = os.pathsep.join([str(usd[0]/'bin'), env.get('LD_LIBRARY_PATH', '')])
        subprocess.run([str(isaac/'python.sh'), str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], env=env, check=True)
        return
    import numpy as np
    from pxr import UsdGeom
    from runtime import Scene
    from kinematics import mesh_triangles, usd_world
    c = read(a.settings)
    scene = Scene(m, c)
    before = sha(scene.paths['source_scene'])
    bind = scene.psm.fk(scene.psm.base)
    meshes, arrays = [], {}
    for prim in scene.stage.Traverse():
        name = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or not name.startswith(scene.psm.root+'/'):
            continue
        if '/visuals' not in name or 'collision' in name.lower() or UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            continue
        body = next((b for b in bind if name.startswith(b+'/')), None)
        if body is None:
            raise ValueError('Unattached visual mesh: '+name)
        vertices, faces = mesh_triangles(prim)
        world = (usd_world(scene.stage, name) @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]*10
        T = bind[body]
        local = (world-T[:3, 3]) @ T[:3, :3]
        key = f'mesh_{len(meshes):03d}'
        arrays[key+'_vertices'] = local
        arrays[key+'_faces'] = faces
        meshes.append({'key': key, 'path': name, 'body': body, 'vertices': len(vertices), 'triangles': len(faces)})
    if not meshes:
        raise ValueError('No actual visible instrument meshes')
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output/'meshes.npz', **arrays)
    frames = []
    for path in sorted(a.run.glob('sequences/*/labels/*.json')):
        label = read(path)
        q = np.array([label['joint_positions']['psm'][n] for n in scene.psm.names])
        poses = scene.psm.fk(q)
        C, d, tips = scene.points(q)
        truth = read(path.parent/label['camera_truth'])
        for eye in ('left', 'right'):
            camera = np.linalg.inv(truth['T_world_from_optical_mm'][eye])
            camera_C = (camera @ np.r_[C, 1])[:3]
            camera_T = (camera @ np.r_[tips.mean(0), 1])[:3]
            expected = label['eyes'][eye]['tool']
            if not np.allclose(camera_C, expected['shaft_reference_position_mm'], atol=1e-8, rtol=0) or not np.allclose(camera_T, expected['tip_position_mm'], atol=1e-8, rtol=0):
                raise ValueError('Saved pose and USD articulation disagree')
        frames.append({'sequence_id': label['sequence_id'], 'job_id': label['job_id'],
                       'label_sha256': sha(path), 'T_world_from_body_mm': {k: v.tolist() for k, v in poses.items()},
                       'roll_body': scene.roll, 'camera_truth': truth})
    if before != sha(scene.paths['source_scene']):
        raise ValueError('Source USD changed')
    write(a.output/'geometry.json', {'unit': 'mm', 'source': 'official_visual_meshes_and_saved_joint_truth',
          'source_scene_sha256': before, 'meshes_sha256': sha(a.output/'meshes.npz'), 'meshes': meshes,
          'frames': frames, 'simulation_app_started': False, 'new_rgb_renders': 0})
    print(json.dumps({'mesh_count': len(meshes), 'triangles': sum(m['triangles'] for m in meshes), 'frames': len(frames), 'new_rgb_renders': 0}), flush=True)


if __name__ == '__main__':
    main()
