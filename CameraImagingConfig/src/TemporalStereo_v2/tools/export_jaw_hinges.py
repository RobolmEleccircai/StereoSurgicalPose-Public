"""Read physical USD jaw-joint anchors for saved poses; no SimulationApp/rendering."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--pose-geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    machine = load_machine(args.machine)
    if not args.worker:
        isaac = Path(machine['isaac_root'])
        libraries = list((isaac/'extscache').glob('omni.usd.libs-*'))
        if len(libraries) != 1:
            raise ValueError('Expected one USD runtime')
        env = os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[name] = '1'
        env['PYTHONPATH'] = os.pathsep.join([str(libraries[0]), machine['deps_root'], str(ROOT), env.get('PYTHONPATH', '')])
        env['LD_LIBRARY_PATH'] = os.pathsep.join([str(libraries[0]/'bin'), env.get('LD_LIBRARY_PATH', '')])
        subprocess.run([str(isaac/'python.sh'), str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], env=env, check=True)
        return

    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from kinematics import joint_frame, rigid_mm, usd_world

    settings = read(args.settings)
    source = Path(machine['asset_root'])/settings['source_scene']
    geometry = read(args.pose_geometry)
    source_hash, geometry_hash = sha(source), sha(args.pose_geometry)
    if source_hash != geometry['source_scene_sha256']:
        raise ValueError('USD source differs from the source of saved body poses')
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError('Cannot open source USD')
    if abs(UsdGeom.GetStageMetersPerUnit(stage)-.01)>1e-12:
        raise ValueError('Expected the validated centimeter USD stage')
    definitions, dependencies = {}, {str(source): source_hash}
    for index in (1, 2):
        name = f'psm_tool_gripper{index}_joint'
        matches = [p for p in stage.Traverse() if p.GetName() == name and str(p.GetPath()).startswith(settings['psm_root']+'/')]
        if len(matches) != 1 or not matches[0].IsA(UsdPhysics.RevoluteJoint):
            raise ValueError(f'Expected one real revolute joint: {name}')
        prim = matches[0]
        item = {'joint_path': str(prim.GetPath()), 'joint_name': name,
                'axis': prim.GetAttribute('physics:axis').Get(),
                'jaw_body': None, 'parent_body': None}
        for side in (0, 1):
            targets = prim.GetRelationship(f'physics:body{side}').GetTargets()
            if len(targets) != 1:
                raise ValueError('Joint body relationship is missing or ambiguous')
            body = str(targets[0])
            raw_body = usd_world(stage, body)
            world_body = rigid_mm(raw_body)
            # Account for the body's actual USD scale before converting to mm.
            world_joint = rigid_mm(raw_body@joint_frame(prim, side))
            local = np.linalg.solve(world_body, world_joint)
            item[f'T_body{side}_from_joint{side}_mm'] = local.tolist()
            item['jaw_body' if side else 'parent_body'] = body
            item[f'authored_localPos{side}'] = list(prim.GetAttribute(f'physics:localPos{side}').Get())
        if not item['jaw_body'].endswith(f'/psm_tool_gripper{index}_link'):
            raise ValueError('Physical jaw identity does not match the joint')
        for spec in prim.GetPrimStack():
            layer_path = spec.layer.realPath
            if layer_path and Path(layer_path).is_file():
                dependencies[layer_path] = sha(Path(layer_path))
        definitions[f'jaw{index}'] = item

    maximum_gap, maximum_axis_error, maximum_hinge_separation = 0., 0., 0.
    for frame in geometry['frames']:
        anchors = []
        for definition in definitions.values():
            poses = frame['T_world_from_body_mm']
            parent = np.array(poses[definition['parent_body']])@np.array(definition['T_body0_from_joint0_mm'])
            child = np.array(poses[definition['jaw_body']])@np.array(definition['T_body1_from_joint1_mm'])
            gap = float(np.linalg.norm(parent[:3, 3]-child[:3, 3]))
            axis_index = 'XYZ'.index(definition['axis'])
            axis_error = float(np.linalg.norm(parent[:3, axis_index]-child[:3, axis_index]))
            maximum_gap = max(maximum_gap, gap)
            maximum_axis_error = max(maximum_axis_error, axis_error)
            if gap > 1e-4 or axis_error > 1e-5:
                raise ValueError('Saved jaw pose does not close at the authored USD hinge')
            anchors.append(child[:3, 3])
        maximum_hinge_separation = max(maximum_hinge_separation, float(np.linalg.norm(anchors[0]-anchors[1])))
    if sha(source) != source_hash or sha(args.pose_geometry) != geometry_hash:
        raise RuntimeError('Source USD or saved body poses changed')
    for filename, expected in dependencies.items():
        if sha(Path(filename)) != expected:
            raise RuntimeError('USD joint definition changed during export')
    result = {'schema_version': 1, 'unit': 'mm', 'source': 'authored_USD_physics_localPos0_localPos1',
              'exporter_sha256': sha(Path(__file__)),
              'pose_geometry_sha256': geometry_hash, 'source_scene_sha256': source_hash,
              'source_layers_sha256': dependencies, 'jaws': definitions,
              'axis_orientation': 'native_jaw_body_XYZ_translated_to_physical_hinge',
              'frames_verified': len(geometry['frames']), 'max_parent_child_anchor_gap_mm': maximum_gap,
              'max_revolute_axis_direction_difference': maximum_axis_error,
              'max_H1_H2_separation_mm': maximum_hinge_separation,
              'source_files_unchanged': True, 'simulation_app_started': False, 'new_rgb_renders': 0}
    write(args.output, result)
    print(json.dumps({k: result[k] for k in ('frames_verified', 'max_parent_child_anchor_gap_mm', 'max_H1_H2_separation_mm', 'new_rgb_renders')}))


if __name__ == '__main__':
    main()
