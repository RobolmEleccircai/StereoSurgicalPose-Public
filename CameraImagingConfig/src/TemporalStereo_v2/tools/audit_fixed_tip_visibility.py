"""Audit saved endpoint visibility against independent triangle intersections.

Reads existing images, labels and USD geometry. Never starts SimulationApp,
changes a production label, or renders a new scene image.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import load_machine, read, write, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('machine', 'settings', 'run', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--reference-sample', type=Path)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    if a.output.resolve().is_relative_to(a.run.resolve()):
        raise ValueError('Audit output must be outside the original run')
    m = load_machine(a.machine)
    if not a.worker:
        isaac = Path(m['isaac_root'])
        usd, = (isaac/'extscache').glob('omni.usd.libs-*')
        env = os.environ.copy()
        for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = '1'
        env['PYTHONPATH'] = os.pathsep.join([str(usd), m['deps_root'], str(ROOT)])
        env['LD_LIBRARY_PATH'] = os.pathsep.join([str(usd/'bin'), env.get('LD_LIBRARY_PATH', '')])
        subprocess.run([str(isaac/'python.sh'), str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                       env=env, check=True)
        return

    import numpy as np
    import trimesh
    from PIL import Image
    from pxr import UsdGeom
    from runtime import Scene
    from kinematics import mesh_triangles, usd_world

    if a.output.exists():
        raise ValueError('Use a new audit directory')
    a.output.mkdir(parents=True)
    c = read(a.settings)
    scene = Scene(m, c)
    paths = sorted(a.run.glob('sequences/*/labels/*.json'))
    if not paths:
        raise ValueError('No saved frames')
    original = {str(f):sha(f) for f in (a.run/'sequences').rglob('*') if f.is_file()}
    asset_hash = sha(scene.paths['source_scene'])
    bind = scene.psm.fk(scene.psm.base)
    meshes = []
    for prim in scene.stage.Traverse():
        name = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            continue
        tissue = name.startswith(c['anatomy_root']+'/')
        tool = name.startswith(scene.psm.root+'/') and '/visuals' in name and 'collision' not in name.lower()
        if not (tissue or tool):
            continue
        vertices, faces = mesh_triangles(prim)
        vertices = (usd_world(scene.stage, name)@np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]*10
        body = next((b for b in bind if name.startswith(b+'/')), None) if tool else None
        if tool and body is None:
            raise ValueError('Unattached tool surface')
        if tool:
            T = bind[body]
            vertices = (vertices-T[:3, 3])@T[:3, :3]
        triangles = vertices[faces]
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume < 0:
            normals *= -1
        meshes.append(dict(name=name, body=body, kind='tissue' if tissue else 'instrument',
                           vertices=vertices, triangles=triangles, normals=normals,
                           bounds=np.array([vertices.min(0), vertices.max(0)]),
                           v0=triangles[:, 0], e1=triangles[:, 1]-triangles[:, 0],
                           e2=triangles[:, 2]-triangles[:, 0]))
    if not any(x['kind']=='tissue' for x in meshes) or not any(x['body'] for x in meshes):
        raise ValueError('Empty geometry inventory')

    def intersections(mesh, origin, direction, length):
        # Independent, two-sided Moller-Trumbore implementation. No trimesh
        # ray API or production Safety.visible / classify_rays is used.
        low, high = 0., length+1e-4
        for axis in range(3):
            if abs(direction[axis]) < 1e-14:
                if not mesh['bounds'][0, axis]-1e-6 <= origin[axis] <= mesh['bounds'][1, axis]+1e-6:
                    return np.array([]), np.array([], dtype=int)
            else:
                interval = (mesh['bounds'][:, axis]-origin[axis])/direction[axis]
                low, high = max(low, interval.min()), min(high, interval.max())
        if high < low-1e-5:
            return np.array([]), np.array([], dtype=int)
        h = np.cross(direction, mesh['e2'])
        det = np.einsum('ij,ij->i', mesh['e1'], h)
        valid = abs(det) > 1e-12
        inverse = np.zeros(len(det)); inverse[valid] = 1/det[valid]
        s = origin-mesh['v0']
        u = inverse*np.einsum('ij,ij->i', s, h)
        q = np.cross(s, mesh['e1'])
        v = inverse*(q@direction)
        t = inverse*np.einsum('ij,ij->i', mesh['e2'], q)
        keep = valid & (u>=-1e-8) & (v>=-1e-8) & (u+v<=1+1e-8) & (t>1e-5) & (t<=length+1e-4)
        return t[keep], np.where(keep)[0]

    tri = np.array([[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]])
    synthetic = dict(bounds=np.array([tri.min(0),tri.max(0)]),v0=tri[0:1],
                     e1=(tri[1]-tri[0])[None],e2=(tri[2]-tri[0])[None])
    for origin,direction,length,expected in [([0,0,0],[0,0,1],3,[2.]),
            ([0,0,4],[0,0,-1],3,[2.]),([3,0,0],[0,0,1],3,[]),
            ([0,0,0],[0,0,1],1,[]),([0,-1,0],[0,0,1],3,[2.])]:
        actual,_ = intersections(synthetic,np.asarray(origin),np.asarray(direction),length)
        if len(actual)!=len(expected) or not np.allclose(actual,expected,atol=1e-10,rtol=0):
            raise ValueError('Independent intersection self-check failed')

    definitions = {}
    for jaw, (body, point) in enumerate(scene.refs, 1):
        own, = [x for x in meshes if x['body']==body]
        nearest = trimesh.triangles.closest_point(own['triangles'], np.repeat(point[None], len(own['triangles']), 0))
        distances = np.linalg.norm(nearest-point, axis=1)
        # Some authored triangles have repeated vertices. They cannot define a
        # surface normal; exclude non-finite nearest-point results explicitly.
        finite = np.isfinite(distances)
        if not finite.any():
            raise ValueError('No finite jaw surface distance')
        minimum_distance = float(distances[finite].min())
        faces = np.where(finite & (distances < max(1e-7, minimum_distance+1e-9)))[0]
        definitions[jaw] = dict(body=body, mesh=own['name'], local_point_mm=point.tolist(),
            distance_to_surface_mm=minimum_distance,
            nonfinite_degenerate_nearest_triangles=int((~finite).sum()),
            behind_distal_Y_extremum_mm=float(own['vertices'][:, 1].max()-point[1]),
            incident_face_indices=faces.tolist(), local_face_normals=own['normals'][faces].tolist())

    reference_records = []
    if a.reference_sample:
        reference = read(a.reference_sample)
        if reference['source_scene_sha256'] != asset_hash:
            raise ValueError('Reference sample uses a different source mesh')
        for obs in reference['observations']:
            poses = {k:np.asarray(v) for k,v in reference['poses'].items()}
            origin = np.asarray(obs['camera_origin_world_mm']);target = np.asarray(obs['point_world_mm'])
            delta=target-origin;length=np.linalg.norm(delta);direction=delta/length;hits=[]
            for mesh in meshes:
                T=poses[mesh['body']] if mesh['body'] else np.eye(4)
                t,_=intersections(mesh,(origin-T[:3,3])@T[:3,:3],direction@T[:3,:3],length)
                if len(t) and t.min()<length-.05:
                    hits.append(dict(mesh=mesh['name'],distance_mm=float(t.min())))
            T=poses[definitions[obs['jaw']]['body']]
            dots=np.asarray(definitions[obs['jaw']]['local_face_normals'])@((-direction)@T[:3,:3])
            reference_records.append(dict(eye=obs['eye'],jaw=obs['jaw'],independent_visible=not hits,
                expected_visible=obs['expected_visible'],normal_dot_toward_camera=dots.tolist(),blockers=hits))
        if any(r['independent_visible']!=r['expected_visible'] for r in reference_records):
            raise ValueError('Existing positive reference visibility did not reproduce')
        write(a.output/'previous_visible_reference.json',dict(source=reference['source'],
            reference_input_sha256=sha(a.reference_sample),records=reference_records))

    records = []
    for path in paths:
        label = read(path)
        truth_path = (path.parent/label['camera_truth']).resolve()
        if sha(truth_path) != label['camera_truth_sha256']:
            raise ValueError('Changed camera truth')
        cameras = read(truth_path)['T_world_from_optical_mm']
        poses = {k:np.asarray(v) for k,v in label['body_pose_truth']['T_world_from_body_mm'].items()}
        for eye, data in label['eyes'].items():
            for field in ('image', 'mask'):
                if sha((path.parent/data[field]).resolve()) != data[field+'_sha256']:
                    raise ValueError('Saved image or mask changed')
            mask = np.asarray(Image.open((path.parent/data['mask']).resolve()))>0
            camera = np.asarray(cameras[eye]); origin = camera[:3, 3]
            for jaw in (1, 2):
                target_cam = np.asarray(data['tool'][f'jaw{jaw}_tip_position_mm'])
                target = (camera@np.r_[target_cam, 1])[:3]
                delta = target-origin; length = np.linalg.norm(delta); direction = delta/length
                body = definitions[jaw]['body']; pose = poses[body]
                local = (target-pose[:3, 3])@pose[:3, :3]
                normal_dots = np.asarray(definitions[jaw]['local_face_normals'])@((-direction)@pose[:3, :3])
                hits = []; own_surface = []
                for mesh in meshes:
                    T = poses[mesh['body']] if mesh['body'] else np.eye(4)
                    o = (origin-T[:3, 3])@T[:3, :3]; d = direction@T[:3, :3]
                    distances, faces = intersections(mesh, o, d, length)
                    if len(distances):
                        i = int(np.argmin(distances))
                        hits.append(dict(distance_mm=float(distances[i]), mesh=mesh['name'],
                                         kind=mesh['kind'], body=mesh['body'], face=int(faces[i])))
                        if mesh['body']==body:
                            own_surface = [dict(distance_from_target_mm=float(abs(t-length)), face=int(f))
                                           for t,f in zip(distances,faces) if abs(t-length)<1e-4]
                hits.sort(key=lambda x:x['distance_mm'])
                blockers = [h for h in hits if h['distance_mm']<length-.05]
                first = blockers[0] if blockers else None
                saved = data['fixed_tip_visibility'][jaw-1]
                classification = 'visible' if first is None else 'tissue' if first['kind']=='tissue' else 'same_jaw' if first['body']==body else 'other_instrument'
                pixel = data['tool']['jaw_tip_pixels'][jaw-1]
                x,y = np.rint(pixel).astype(int)
                record = dict(sequence_id=label['sequence_id'], job_id=label['job_id'], eye=eye, jaw=jaw,
                    category=label['category'], phase=label['phase'], jaw_half_angle_deg=label['jaw_half_angle_deg'],
                    classification=classification, saved_visibility=saved,
                    visible_decision_matches=(first is None)==saved['visible'],
                    blocker_name_matches=(first['mesh'] if first else None)==saved['occluder'],
                    closest_blocker=first, target_camera_distance_mm=float(length),
                    blocker_to_target_gap_mm=float(length-first['distance_mm']) if first else None,
                    normal_dot_toward_camera=normal_dots.tolist(),
                    fixed_point_local_mm=local.tolist(), fixed_local_difference_mm=float(np.linalg.norm(local-definitions[jaw]['local_point_mm'])),
                    target_on_ray_surface_hits=own_surface, pixel=pixel,
                    pixel_inside_instrument_mask=bool(0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x]),
                    tip_region_visible=data['jaw_tip_region_visibility'][jaw-1]['region_visible'],
                    tolerance_sensitivity={str(tol):not any(h['distance_mm']<length-tol for h in hits) for tol in (.0001,.01,.05,.1,.2)},
                    all_mesh_first_hits=hits)
                records.append(record)
        print('AUDITED',label['sequence_id'],label['job_id'],flush=True)

    planned = []
    index = read(a.run/'trajectories/index.json')
    for sid in index['sequences']:
        seq = read(a.run/'trajectories'/f'{sid}.json'); signs = Counter(); minimum = 1.; maximum = -1.
        for frame in seq['frames']:
            poses = scene.psm.fk(frame['q_psm']); cameras = scene.cameras(frame['q_ecm'],seq)
            for camera in cameras.values():
                for definition in definitions.values():
                    T = poses[definition['body']]; origin = (camera[:3,3]-T[:3,3])@T[:3,:3]
                    direction = origin-np.asarray(definition['local_point_mm']); direction /= np.linalg.norm(direction)
                    dots = np.asarray(definition['local_face_normals'])@direction
                    signs['back_facing' if dots.max()<0 else 'front_facing' if dots.min()>0 else 'mixed_edge'] += 1
                    minimum=min(minimum,float(dots.min()));maximum=max(maximum,float(dots.max()))
        planned.append(dict(sequence_id=sid,split=seq['split'],category=seq['category'],observations=sum(signs.values()),
                            counts=dict(signs),normal_dot_range=[minimum,maximum]))

    after = {str(f):sha(f) for f in (a.run/'sequences').rglob('*') if f.is_file()}
    if original != after or asset_hash != sha(scene.paths['source_scene']):
        raise ValueError('Source data was modified')
    if any(name.startswith(('isaacsim','omni')) for name in sys.modules):
        raise ValueError('Unexpected simulator application import')
    counts = Counter(r['classification'] for r in records)
    summary = dict(existing_pairs=len(paths), endpoint_observations=len(records), new_rgb_renders=0,
        synthetic_ray_checks_passed=5,previous_visible_reference_checks=len(reference_records),
        source_files_unchanged=len(original), source_scene_sha256=asset_hash,
        mesh_count=len(meshes), anatomy_mesh_count=sum(x['kind']=='tissue' for x in meshes),
        instrument_mesh_count=sum(x['kind']=='instrument' for x in meshes), counts=dict(counts),
        visibility_disagreements=sum(not r['visible_decision_matches'] for r in records),
        blocker_name_disagreements=sum(not r['blocker_name_matches'] for r in records),
        back_facing_observations=sum(max(r['normal_dot_toward_camera'])<0 for r in records),
        front_facing_observations=sum(min(r['normal_dot_toward_camera'])>0 for r in records),
        fixed_local_max_difference_mm=max(r['fixed_local_difference_mm'] for r in records),
        directly_visible_region_count=sum(r['tip_region_visible'] for r in records),
        point_surface_hit_confirmations=sum(bool(r['target_on_ray_surface_hits']) for r in records),
        tool_gap_range_mm=[min(r['blocker_to_target_gap_mm'] for r in records if r['classification'] in ('same_jaw','other_instrument')),
                           max(r['blocker_to_target_gap_mm'] for r in records if r['classification'] in ('same_jaw','other_instrument'))],
        audit_code_sha256=sha(Path(__file__)),
        method='Independent two-sided NumPy ray-triangle intersections; raw USD visual meshes and saved body/camera transforms; no production visibility function called',
        planned_normal_scope='Surface orientation only, not full visibility or rendered-image verification')
    write(a.output/'summary.json',summary)
    write(a.output/'point_definitions.json',definitions)
    write(a.output/'observations.json',records)
    write(a.output/'planned_surface_orientations.json',planned)
    print(summary,flush=True)


if __name__ == '__main__':
    main()
