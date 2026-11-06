"""Freeze one plan, assign whole sequences, and merge disjoint worker outputs.

The manifest is immutable during a run. There is no automatic work stealing:
an unfinished sequence cannot silently acquire a second owner.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from io_utils import (HERE, read, write, sha, canonical_hash, source_hash,
                      require_seal, frame_valid, load_machine)

CONFIG_FILES = ('manifest.json', 'effective_settings.json', 'joints.json',
                'camera_effective.json', 'camera_source.json',
                'appearance_source.json', 'dataset_split_policy.json')
RENDER_CONFIG_FILES = ('tip_regions.json', 'render_map.json', 'native_camera_readback.json')


def assignments(settings):
    """The agreed 352 / 1,748 allocation, independent of processing order."""
    ids = [r['sequence_id'] for r in settings['sequence_manifest']]
    laptop = ['seq_003', 'seq_013', 'seq_011']
    result = {
        'rtx4060': {'role': 'laptop', 'sequence_ids': laptop, 'pair_count': 352},
        'rtx4080super': {'role': 'server', 'sequence_ids': [s for s in ids if s not in laptop],
                         'pair_count': 1748},
    }
    validate_assignments(result, settings)
    return result


def validate_assignments(workers, settings):
    expected = {r['sequence_id']: r['frame_count'] for r in settings['sequence_manifest']}
    if len(expected) != 21 or sum(expected.values()) != 2100:
        raise ValueError('Distributed release requires the approved 21 sequences / 2100 pairs')
    if set(workers) not in ({'rtx4060', 'rtx4080super'}, {'rtx4080super', 'rtx5090'}):
        raise ValueError('Unexpected workers')
    seen = set()
    for name, worker in workers.items():
        ids = worker['sequence_ids']
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Duplicate sequence ownership')
        if not set(ids) <= expected.keys():
            raise ValueError('Unknown sequence in assignment')
        count = sum(expected[s] for s in ids)
        if count != worker['pair_count']:
            raise ValueError('Worker pair count differs from whole-sequence assignment')
        if worker['role'] != ('laptop' if name == 'rtx4060' else 'server'):
            raise ValueError('Worker role mismatch')
        if name == 'rtx4060' and count > 352:
            raise ValueError('Laptop allocation exceeds the authorized 352 pairs')
        seen.update(ids)
    if seen != expected.keys():
        raise ValueError('Assignments omit sequences')


def portable_dependencies(deps):
    """Compare actual resolved content, not host-specific absolute directories."""
    if deps['unresolved']:
        raise ValueError('Unresolved assets')
    hashes = {str(Path(f['path']).resolve()): f['sha256'] for f in deps['files']}
    bindings = [{'attribute': row['attribute'],
                 'content': sorted(hashes[str(Path(p).resolve())] for p in row['resolved_files'])}
                for row in deps['active_asset_attributes']]
    return {'files': sorted((f['sha256'], f['bytes']) for f in deps['files']),
            'bindings': sorted(bindings, key=lambda r: r['attribute']),
            'python_packages': deps['python_packages'], 'isaac_version': deps['isaac_version']}


def validate_plan(run, settings, fingerprint):
    run = Path(run)
    require_seal(run/'trajectories/index.json', fingerprint)
    index = read(run/'trajectories/index.json')
    expected = {r['sequence_id']: r['frame_count'] for r in settings['sequence_manifest']}
    if index['sequence_frame_counts'] != expected or index['total_pairs'] != 2100:
        raise ValueError('Frozen plan counts differ from settings')
    if len(index['sequences']) != len(expected) or set(index['sequences']) != expected.keys():
        raise ValueError('Frozen plan sequence list is incomplete or repeated')
    for sid in index['sequences']:
        path = run/'trajectories'/f'{sid}.json'
        require_seal(path, fingerprint)
        if sha(path) != index['trajectory_hashes'][sid]:
            raise ValueError('Frozen trajectory changed: '+sid)
        seq = read(path)
        if seq['fingerprint'] != fingerprint or seq['sequence_id'] != sid or len(seq['frames']) != expected[sid]:
            raise ValueError('Frozen trajectory identity/count mismatch')
        if [f['frame_index'] for f in seq['frames']] != list(range(expected[sid])):
            raise ValueError('Frozen trajectory frames are missing or repeated')
    return index


def prepare(plan_run, output):
    """Export a checked plan with no RGB and no dependency on its original paths."""
    plan_run, output = Path(plan_run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a NEW dispatch directory; running assignments are immutable')
    m = read(plan_run/'config/manifest.json'); settings = m['settings']
    if m['code_sha256'] != source_hash():
        raise ValueError('Plan belongs to a different code version; plan with this release first')
    validate_plan(plan_run, settings, m['fingerprint'])
    from dataset_policy import audit
    seqs = [read(p) for p in sorted((plan_run/'trajectories').glob('seq_*.json')) if not p.name.endswith('.sha256.json')]
    report = audit(seqs, settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name+'.partial-', dir=output.parent))
    try:
        for name in CONFIG_FILES:
            (temp/'config').mkdir(exist_ok=True)
            shutil.copy2(plan_run/'config'/name, temp/'config'/name)
        shutil.copytree(plan_run/'trajectories', temp/'trajectories')
        write(temp/'validation/planned_split_coverage.json', report)
        payload = {'schema_version': 1, 'code_sha256': source_hash(), 'fingerprint': m['fingerprint'],
                   'settings': settings, 'dependency_identity': portable_dependencies(m['dependencies']),
                   'workers': assignments(settings),
                   'files': {p.relative_to(temp).as_posix(): sha(p) for p in sorted(temp.rglob('*')) if p.is_file()}}
        payload['dispatch_id'] = canonical_hash(payload)
        write(temp/'dispatch.json', payload)
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return payload


def validate_dispatch(d, settings):
    if canonical_hash({k: v for k, v in d.items() if k != 'dispatch_id'}) != d['dispatch_id']:
        raise ValueError('Dispatch manifest changed')
    if d['code_sha256'] != source_hash() or d['settings'] != settings:
        raise ValueError('Dispatch code/configuration changed; cannot mix releases')
    validate_assignments(d['workers'], settings)
    if d.get('schema_version') == 2:
        origin = d['plan_origin']
        for digest, manifest in d['origin_manifests'].items():
            if canonical_hash(manifest) != digest or manifest['fingerprint'] != d['fingerprint']:
                raise ValueError('Original renderer provenance changed')
        if origin['fingerprint'] != d['fingerprint'] or not origin['dispatch_id']:
            raise ValueError('Redistribution changed frozen plan identity')
        counts = {r['sequence_id']: r['frame_count'] for r in settings['sequence_manifest']}
        for key, item in d['imported_frames'].items():
            sid, ident = key.split('/')
            if sid not in counts or ident != f'{int(ident):06d}' or not 0 <= int(ident) < counts[sid]:
                raise ValueError('Invalid imported frame')
            if len(item['label_sha256']) != 64 or item['source_manifest_sha256'] not in d['origin_manifests']:
                raise ValueError('Imported frame has no immutable provenance')
            source_manifest=d['origin_manifests'][item['source_manifest_sha256']]
            if item['render_worker']['dispatch_id'] != source_manifest.get('dispatch_id',origin['dispatch_id']):
                raise ValueError('Imported frame belongs to another source dispatch')


def load_dispatch(folder, settings):
    folder = Path(folder)
    d = read(folder/'dispatch.json'); validate_dispatch(d, settings)
    for name, checksum in d['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != checksum:
            raise ValueError('Changed dispatch file: '+name)
    validate_plan(folder, settings, d['fingerprint'])
    return d


def worker_context(d, worker_id, machine):
    if worker_id not in d['workers']:
        raise ValueError('Unknown worker')
    spec = d['workers'][worker_id]
    if machine.get('worker_id') != worker_id or machine['role'] != spec['role']:
        raise ValueError('Machine worker_id/role does not match the selected worker')
    return {'dispatch_id': d['dispatch_id'], 'worker_id': worker_id, **spec}


def jobs_for(index, context=None):
    ids = index['sequences'] if context is None else context['sequence_ids']
    return [{'sequence_id': sid, 'frame_index': i, 'job_id': f'{i:06d}', 'reason': 'dataset_frame'}
            for sid in ids for i in range(index['sequence_frame_counts'][sid])]


def run_context(run, machine, settings):
    d = read(Path(run)/'config/dispatch.json'); validate_dispatch(d, settings)
    manifest = read(Path(run)/'config/manifest.json')
    if manifest['fingerprint'] != d['fingerprint'] or manifest.get('dispatch_id') != d['dispatch_id']:
        raise ValueError('Run belongs to another frozen plan')
    work = read(Path(run)/'config/worker.json')
    if work != worker_context(d, work['worker_id'], machine):
        raise ValueError('Worker assignment changed')
    return work


def validate_frame_owner(label, context, dispatch):
    """Keep prior receipts byte-identical; a transfer changes scheduling, not origin."""
    record = read(label)
    if record['sequence_id'] not in context['sequence_ids']:
        raise ValueError('Frame is outside the assigned sequences')
    key = record['sequence_id'] + '/' + record['job_id']
    imported = dispatch.get('imported_frames', {}).get(key)
    if imported is not None:
        if sha(label) != imported['label_sha256'] or record.get('render_worker') != imported['render_worker']:
            raise ValueError('Transferred frame changed or lost original renderer identity')
    elif record.get('render_worker') != context:
        raise ValueError('Frame renderer ownership differs from assignment')


def install_plan(folder, run, machine, settings, d):
    """Rebind local assets while keeping common trajectory fingerprints intact."""
    from runtime import dependency_audit, source_paths
    local_deps = dependency_audit(machine, settings)
    if canonical_hash(portable_dependencies(local_deps)) != canonical_hash(d['dependency_identity']):
        raise ValueError('Worker asset content or runtime differs from the frozen plan')
    run = Path(run); marker = run/'.dispatch_preparing.json'
    existed = (run/'config/manifest.json').exists()
    if existed and not marker.exists():
        if read(run/'config/dispatch.json') != d:
            raise ValueError('Run already belongs to another dispatch')
        validate_plan(run, settings, d['fingerprint'])
        for name in CONFIG_FILES:
            if name != 'manifest.json' and sha(run/'config'/name) != d['files']['config/'+name]:
                raise ValueError('Worker configuration changed: '+name)
    else:
        # A small intent record makes interrupted preparation safely resumable.
        if marker.exists() and read(marker) != {'dispatch_id': d['dispatch_id']}:
            raise ValueError('Interrupted preparation belongs to another dispatch')
        if (run/'sequences').exists() or (not marker.exists() and
                ((run/'trajectories').exists() or (run/'config').exists())):
            raise ValueError('Unrecognized existing output; use a new run directory')
        write(marker, {'dispatch_id': d['dispatch_id']})
        for name in CONFIG_FILES:
            if name == 'manifest.json': continue
            copy_atomic(Path(folder)/'config'/name, run/'config'/name)
        for path in (Path(folder)/'trajectories').rglob('*'):
            if path.is_file():copy_atomic(path, run/'trajectories'/path.relative_to(Path(folder)/'trajectories'))
        write(run/'config/dispatch.json', d)
    manifest = read(Path(folder)/'config/manifest.json')
    manifest.update(dependencies=local_deps, machine_role=machine['role'],
                    source_paths={k: str(v) for k, v in source_paths(machine, settings).items()},
                    dispatch_id=d['dispatch_id'])
    write(run/'config/manifest.json', manifest)
    marker.unlink(missing_ok=True)


def copy_atomic(source, target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.copying')
    shutil.copy2(source, temporary)
    temporary.replace(target)


def prepare_worker(folder, run, machine, settings, worker_id):
    d = load_dispatch(folder, settings); context = worker_context(d, worker_id, machine)
    path = Path(run)/'config/worker.json'
    if path.exists() and read(path) != context:
        raise ValueError('Cannot change the owner of an existing worker run')
    install_plan(folder, run, machine, settings, d)
    write(path, context)
    return context


def validate_inventory(inv, index, context=None):
    expected = jobs_for(index, context)
    if inv['jobs'] != expected or inv['pairs'] != len(expected) or not inv['complete']:
        raise ValueError('Missing, duplicate, or unassigned render jobs')
    if context is not None and inv.get('worker') != context:
        raise ValueError('Inventory worker does not match its assignment')
    return expected


def completion_status(mode, rendered, index, worker=None):
    full = mode in ('generate', 'merge')
    expected = worker['pair_count'] if worker else len(index['validation_jobs']) if mode == 'validate' else index['total_pairs']
    if rendered != expected:
        raise ValueError('Incomplete output count')
    return {'state': 'complete' if full else 'shard_complete' if worker else 'validated',
            'validation_complete': True, 'dataset_complete': full, 'shard_complete': worker is not None,
            'rendered_pairs': rendered, 'target_pairs': expected, 'planned_pairs': index['total_pairs']}


def inspect_shards(inputs, settings):
    """Read and validate both local/transported worker runs before copying anything."""
    runs = [Path(p).resolve() for p in inputs]
    if len(runs) != 2 or len(set(runs)) != 2:
        raise ValueError('Provide two distinct worker runs')
    common = None; owners = set(); result = []
    for run in runs:
        d = read(run/'config/dispatch.json'); validate_dispatch(d, settings)
        if common is not None and d != common: raise ValueError('Worker dispatches differ')
        common = d
        context = read(run/'config/worker.json'); name = context['worker_id']
        if context != {'dispatch_id': d['dispatch_id'], 'worker_id': name, **d['workers'][name]} or name in owners:
            raise ValueError('Repeated or changed worker ownership')
        owners.add(name)
        manifest = read(run/'config/manifest.json')
        if (manifest['fingerprint'] != d['fingerprint'] or manifest['code_sha256'] != d['code_sha256']
                or manifest['settings'] != settings
                or canonical_hash(portable_dependencies(manifest['dependencies'])) != canonical_hash(d['dependency_identity'])):
            raise ValueError('Worker manifest differs from frozen input')
        for cfg in CONFIG_FILES:
            if cfg != 'manifest.json' and sha(run/'config'/cfg) != d['files']['config/'+cfg]:
                raise ValueError('Worker config differs from frozen plan: '+cfg)
        index = validate_plan(run, settings, d['fingerprint'])
        inv = read(run/'validation/render_inventory.json'); validate_inventory(inv, index, context)
        status = read(run/'status.json'); report = read(run/'validation/report.json')
        if not status.get('shard_complete') or not report['passed'] or report['rendered_pairs'] != context['pair_count']:
            raise ValueError('Worker has not completed verification')
        dirs = {p.name for p in (run/'sequences').iterdir() if p.is_dir()}
        if dirs != set(context['sequence_ids']): raise ValueError('Unassigned or missing sequence directory')
        for sid in context['sequence_ids']:
            labels = run/'sequences'/sid/'labels'
            names = {p.name for p in labels.glob('*.json')}
            if names != {f'{i:06d}.json' for i in range(index['sequence_frame_counts'][sid])}:
                raise ValueError('Unassigned or missing frame files')
            for label in labels.glob('*.json'):
                if not frame_valid(label, d['fingerprint']): raise ValueError('Corrupt worker frame: '+str(label))
                validate_frame_owner(label, context, d)
        result.append((run, context))
    if owners != common['workers'].keys(): raise ValueError('Missing worker')
    for name in RENDER_CONFIG_FILES:
        if read(runs[0]/'config'/name) != read(runs[1]/'config'/name):
            raise ValueError('Render configuration differs between workers: '+name)
    return common, result


def merge(inputs, run, machine, settings):
    d, sources = inspect_shards(inputs, settings)
    run = Path(run)
    if (run/'config').exists() or (run/'sequences').exists():
        raise ValueError('Merge needs a new output run; source shards remain unchanged')
    # Both workers contain the entire frozen plan. Only their render jobs differ.
    install_plan(sources[0][0], run, machine, settings, d)
    for name in RENDER_CONFIG_FILES:
        shutil.copy2(sources[0][0]/'config'/name, run/'config'/name)
    (run/'previews').mkdir(exist_ok=True)
    for source, context in sources:
        for definition in (source/'config/definition_versions').glob('*.json'):
            if definition.stem != sha(definition):raise ValueError('Corrupt archived definition')
            copy_atomic(definition,run/'config/definition_versions'/definition.name)
        name = context['worker_id']
        write(run/'config/workers'/f'{name}.json', read(source/'config/manifest.json'))
        for sid in context['sequence_ids']:
            shutil.copytree(source/'sequences'/sid, run/'sequences'/sid)
            for p in (source/'previews').glob(sid+'_*'):
                if p.is_file(): shutil.copy2(p, run/'previews'/p.name)
    for digest, original in d.get('origin_manifests', {}).items():
        write(run/'config/original_workers'/f'{digest}.json', original)
    index = read(run/'trajectories/index.json'); jobs = jobs_for(index)
    write(run/'validation/render_inventory.json', {'mode': 'merge', 'fingerprint': d['fingerprint'],
          'jobs': jobs, 'pairs': len(jobs), 'complete': True, 'dispatch_id': d['dispatch_id']})
    write(run/'config/merge_sources.json', [{'worker': ctx, 'source_run': str(src)} for src, ctx in sources])
    write(run/'status.json', {'state': 'merged_pending_verification', 'dataset_complete': False})
    # verify.py performs the existing full-dataset checks before marking complete.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--plan-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('worker')
    for flag in ('machine', 'settings', 'run', 'dispatch'): p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--worker', required=True)
    p = commands.add_parser('merge')
    for flag in ('machine', 'settings', 'run'): p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    a = parser.parse_args()
    if a.action == 'prepare':
        d = prepare(a.plan_run, a.output)
        print('DISPATCH_PREPARED', d['dispatch_id'], d['workers'], 'NO_RENDER')
    elif a.action == 'worker':
        context = prepare_worker(a.dispatch, a.run, load_machine(a.machine), read(a.settings), a.worker)
        print('WORKER_PREPARED', context, 'NO_RENDER')
    else:
        merge(a.inputs, a.run, load_machine(a.machine), read(a.settings))
        print('MERGED_PENDING_VERIFICATION', a.run, 'NO_RENDER')


if __name__ == '__main__':
    main()
