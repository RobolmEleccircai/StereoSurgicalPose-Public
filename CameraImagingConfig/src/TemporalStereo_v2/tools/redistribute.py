"""Create a new two-server assignment, preserving frozen geometry and old receipts.

Use stopped source snapshots. No renderer runs here, and sources are never edited.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dispatch as ds
from io_utils import HERE, read, write, sha, canonical_hash, source_hash, frame_valid, resolve_tip_definition

ALLOWED_SOURCE_CHANGES = {'capture.py', 'dispatch.py', 'generate.py', 'io_utils.py', 'verify.py'}


def source_inventory(root):
    root = Path(root)
    files = list(root.glob('*.py')) + [root/'tools/reannotate.py'] + list((root/'pose_display').glob('*.py'))
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def choose_workers(settings, imports, seconds_4080, seconds_5090, pinned, pinned_5090=()):
    if min(seconds_4080, seconds_5090) <= 0:
        raise ValueError('Invalid measured throughput')
    counts = {r['sequence_id']: r['frame_count'] for r in settings['sequence_manifest']}
    remaining = {sid: n - sum(k.startswith(sid+'/') for k in imports) for sid, n in counts.items()}
    pinned = set(pinned)
    pinned_5090 = set(pinned_5090)
    if pinned & pinned_5090:
        raise ValueError('A sequence cannot be pinned to both workers')
    if not (pinned | pinned_5090) <= counts.keys():
        raise ValueError('Unknown pinned sequence')
    fixed = sum(remaining[s] for s in pinned)
    choices = {0: []}
    for sid, n in remaining.items():
        if sid in pinned or sid in pinned_5090:
            continue
        for total, ids in list(choices.items()):
            choices.setdefault(total+n, ids+[sid])
    total = sum(remaining.values())
    feasible = [(max((fixed+n)*seconds_4080, (total-fixed-n)*seconds_5090), ids)
                for n, ids in choices.items() if 0 < len(ids)+len(pinned) < len(counts)]
    _, extra = min(feasible, key=lambda item: (item[0], len(item[1]), item[1]))
    left = pinned | set(extra)
    result = {}
    for worker, ids in [('rtx4080super', [s for s in counts if s in left]),
                        ('rtx5090', [s for s in counts if s not in left])]:
        result[worker] = {'role': 'server', 'sequence_ids': ids,
                          'pair_count': sum(counts[s] for s in ids)}
    ds.validate_assignments(result, settings)
    return result, remaining


def create(folder, output, source_code, source_runs, dependencies, seconds_4080, seconds_5090, pinned, pinned_5090=()):
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise ValueError('Use a new dispatch directory')
    old = read(folder/'dispatch.json')
    if canonical_hash({k:v for k,v in old.items() if k!='dispatch_id'}) != old['dispatch_id']:
        raise ValueError('Source dispatch changed')
    for name, digest in old['files'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != digest:
            raise ValueError('Source plan file changed: '+name)
    before, after = source_inventory(source_code), source_inventory(HERE)
    if canonical_hash(before) != old['code_sha256'] or before.keys() != after.keys():
        raise ValueError('Original code identity does not match frozen plan')
    changed = {k for k in before if before[k] != after[k]}
    if not changed <= ALLOWED_SOURCE_CHANGES:
        raise ValueError('Redistribution may not change rendering/geometry modules: '+str(changed))
    settings, fingerprint = old['settings'], old['fingerprint']
    ds.validate_plan(folder, settings, fingerprint)
    from dataset_policy import audit
    audit([read(folder/'trajectories'/f'{r["sequence_id"]}.json') for r in settings['sequence_manifest']], settings)
    imports, origins = {}, dict(old.get('origin_manifests', {}))
    for source in map(Path, source_runs):
        marker = read(source/'paused_for_redistribution.json')
        if marker.get('renderer_stopped') is not True:
            raise ValueError('Source renderer has not stopped')
        original = read(source/'config/manifest.json')
        if original['fingerprint'] != fingerprint or original['code_sha256'] != old['code_sha256']:
            raise ValueError('Source output belongs to another release')
        digest = canonical_hash(original)
        origins[digest] = original
        labels = list((source/'sequences').glob('*/labels/*.json'))
        if marker['committed_pairs'] != len(labels):
            raise ValueError('Stopped snapshot changed after inspection')
        for label in labels:
            if not frame_valid(label, fingerprint):
                raise ValueError('Source receipt incomplete/corrupt: '+str(label))
            record = read(label)
            worker = record['render_worker']
            expected = {'dispatch_id': old['dispatch_id'], 'worker_id': worker['worker_id'], **old['workers'].get(worker['worker_id'],{})}
            key = record['sequence_id']+'/'+record['job_id']
            prior=old.get('imported_frames',{}).get(key)
            if prior is not None:
                if sha(label)!=prior['label_sha256'] or worker!=prior['render_worker']:
                    raise ValueError('Prior import provenance changed')
            elif worker != expected or record['sequence_id'] not in worker['sequence_ids']:
                raise ValueError('Source ownership mismatch')
            if key in imports:
                raise ValueError('Duplicate prior frame: '+key)
            imports[key] = {'label_sha256': sha(label), 'render_worker': worker,
                            'source_manifest_sha256': prior['source_manifest_sha256'] if prior else digest}
    workers, remaining = choose_workers(settings, imports, seconds_4080, seconds_5090, pinned, pinned_5090)
    output.mkdir(parents=True)
    for name in ds.CONFIG_FILES:
        ds.copy_atomic(folder/'config'/name, output/'config'/name)
    shutil.copytree(folder/'trajectories', output/'trajectories')
    manifest = read(output/'config/manifest.json')
    manifest.update(code_sha256=source_hash(), dependencies=dependencies,
                    planned_code_sha256=old['code_sha256'],
                    fingerprint_purpose='unchanged original frozen physical plan; current execution code is recorded separately')
    write(output/'config/manifest.json', manifest)
    report = {'changed_execution_files': sorted(changed), 'trajectory_bytes_unchanged': True,
              'old_code_sha256': old['code_sha256'], 'execution_code_sha256': source_hash(),
              'imported_pairs': len(imports), 'remaining_by_sequence': remaining,
              'measured_seconds_per_pair': {'rtx4080super': seconds_4080, 'rtx5090': seconds_5090},
              'workers': workers, 'sources_stopped': True}
    write(output/'validation/redistribution.json', report)
    payload = {'schema_version': 2, 'code_sha256': source_hash(), 'fingerprint': fingerprint,
               'settings': settings, 'dependency_identity': ds.portable_dependencies(dependencies),
               'workers': workers, 'plan_origin': {'dispatch_id': old['dispatch_id'],
                    'fingerprint': fingerprint, 'code_sha256': old['code_sha256']},
               'origin_manifests': origins, 'imported_frames': imports,
               'files': {p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    payload['dispatch_id'] = canonical_hash(payload)
    write(output/'dispatch.json', payload)
    ds.load_dispatch(output, settings)
    return payload


def import_frames(source_runs, run, worker_id):
    run = Path(run)
    dispatch = read(run/'config/dispatch.json')
    context = {'dispatch_id': dispatch['dispatch_id'], 'worker_id':worker_id, **dispatch['workers'][worker_id]}
    if read(run/'config/worker.json') != context:
        raise ValueError('Destination worker mismatch')
    wanted = {key:item for key,item in dispatch['imported_frames'].items() if key.split('/')[0] in context['sequence_ids']}
    found = set()
    for source in map(Path, source_runs):
        for label in (source/'sequences').glob('*/labels/*.json'):
            record = read(label); key = record['sequence_id']+'/'+record['job_id']
            if key not in wanted:
                continue
            if key in found or sha(label) != wanted[key]['label_sha256'] or not frame_valid(label, dispatch['fingerprint']):
                raise ValueError('Corrupt or duplicate source import: '+key)
            definition=resolve_tip_definition(label,record)
            ds.copy_atomic(definition,run/'config/definition_versions'/f'{sha(definition)}.json')
            for archived in (source/'config/definition_versions').glob('*.json'):
                if archived.stem!=sha(archived):raise ValueError('Corrupt archived definition')
                ds.copy_atomic(archived,run/'config/definition_versions'/archived.name)
            for name in ds.RENDER_CONFIG_FILES:
                target=run/'config'/name
                if target.exists() and read(target) != read(source/'config'/name):
                    raise ValueError('Imported render configuration differs')
                ds.copy_atomic(source/'config'/name,target)
            base = run/label.relative_to(source)
            for eye in record['eyes'].values():
                for kind in ('image','mask'):
                    ds.copy_atomic(label.parent/eye[kind],base.parent/eye[kind])
            ds.copy_atomic(label.parent/record['camera_truth'],base.parent/record['camera_truth'])
            ds.copy_atomic(label,base)
            ds.copy_atomic(label.parent/'.commits'/label.name,base.parent/'.commits'/label.name)
            prefix=record['sequence_id']+'_'+record['job_id']
            for preview in (source/'previews').glob(prefix+'.*'):
                ds.copy_atomic(preview,run/'previews'/preview.name)
            ds.validate_frame_owner(base,context,dispatch)
            if not frame_valid(base,dispatch['fingerprint']):
                raise ValueError('Transfer did not preserve complete receipt')
            found.add(key)
    if found != wanted.keys():
        raise ValueError('Some assigned imports were not provided')
    for digest, manifest in dispatch['origin_manifests'].items():
        write(run/'config/original_workers'/f'{digest}.json',manifest)
    write(run/'validation/imported_frames.json',{'passed':True,'count':len(found),'keys':sorted(found),
          'labels_and_rgb_byte_identical':True,'original_renderer_identity_retained':True})
    return len(found)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    p=commands.add_parser('create')
    for flag in ('dispatch','output','source-code','dependencies'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--sources',type=Path,nargs='+',required=True)
    p.add_argument('--seconds-4080',type=float,required=True)
    p.add_argument('--seconds-5090',type=float,required=True)
    p.add_argument('--pin-4080',nargs='+',required=True)
    p.add_argument('--pin-5090',nargs='*',default=[])
    p=commands.add_parser('import')
    p.add_argument('--sources',type=Path,nargs='+',required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--worker',required=True)
    a=parser.parse_args()
    if a.action=='create':
        deps=read(a.dependencies); deps=deps.get('dependencies',deps)
        result=create(a.dispatch,a.output,a.source_code,a.sources,deps,a.seconds_4080,a.seconds_5090,a.pin_4080,a.pin_5090)
        print('REDISTRIBUTION_CREATED',result['dispatch_id'],result['workers'])
    else:
        print('IMPORTED_PAIRS',import_frames(a.sources,a.run,a.worker))
