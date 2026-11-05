"""Run this deployment's unchanged collector using the verified Isaac Sim 6 adapter.

Planning identity remains the shared 5.1 release. Actual rendering dependencies
are recorded separately, including the adapter hash; neither is mislabelled.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
project = root / 'project'
code = project / 'code_distributed_7d4a31c'
ops = root / 'operations/4080_20261109'
ops.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(code))
from io_utils import read, write, sha, source_hash, load_machine, lock, failure
from dispatch import load_dispatch, worker_context, run_context

planning_machine = project / 'config/machine.4080.local.json'
render_machine = project / 'config/machine.4080.runtime6.local.json'
pm, rm = load_machine(planning_machine), load_machine(render_machine)
for key in ('asset_root', 'output_root', 'worker_id', 'role'):
    assert pm[key] == rm[key], 'Planning/render machine identity differs: ' + key
dispatch = project / 'dispatch_2100_20261109'
settings = dispatch / 'config/effective_settings.json'
d = load_dispatch(dispatch, read(settings))
context = worker_context(d, 'rtx4080super', rm)
assert context['pair_count'] == 1748
run = Path(rm['output_root']) / 'worker_4080_20261109'
adapter = root / 'operations/capture_runtime6.py'
audit = read(project / 'checks/runtime6_geometry.json')
comparison = read(project / 'checks/runtime6_geometry_comparison.json')
probe_path = project / 'checks/runtime6_probe_validation.json'
probe = read(probe_path)
assert comparison['passed'] and probe['passed'] and probe['pairs'] == 2
assert probe['adapter_sha256'] == sha(adapter)
assert audit['code_sha256'] == source_hash() == d['code_sha256']
assert (Path(rm['isaac_root']) / 'VERSION').read_text().strip() == audit['dependencies']['isaac_version']
for item in audit['dependencies']['files']:
    assert sha(item['path']) == item['sha256'], 'Render dependency changed: ' + item['path']

def env_for(machine):
    sdk = Path(machine['isaac_root'])
    usd = list((sdk / 'extscache').glob('omni.usd.libs-*'))
    assert len(usd) == 1
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=str(usd[0]) + os.pathsep + machine['deps_root'],
               LD_LIBRARY_PATH=str(usd[0] / 'bin'), OMNI_KIT_ALLOW_ROOT='1',
               OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA='1', XDG_CACHE_HOME=str(root / 'cache6'))
    return sdk, env

sdk, env = env_for(rm)
versions = audit['dependencies']['python_packages']
package_check = 'import importlib.metadata; expected=' + repr(versions) + '; assert {k:importlib.metadata.version(k) for k in expected} == expected'
subprocess.run([str(sdk / 'python.sh'), '-c', package_check], env=env, check=True)
driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
execution = {
    'isaac_version': audit['dependencies']['isaac_version'], 'python': audit['python'],
    'opencv': audit['opencv'], 'code_sha256': source_hash(), 'adapter_sha256': sha(adapter),
    'launcher_sha256': sha(__file__), 'driver': driver,
    'adapter': 'legacy add_update_semantics -> native class labels',
    'geometry_comparison_sha256': sha(project / 'checks/runtime6_geometry_comparison.json'),
    'probe_validation_sha256': sha(probe_path),
}
claim = Path(rm['output_root']) / '.worker_claims' / (d['dispatch_id'] + '.rtx4080super.json')
state = {'phase': 'generating_4080_1748_pairs', 'pid': os.getpid(), 'started_unix': time.time(),
         'worker': 'rtx4080super', 'assigned_pairs': 1748, 'assigned_sequences': context['sequence_ids'],
         'worker_run': str(run), 'render_execution': execution, 'complete': False}
with lock(ops / 'workflow.lock'), lock(run / '.run.lock'), lock(claim.with_suffix('.lock')):
    assert read(claim) == {'run': str(run.resolve()), 'worker': context}
    assert run_context(run, rm, read(settings)) == context
    manifest = read(run / 'config/manifest.json')
    if 'render_execution' in manifest:
        assert manifest['render_execution'] == execution, 'Cannot mix renderer versions in this run'
    else:
        assert not list((run / 'sequences').glob('*/labels/*.json')), 'Existing images lack renderer provenance'
    manifest.update(dependency_audit_purpose='shared frozen-plan preparation with Isaac Sim 5.1',
                    render_dependencies=audit['dependencies'], render_execution=execution)
    write(run / 'config/manifest.json', manifest)
    write(ops / 'state.json', state)
    try:
        subprocess.run([str(sdk / 'python.sh'), str(adapter), '--code', str(code),
                        '--machine', str(render_machine), '--settings', str(settings),
                        '--run', str(run), '--mode', 'shard'], cwd=project, env=env, check=True)
        verification_sdk, verification_env = env_for(pm)
        subprocess.run([str(verification_sdk / 'python.sh'), str(code / 'verify.py'),
                        '--machine', str(planning_machine), '--settings', str(settings),
                        '--run', str(run), '--mode', 'shard'], cwd=project, env=verification_env, check=True)
        result = read(run / 'status.json')
        assert result['shard_complete'] and result['rendered_pairs'] == 1748
        state.update(phase='4080_shard_complete', complete=True, result=result, finished_unix=time.time())
        write(ops / 'state.json', state)
    except BaseException as error:
        failure(run, 'runtime6_worker', error)
        state.update(phase='failed', error=repr(error), failed_unix=time.time())
        write(ops / 'state.json', state)
        raise
