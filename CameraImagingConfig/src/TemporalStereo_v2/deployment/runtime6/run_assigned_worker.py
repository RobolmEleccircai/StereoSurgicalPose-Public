"""Run an already prepared, immutable assignment with the Isaac 6 adapter."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
for flag in ('code','machine','dispatch','run','validation-report'):
    p.add_argument('--'+flag,type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.code.resolve()))
from io_utils import read,write,sha,source_hash,load_machine,lock,failure
from dispatch import load_dispatch,worker_context,run_context

m=load_machine(a.machine)
settings=a.dispatch/'config/effective_settings.json'
d=load_dispatch(a.dispatch,read(settings))
worker=worker_context(d,m['worker_id'],m)
run=a.run.resolve()
assert run.parent==Path(m['output_root']).resolve()
assert run_context(run,m,read(settings))==worker
adapter=Path(__file__).with_name('capture_runtime6.py')
validation=read(a.validation_report)
assert validation['passed'] and validation['adapter_sha256']==sha(adapter)
sdk=Path(m['isaac_root'])
version=(sdk/'VERSION').read_text().strip()
assert version.startswith('6.0.0-') and version.startswith(m['expected_isaac_version'])
usd=list((sdk/'extscache').glob('omni.usd.libs-*'))
assert len(usd)==1
env=os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
           PYTHONPATH=str(usd[0])+os.pathsep+m['deps_root'],LD_LIBRARY_PATH=str(usd[0]/'bin'),
           OMNI_KIT_ALLOW_ROOT='1',OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA='1',
           XDG_CACHE_HOME=str(sdk.parent/'cache6'))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
execution={'isaac_version':version,'gpu':gpu,'code_sha256':source_hash(),
           'adapter_sha256':sha(adapter),'launcher_sha256':sha(__file__),
           'validation_report_sha256':sha(a.validation_report)}
claim=Path(m['output_root'])/'.worker_claims'/(d['dispatch_id']+'.'+m['worker_id']+'.json')
with lock(run/'.run.lock'),lock(claim.with_suffix('.lock')):
    assert read(claim)=={'run':str(run),'worker':worker}
    manifest=read(run/'config/manifest.json')
    for item in manifest['dependencies']['files']:
        assert sha(item['path'])==item['sha256'],'Changed dependency: '+item['path']
    if 'render_execution' in manifest:
        assert manifest['render_execution']==execution,'Execution version changed during resume'
    manifest['render_execution']=execution
    write(run/'config/manifest.json',manifest)
    state={'phase':'rendering','pid':os.getpid(),'worker':worker,'started_unix':time.time(),
           'render_execution':execution,'complete':False}
    write(run/'workflow_state.json',state)
    args=['--machine',str(a.machine.resolve()),'--settings',str(settings.resolve()),'--run',str(run),'--mode','shard']
    try:
        subprocess.run([str(sdk/'python.sh'),str(adapter),'--code',str(a.code.resolve()),*args],env=env,check=True)
        state['phase']='verifying';write(run/'workflow_state.json',state)
        subprocess.run([str(sdk/'python.sh'),str(a.code/'verify.py'),*args],env=env,check=True)
        result=read(run/'status.json')
        assert result['shard_complete'] and result['rendered_pairs']==worker['pair_count']
        state.update(phase='shard_complete',complete=True,finished_unix=time.time())
        write(run/'workflow_state.json',state)
    except BaseException as error:
        failure(run,'assigned_worker',error)
        state.update(phase='failed',error=repr(error),complete=False)
        write(run/'workflow_state.json',state)
        raise
