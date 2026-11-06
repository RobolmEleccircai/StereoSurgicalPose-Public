#!/usr/bin/env python3
"""Explicit plan/validate/generate entry point. Never defaults to full generation."""
import argparse, os, subprocess, sys
from pathlib import Path
from contextlib import ExitStack
from io_utils import HERE,read,write,load_machine,assert_mode,lock,failure

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plan','validate','generate','shard','merge'],default='validate')
    p.add_argument('--machine',type=Path,required=True)
    p.add_argument('--settings',type=Path,default=HERE/'settings.balanced21.json')
    p.add_argument('--run-id',default='validation_v2')
    p.add_argument('--tests-only',action='store_true')
    p.add_argument('--pack-assets',type=Path,help='Export assets into a NEW directory; does not render')
    p.add_argument('--dispatch',type=Path,help='Frozen shared plan directory, required for shard mode')
    p.add_argument('--worker',choices=['rtx4060','rtx4080super','rtx5090'])
    p.add_argument('--prepare-only',action='store_true',help='Initialize assigned run and audit local assets without rendering')
    p.add_argument('--shard-runs',type=Path,nargs='+',help='Two finished worker runs, required for merge mode')
    a=p.parse_args();m=load_machine(a.machine);assert_mode(a.mode,m)
    if a.mode=='shard':
        if not a.dispatch or not a.worker: p.error('shard requires --dispatch and --worker')
    elif a.dispatch or a.worker or a.prepare_only: p.error('dispatch/worker/prepare-only are only valid in shard mode')
    if (a.mode=='merge') != bool(a.shard_runs):p.error('merge requires --shard-runs; other modes do not accept it')
    if a.mode in ('shard','merge') and (a.tests_only or a.pack_assets):p.error('Distributed modes cannot run tests-only or pack-assets')
    if not a.run_id or Path(a.run_id).name!=a.run_id:raise ValueError('run-id must be one directory name')
    isaac=Path(m['isaac_root']);version=(isaac/'VERSION').read_text().strip()
    if not version.startswith(m['expected_isaac_version']):raise ValueError('Isaac Sim version mismatch: '+version)
    usd=sorted((isaac/'extscache').glob('omni.usd.libs-*'))
    if len(usd)!=1:raise ValueError('Expected one USD runtime extension; found '+str(usd))
    env=os.environ.copy()
    # Tiny FK/IK matrices are slower with a large BLAS thread pool.
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    env['PYTHONPATH']=os.pathsep.join([str(usd[0]),m['deps_root'],env.get('PYTHONPATH','')])
    env['LD_LIBRARY_PATH']=os.pathsep.join([str(usd[0]/'bin'),env.get('LD_LIBRARY_PATH','')])
    if a.pack_assets:
        subprocess.run([str(isaac/'python.sh'),str(HERE/'pack_assets.py'),'--machine',str(a.machine.resolve()),
                        '--settings',str(a.settings.resolve()),'--destination',str(a.pack_assets.resolve())],env=env,check=True)
        return
    run=Path(m['output_root'])/a.run_id
    run.mkdir(parents=True,exist_ok=True)
    args=['--machine',str(a.machine.resolve()),'--settings',str(a.settings.resolve()),'--run',str(run),'--mode',a.mode]
    with ExitStack() as locks:
        locks.enter_context(lock(run/'.run.lock'))
        if a.mode=='shard':
            from dispatch import load_dispatch,worker_context
            d=load_dispatch(a.dispatch,read(a.settings));context=worker_context(d,a.worker,m)
            claim=Path(m['output_root'])/'.worker_claims'/(d['dispatch_id']+'.'+a.worker+'.json')
            locks.enter_context(lock(claim.with_suffix('.lock')))
            owner={'run':str(run.resolve()),'worker':context}
            if claim.exists() and read(claim)!=owner:raise ValueError('This worker is already assigned to another run directory')
            write(claim,owner)
        elif (run/'config/dispatch.json').exists():
            raise ValueError('Distributed run cannot be used by another mode or overwritten')
        phase='tests' if a.tests_only else 'plan'
        try:
            if a.mode in ('shard','merge'):
                phase='prepare_'+a.mode
                cmd=[str(isaac/'python.sh'),str(HERE/'dispatch.py'),'worker' if a.mode=='shard' else 'merge',
                     '--machine',str(a.machine.resolve()),'--settings',str(a.settings.resolve()),'--run',str(run)]
                cmd+=['--dispatch',str(a.dispatch.resolve()),'--worker',a.worker] if a.mode=='shard' else ['--inputs',*[str(p.resolve()) for p in a.shard_runs]]
                subprocess.run(cmd,env=env,check=True)
                if a.prepare_only:return
            else:
                subprocess.run([str(isaac/'python.sh'),str(HERE/('test_suite.py' if a.tests_only else 'planner.py')),*args],env=env,check=True)
                if a.tests_only or a.mode=='plan':return
            if a.mode!='merge':
                phase='capture'
                subprocess.run([str(isaac/'python.sh'),str(HERE/'capture.py'),*args],env=env,check=True)
            phase='verify'
            subprocess.run([str(isaac/'python.sh'),str(HERE/'verify.py'),*args],env=env,check=True)
        except BaseException as e:
            failure(run,phase,e);raise

if __name__=='__main__':main()
