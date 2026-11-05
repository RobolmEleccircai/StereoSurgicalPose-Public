#!/usr/bin/env python3
"""Explicit plan/validate/generate entry point. Never defaults to full generation."""
import argparse, os, subprocess, sys
from pathlib import Path
from io_utils import HERE,read,write,load_machine,assert_mode,lock,failure

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plan','validate','generate'],default='validate')
    p.add_argument('--machine',type=Path,required=True)
    p.add_argument('--settings',type=Path,default=HERE/'settings.balanced21.json')
    p.add_argument('--run-id',default='validation_v2')
    p.add_argument('--tests-only',action='store_true')
    p.add_argument('--pack-assets',type=Path,help='Export assets into a NEW directory; does not render')
    a=p.parse_args();m=load_machine(a.machine);assert_mode(a.mode,m)
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
    with lock(run/'.run.lock'):
        phase='tests' if a.tests_only else 'plan'
        try:
            subprocess.run([str(isaac/'python.sh'),str(HERE/('test_suite.py' if a.tests_only else 'planner.py')),*args],env=env,check=True)
            if a.tests_only or a.mode=='plan':return
            phase='capture'
            subprocess.run([str(isaac/'python.sh'),str(HERE/'capture.py'),*args],env=env,check=True)
            phase='verify'
            subprocess.run([str(isaac/'python.sh'),str(HERE/'verify.py'),*args],env=env,check=True)
        except BaseException as e:
            failure(run,phase,e);raise

if __name__=='__main__':main()
