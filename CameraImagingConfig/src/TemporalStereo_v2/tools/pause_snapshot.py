"""Pause only the named generation processes after their next committed pair."""
import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

p=argparse.ArgumentParser()
for flag in ('run','output','source-code'):
    p.add_argument('--'+flag,type=Path,required=True)
p.add_argument('--wrapper-pid',type=int,required=True)
p.add_argument('--renderer-pid',type=int,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.source_code))
from io_utils import read,write,frame_valid,sha
run=a.run.resolve();out=a.output.resolve()
assert not out.exists()
cmd=Path(f'/proc/{a.renderer_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(run) in cmd,cmd
wrapper_cmd=Path(f'/proc/{a.wrapper_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'run_work' in wrapper_cmd or 'run_assigned_worker.py' in wrapper_cmd,wrapper_cmd
initial=read(run/'status.json')['completed_pairs']
print('WAITING_FOR_NEXT_PAIR',initial,flush=True)
deadline=time.monotonic()+600
while True:
    status=read(run/'status.json')
    if status.get('state')!='rendering':raise RuntimeError('Source is no longer rendering')
    if status['completed_pairs']>initial:break
    if time.monotonic()>deadline:raise TimeoutError('No new committed frame within 10 minutes')
    time.sleep(.2)
paused=[]
try:
    for pid in (a.renderer_pid,a.wrapper_pid):
        os.kill(pid,signal.SIGSTOP);paused.append(pid)
    time.sleep(.1)
    for pid in paused:
        state=next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('State:'))
        assert 'T (stopped)' in state,state
    manifest=read(run/'config/manifest.json')
    labels=sorted((run/'sequences').glob('*/labels/*.json'))
    # Imported pairs in later sequences exist before the traversal reaches them.
    assert len(labels)>=status['completed_pairs'],(len(labels),status)
    for label in labels:
        assert frame_valid(label,manifest['fingerprint']),label
    shutil.copytree(run/'config',out/'config')
    shutil.copytree(run/'trajectories',out/'trajectories')
    for label in labels:
        r=read(label);target=out/label.relative_to(run)
        pairs=[(label,target),(label.parent/'.commits'/label.name,target.parent/'.commits'/label.name)]
        for eye in r['eyes'].values():
            for kind in ('image','mask'):pairs.append((label.parent/eye[kind],target.parent/eye[kind]))
        pairs.append((label.parent/r['camera_truth'],target.parent/r['camera_truth']))
        for source,destination in pairs:
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        prefix=r['sequence_id']+'_'+r['job_id']
        for source in (run/'previews').glob(prefix+'.*'):
            destination=out/'previews'/source.name;destination.parent.mkdir(exist_ok=True);shutil.copy2(source,destination)
        assert frame_valid(target,manifest['fingerprint']),target
    marker={'renderer_stopped':True,'committed_pairs':len(labels),'source_run':str(run),
            'renderer_pid':a.renderer_pid,'wrapper_pid':a.wrapper_pid,'paused_unix':time.time(),
            'label_hashes':{str(p.relative_to(run)):sha(p) for p in labels}}
    write(out/'paused_for_redistribution.json',marker)
    print('PAUSED_AND_SNAPSHOTTED',len(labels),out,flush=True)
except BaseException:
    for pid in paused:
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    raise
