"""Portable, finite, atomic records; immutable provenance; strict render budgets."""
from pathlib import Path
from contextlib import contextmanager
import hashlib, json, os, fcntl, time

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def source_hash():
    files=list(HERE.glob('*.py'))+[HERE/'tools/reannotate.py']
    return canonical_hash({str(p.relative_to(HERE)):sha(p) for p in sorted(files)})

def load_machine(path):
    path=Path(path).resolve();m=read(path)
    if m.get('role') not in ('laptop','server'):raise ValueError('Machine role must be laptop or server')
    for key in ('isaac_root','asset_root','output_root','deps_root'):
        p=Path(os.path.expandvars(os.path.expanduser(m[key])))
        m[key]=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    if m.get('preview_font'):
        p=Path(os.path.expandvars(os.path.expanduser(m['preview_font'])))
        m['preview_font']=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    return m

def assert_mode(mode,machine):
    if mode not in ('plan','validate','generate'):raise ValueError('Unknown mode')
    if mode=='generate' and machine['role']!='server':
        raise ValueError('Full generation is forbidden on a laptop; use validate (maximum 32 render attempts).')

@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process owns '+str(path))
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def reserve_render(machine,settings,run_id,job_id):
    """Laptop scope is cumulative across runs and crashes, not reset by a new run id."""
    if machine['role']!='laptop':return
    limit=min(32,int(settings['validation_render_limit']))
    path=Path(machine['output_root'])/(settings['validation_budget_scope']+'.budget.json')
    with lock(path.with_suffix('.lock')):
        ledger=read(path) if path.exists() else {'attempts':[],'limit':limit}
        if len(ledger['attempts'])>=limit:raise RuntimeError('Laptop validation render budget exhausted; no more images will be rendered.')
        ledger['attempts'].append({'run_id':run_id,'job_id':job_id,'utc_unix':time.time()})
        write(path,ledger)

def frame_valid(label_path,expected_fingerprint):
    """A label is the final commit marker; every referenced file must match."""
    p=Path(label_path)
    try:
        receipt=read(p.parent/'.commits'/p.name)
        if receipt!={'fingerprint':expected_fingerprint,'sha256':sha(p)}:return False
        d=read(p)
        if d['fingerprint']!=expected_fingerprint:return False
        if set(d['eyes'])!={'left','right'}:return False
        for e in d['eyes'].values():
            for key in ('image','mask'):
                target=p.parent/e[key]
                if not target.is_file() or sha(target)!=e[key+'_sha256']:return False
        truth=p.parent/d['camera_truth']
        return truth.is_file() and sha(truth)==d['camera_truth_sha256']
    except (OSError,ValueError,KeyError,TypeError):return False

def commit_label(path,record):
    path=Path(path);write(path,record)
    write(path.parent/'.commits'/path.name,{'fingerprint':record['fingerprint'],'sha256':sha(path)})

def seal(path,fingerprint):
    path=Path(path);write(path.with_suffix('.sha256.json'),{'fingerprint':fingerprint,'sha256':sha(path)})

def require_seal(path,fingerprint):
    path=Path(path)
    if read(path.with_suffix('.sha256.json'))!={'fingerprint':fingerprint,'sha256':sha(path)}:
        raise ValueError('Changed or corrupt trajectory cache: '+str(path))

def validate_settings(c):
    if (c['width'],c['height'],c['fps'])!=(1280,720,60):raise ValueError('This profile requires 1280x720 at 60 Hz')
    if c['sequence_count']<14 or c['frames_per_sequence']<8:raise ValueError('At least 14 sequences and 8 frames are required for category coverage')
    if c['renderer']!='PathTracing' or c['rt_subframes']<4 or c['samples_per_pixel_per_frame']<64:raise ValueError('Approved path-tracing quality cannot be reduced')
    if not 0<c['contact_target_mm']<c['contact_max_mm'] or c['free_clearance_mm']<2:raise ValueError('Invalid tissue-clearance configuration')
    if c['jaw_speed_rad_s']>.2 or c['jaw_half_angle_deg']!=[0.,30.]:raise ValueError('Jaw limits exceed the agreed physical range')

def failure(run,phase,error):
    write(Path(run)/'status.json',{'state':'failed','phase':phase,'error':str(error),'complete':False})
