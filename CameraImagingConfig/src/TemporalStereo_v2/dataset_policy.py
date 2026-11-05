"""Balanced 2,100-pair policy, independent trajectory groups and coverage gates.

Sampling ranges and duplicate tolerances are dataset design choices, not measured
camera specifications. This module has no USD or renderer dependency.
"""
import copy
from collections import Counter, defaultdict
import numpy as np
from io_utils import canonical_hash

POLICY = 'balanced_2100_v1'
SPLITS = ('train', 'validation', 'test')
# category, physical parent path, train/validation/test frames, jaw start/end degrees
LAYOUT = (
    ('pose_motion', 'return', (180, 32, 32), (30., 24.)),
    ('lighting', 'hover', (180, 32, 32), (4., 9.)),
    ('sensor_quality', 'slide', (180, 32, 32), (11., 17.)),
    ('occlusion', 'occlusion', (240, 140, 140), (12., 16.)),
    ('ecm_motion', 'arc', (180, 32, 32), (19., 13.)),
    ('stereo_rig', 'return', (180, 32, 32), (0., 6.)),
    ('combined_pose_jaw', 'slide', (260, 50, 50), (22., 28.)),
)
CATEGORIES = tuple(row[0] for row in LAYOUT)
TOTALS = {'train': 1400, 'validation': 350, 'test': 350}
RANGES = {'light': (.7, 1.3), 'noise': (.5, 1.5), 'blur': (.75, 1.25),
          'baseline': (.95, 1.05), 'relative_yaw': (-.5, .5)}
BIN_ASSIGNMENTS = {
    'light': (1, 2, 0, 1, 0, 1, 2), 'noise': (0, 1, 2, 0, 2, 1, 1),
    'blur': (2, 0, 1, 1, 2, 1, 0), 'baseline': (0, 1, 2, 1, 0, 2, 1),
    'relative_yaw': (2, 1, 0, 1, 2, 0, 1),
}
ECM_CATEGORIES = ('occlusion', 'ecm_motion', 'combined_pose_jaw')


def enabled(c):
    return c.get('dataset_policy') == POLICY


def make_manifest(seed):
    rows = []
    for ci, (category, kind, counts, angles) in enumerate(LAYOUT):
        for si, split in enumerate(SPLITS):
            rows.append({'sequence_id': f'seq_{3*ci+si:03d}', 'category': category,
                         'kind': kind, 'split': split, 'frame_count': counts[si],
                         'parent_frame_count': counts[si] if split == 'train' else 150,
                         'seed': int(seed + 1009*ci + 101*si),
                         'feature_profile_id': POLICY + '/' + category,
                         'condition_bins': {k: v[ci] for k, v in BIN_ASSIGNMENTS.items()},
                         'features': ['psm_motion', 'jaw_motion', 'lighting', 'noise', 'blur', 'stereo_variation']
                                     + (['ecm_motion'] if category in ECM_CATEGORIES else []),
                         'jaw_angles_deg': list(angles),
                         'jaw_hold_start_s': .25 if category in ('pose_motion', 'stereo_rig') else 0.})
    return rows


def validate_policy(c):
    rows = c.get('sequence_manifest', [])
    expected = make_manifest(c['seed'])
    if rows != expected:
        raise ValueError('Balanced manifest must match the versioned 21-sequence policy')
    if c.get('sequence_count') != 21 or 'frames_per_sequence' in c:
        raise ValueError('Balanced policy uses 21 per-sequence lengths, not a global frame count')
    if c.get('validation_selection') != 'balanced_32':
        raise ValueError('Balanced policy requires balanced_32 smoke selection')
    if c.get('sampling_ranges') != {k: list(v) for k, v in RANGES.items()}:
        raise ValueError('Balanced sampling ranges changed without a new policy version')


def bin_edges(factor, level):
    lo, hi = RANGES[factor]
    return lo + (hi-lo)*level/3, lo + (hi-lo)*(level+1)/3


def sample_conditions(row, rng):
    result = copy.deepcopy(row)
    samples = {}
    for factor, level in row['condition_bins'].items():
        lo, hi = bin_edges(factor, level)
        samples[factor] = float(rng.uniform(lo + 1e-8, hi - 1e-8))
    lo, hi = bin_edges('light', row['condition_bins']['light'])
    ends = [float(rng.uniform(lo, lo + .3*(hi-lo))),
            float(rng.uniform(hi - .3*(hi-lo), hi))]
    if rng.random() < .5:
        ends.reverse()
    result.update(light_start=ends[0], light_end=ends[1], noise_multiplier=samples['noise'],
                  blur_multiplier=samples['blur'], baseline_scale=samples['baseline'],
                  relative_yaw_delta_deg=samples['relative_yaw'], ecm_motion='ecm_motion' in row['features'])
    # Keep the physical hard-stop examples exactly 0/30. Other endpoints stay in
    # the same opening interval as their training counterpart.
    a0, a1 = row['jaw_angles_deg']
    result['jaw_angles_deg'] = [a0, a1 + float(rng.uniform(-.5, .5))]
    return result


def effects_at(spec, source_frame_index, fps):
    from paths import smoother
    n = spec.get('parent_frame_count') or len(spec['frames'])
    u = source_frame_index/(n-1)
    return {'light_multiplier': float(spec['light_start'] + (spec['light_end']-spec['light_start'])*smoother(u)),
            'noise_multiplier': spec['noise_multiplier'], 'blur_multiplier': spec['blur_multiplier']}


def tissue_hidden(frame):
    return any(v.get('occluder_kind') == 'tissue' or
               (v.get('occluder') or '').startswith('/models_topo_blender/')
               for eye in frame['jaw_visibility'].values() for v in eye)


def target_hidden(frame, target):
    v = frame['jaw_visibility'][target['eye']][target['jaw_index']]
    return v.get('occluder_kind') == 'tissue' or (v.get('occluder') or '').startswith('/models_topo_blender/')


def occlusion_coverage(frames, target):
    hidden = [target_hidden(f, target) for f in frames]
    runs = []
    start = 0
    for i in range(1, len(hidden)+1):
        if i == len(hidden) or hidden[i] != hidden[start]:
            runs.append((hidden[start], i-start)); start = i
    return any(not a[0] and a[1] >= 5 and b[0] and b[1] >= 15 and not z[0] and z[1] >= 5
               for a, b, z in zip(runs, runs[1:], runs[2:]))


def jaw_state(angle):
    if abs(angle) <= 1e-6: return 'closed'
    if angle < 10: return 'small'
    if angle < 20: return 'medium'
    return 'large'


def signatures(seq):
    # Fixed-point self-occlusion is deliberately not equated with tissue occlusion.
    return {(f['phase'], jaw_state(f['jaw_half_angle_deg']), tissue_hidden(f)) for f in seq['frames']}


def fixed_visibility_facts(seq):
    return {(f['phase'],'visible' if v['visible'] else v['occluder_kind'])
            for f in seq['frames'] for eye in f['jaw_visibility'].values() for v in eye}


def adequate_clip_phases(spec, frames):
    if spec['split']=='train':return True
    counts=Counter(f['phase'] for f in frames)
    required={'sensor_quality':('approach','contact_hold'),
              'combined_pose_jaw':('contact_slide','withdraw')}.get(spec['category'],())
    # A single boundary sample does not exercise an action. Reserve at least
    # one third of these short clips for each of the two requested phases.
    return all(counts[phase]>=max(5,spec['frame_count']//3) for phase in required)


def choose_clip(spec, parent, fps, training=None):
    count = spec['frame_count']
    if len(parent) != spec['parent_frame_count'] or count > len(parent):
        raise ValueError('Invalid parent/clip length')
    if spec['split'] == 'train':
        starts = [0]
    elif spec['category'] in ('pose_motion', 'stereo_rig'):
        starts = [0]  # Retain the true 30/0 degree jaw hard stops.
    else:
        starts = list(range(len(parent)-count+1))
    eligible = []
    for start in starts:
        clip = copy.deepcopy(parent[start:start+count])
        phases = {f['phase'] for f in clip}
        if spec['kind'] == 'occlusion' and not occlusion_coverage(clip, spec['occlusion_target']): continue
        if not adequate_clip_phases(spec,clip):continue
        if spec['split'] != 'train':
            if spec['category'] == 'lighting' and 'hover' not in phases: continue
        candidate = dict(spec, frames=clip)
        if training is not None and not signatures(candidate) <= signatures(training): continue
        if training is not None and not fixed_visibility_facts(candidate) <= fixed_visibility_facts(training): continue
        q = np.array([f['q_psm'] for f in clip]); scale = np.std(q, axis=0)
        # A scalar score is only used to pick among already-safe, covered windows.
        score = float(np.linalg.norm(scale))
        eligible.append((score, start, clip))
    if not eligible:
        raise ValueError('No continuous clip meets feature/training coverage and duration')
    _, start, clip = max(eligible, key=lambda row: (row[0], -row[1]))
    for i, frame in enumerate(clip):
        frame['source_frame_index'] = frame['frame_index']
        frame['source_time_s'] = frame['time_s']
        frame.update(frame_index=i, time_s=i/fps, timestamp_ns=round(i*1e9/fps))
    spec['clip_start_frame'] = start
    spec['clip_start_time_s'] = start/fps
    return clip


def trajectory_descriptor(scene, frames):
    indices = np.rint(np.linspace(0, len(frames)-1, 33)).astype(int)
    C, R, q = [], [], []
    for i in indices:
        v = np.array(frames[i]['q_psm']); t = scene.psm.fk(v)[scene.roll]
        C.append(t[:3, 3]); R.append(t[:3, :3]); q.append(v)
    return {'C_mm': np.array(C).tolist(), 'R_world': np.array(R).tolist(), 'q_psm': np.array(q).tolist(),
            'joint_names': list(scene.psm.names)}


def group_id(descriptor):
    q = np.round(descriptor['q_psm'], 8).tolist()
    # Camera/light/noise/time changes do not create a new motion group.
    return 'trajectory_' + min(canonical_hash(q), canonical_hash(q[::-1]))[:24]


def frechet(cost):
    acc = np.full(cost.shape, np.inf)
    for i in range(len(cost)):
        for j in range(cost.shape[1]):
            previous = min(acc[i-1,j] if i else np.inf, acc[i,j-1] if j else np.inf,
                           acc[i-1,j-1] if i and j else np.inf)
            acc[i,j] = max(float(cost[i,j]), 0. if i == j == 0 else previous)
    return float(acc[-1,-1])


def near_duplicate(a, b):
    """Time-warp/reversal invariant near-copy check; no camera or RGB dependence."""
    A, B = np.array(a['C_mm']), np.array(b['C_mm'])
    RA, RB = np.array(a['R_world']), np.array(b['R_world'])
    distances = np.linalg.norm(A[:,None,:]-B[None,:,:], axis=2)  # threshold 1 mm
    cosine = np.clip((np.einsum('aij,bij->ab', RA, RB)-1)/2, -1, 1)
    rotations = np.arccos(cosine)/np.deg2rad(2.)
    names = a['joint_names']
    if names != b['joint_names']: raise ValueError('Incompatible trajectory descriptor joints')
    active = [i for i,n in enumerate(names) if 'tool_pitch' in n or 'tool_yaw' in n or 'gripper' in n]
    qa, qb = np.array(a['q_psm'])[:,active], np.array(b['q_psm'])[:,active]
    joints = np.max(abs(qa[:,None,:]-qb[None,:,:]), axis=2)/np.deg2rad(2.)
    costs = np.maximum.reduce([distances, rotations, joints])
    return min(frechet(costs), frechet(costs[:,::-1])) <= 1.


def check_against(seq, accepted):
    for other in accepted:
        if seq['split'] == other['split']: continue
        if seq['trajectory_group_id'] == other['trajectory_group_id']:
            raise ValueError('Source trajectory group crosses dataset splits')
        if near_duplicate(seq['trajectory_descriptor'], other['trajectory_descriptor']):
            raise ValueError('Near-duplicate trajectory crosses dataset splits')


def audit(sequences, c):
    validate_policy(c)
    expected = {r['sequence_id']:r for r in c['sequence_manifest']}
    if len(sequences) != 21 or {s['sequence_id'] for s in sequences} != set(expected):
        raise ValueError('Missing or duplicate balanced sequence')
    counts = Counter(); bins = defaultdict(Counter); cats = defaultdict(Counter); states = defaultdict(Counter)
    accepted = []; train = {s['category']:s for s in sequences if s['split'] == 'train'}
    details = {}
    for seq in sequences:
        row = expected[seq['sequence_id']]
        for field in ('split','category','kind','seed','features','condition_bins','feature_profile_id','frame_count','parent_frame_count'):
            if seq[field] != row[field]: raise ValueError('Sequence policy mismatch: '+field)
        if len(seq['frames']) != row['frame_count'] or len(seq['parent_frames']) != row['parent_frame_count']:
            raise ValueError('Balanced frame count changed')
        check_against(seq, accepted); accepted.append(seq)
        if seq['trajectory_group_id'] != group_id(seq['trajectory_descriptor']):
            raise ValueError('Trajectory group ID does not match geometry descriptor')
        for factor, level in seq['condition_bins'].items():
            bins[(seq['split'],factor)][level] += 1
            fields = {'light':('light_start','light_end'), 'noise':('noise_multiplier',),
                      'blur':('blur_multiplier',),'baseline':('baseline_scale',),
                      'relative_yaw':('relative_yaw_delta_deg',)}[factor]
            lo, hi = bin_edges(factor, level)
            if any(not lo-1e-10 <= seq[f] <= hi+1e-10 for f in fields):
                raise ValueError('Condition outside its training-covered bin')
        if seq['split'] != 'train' and not signatures(seq) <= signatures(train[seq['category']]):
            raise ValueError('Held-out phase/opening/tissue condition missing from training')
        if seq['split'] != 'train' and not fixed_visibility_facts(seq)<=fixed_visibility_facts(train[seq['category']]):
            raise ValueError('Held-out fixed-point visibility condition missing from training')
        if seq['kind'] == 'occlusion' and not occlusion_coverage(seq['frames'], seq['occlusion_target']):
            raise ValueError('Incomplete visible-hidden-visible clip')
        if not adequate_clip_phases(seq,seq['frames']):
            raise ValueError('Short clip has insufficient samples of requested action phases')
        for i,f in enumerate(seq['frames']):
            source = seq['clip_start_frame']+i
            if f['frame_index'] != i or f['source_frame_index'] != source or abs(f['source_time_s']-source/c['fps'])>1e-12:
                raise ValueError('Clip compressed, discontinuous or source time changed')
            original = seq['parent_frames'][source]
            for key in ('q_psm','q_ecm','v_psm','v_ecm','appearance','jaw_visibility','phase','jaw_half_angle_deg'):
                if f[key] != original[key]: raise ValueError('Clip state differs from safe parent: '+key)
            if abs(f['time_s']-i/c['fps'])>1e-12 or f['timestamp_ns'] != round(i*1e9/c['fps']):
                raise ValueError('Clip output time mismatch')
            states[seq['split']][jaw_state(f['jaw_half_angle_deg'])] += 1
        counts[seq['split']] += len(seq['frames']); cats[seq['split']][seq['category']] += len(seq['frames'])
        frames=seq['frames'];angles=[f['jaw_half_angle_deg'] for f in frames]
        joints=np.asarray([f['q_psm'] for f in frames])
        details[seq['sequence_id']]={'split':seq['split'],'category':seq['category'],
            'output_frames':len(frames),'parent_frames':len(seq['parent_frames']),
            'source_start_frame':seq['clip_start_frame'],'timestamp_span_s':frames[-1]['time_s'],
            'motion_phase_counts':dict(Counter(f['phase'] for f in frames)),
            'opening_counts':dict(Counter(jaw_state(v) for v in angles)),
            'jaw_half_angle_range_deg':[min(angles),max(angles)],
            'tissue_occluded_frames':sum(tissue_hidden(f) for f in frames),
            'geometric_contact_frames':sum(f.get('contact',{}).get('geometric_contact',False) for f in frames),
            'condition_bins':seq['condition_bins'],
            'actual_image_parameter_ranges':{key:[min(f['appearance'][key] for f in frames),max(f['appearance'][key] for f in frames)]
                for key in ('light_multiplier','noise_multiplier','blur_multiplier')},
            'joint_ranges':{name:[float(lo),float(hi)] for name,lo,hi in
                zip(seq['trajectory_descriptor']['joint_names'],joints.min(0),joints.max(0))}}
    if dict(counts) != TOTALS: raise ValueError('Balanced split totals changed')
    for split in SPLITS:
        if set(cats[split]) != set(CATEGORIES) or not {'closed','small','medium','large'} <= set(states[split]):
            raise ValueError('Missing category or jaw-opening state: '+split)
        for factor in RANGES:
            if bins[(split,factor)] != Counter({0:2,1:3,2:2}): raise ValueError('Unbalanced condition bins')
    return {'passed':True, 'policy':POLICY, 'planned_pairs':2100, 'counts':dict(counts),
            'category_frames':dict(cats), 'opening_frames':dict(states),
            'condition_bin_sequence_counts':{s:{f:dict(bins[(s,f)]) for f in RANGES} for s in SPLITS},
            'independent_motion_groups':len({s['trajectory_group_id'] for s in sequences}),
            'sequence_state_coverage':details,
            'held_out_feature_coverage_passed':True, 'near_copy_thresholds':{'C_mm':1.,'orientation_deg':2.,'articulation_deg':2.},
            'rendered_visibility_and_exposure_coverage':'pending',
            'scope':'planned trajectories and parameter/state coverage, not learned model accuracy'}


def diagnostic_jobs(sequences):
    jobs=[]; by_cat=defaultdict(list)
    for seq in sequences:
        # Select an actual moving interval, not the zero-velocity midpoint of a
        # quintic return. Normalize per-joint spans to combine angles/meters.
        q=np.asarray([f['q_psm']+f['q_ecm'] for f in seq['frames']])
        span=np.ptp(q,axis=0);active=span>1e-8
        if not active.any():raise ValueError('Diagnostic sequence has no motion')
        speed=np.linalg.norm(np.diff(q[:,active],axis=0)/span[active],axis=1)
        i=int(np.argmax(speed))
        jobs.append({'sequence_id':seq['sequence_id'],'frame_index':i,'reason':'balanced_sequence_sample'})
        by_cat[seq['category']].append(seq)
    for ci,cat in enumerate(CATEGORIES):
        seq=next(s for s in by_cat[cat] if s['split']==SPLITS[ci%3])
        i=next(j['frame_index'] for j in jobs if j['sequence_id']==seq['sequence_id'])+1
        jobs.append({'sequence_id':seq['sequence_id'],'frame_index':i,'reason':'balanced_adjacent_sample'})
    closed=next(s for s in sequences if s['split']=='train' and s['category']=='stereo_rig')
    opened=next(s for s in sequences if s['split']=='train' and s['category']=='pose_motion')
    contact=next(s for s in sequences if s['split']=='train' and s['category']=='combined_pose_jaw')
    jobs += [{'sequence_id':closed['sequence_id'],'frame_index':0,'reason':'closed_jaw'},
             {'sequence_id':opened['sequence_id'],'frame_index':0,'reason':'maximum_opening'},
             {'sequence_id':contact['sequence_id'],'frame_index':len(contact['frames'])//2,'reason':'contact_keyframe'},
             {'sequence_id':contact['sequence_id'],'frame_index':len(contact['frames'])-1,'reason':'withdraw_keyframe'}]
    for i,j in enumerate(jobs): j['job_id']=f'validation_{i:03d}'
    if len(jobs)!=32 or len({j['sequence_id'] for j in jobs})!=21: raise ValueError('Invalid diagnostic selection')
    return jobs


def rendered_coverage(records):
    """Called only for complete full output; partial smoke evidence is not sufficient."""
    train=defaultdict(set); observed=defaultdict(set);counts=defaultdict(Counter)
    for r in records:
        key=(r['category'],r['phase'])
        eye_states=[]
        for e in r['eyes'].values():
            eye_states += [('fixed',v['occluder_kind'] if not v['visible'] else 'visible') for v in e['fixed_tip_visibility']]
            eye_states += [('region_visible',bool(v['region_visible'])) for v in e['jaw_tip_region_visibility']]
        facts=set(eye_states)|{('overexposed',bool(r['quality_flags']['overexposed_any_eye']))}
        counts[(r['split'],r['category'])]['pairs']+=1
        counts[(r['split'],r['category'])]['overexposed_pairs']+=int(r['quality_flags']['overexposed_any_eye'])
        for name,value in eye_states:counts[(r['split'],r['category'])][name+':'+str(value)]+=1
        observed[(r['split'],key)].update(facts)
        if r['split']=='train': train[key].update(facts)
    missing=[]
    for (split,key),facts in observed.items():
        if split!='train' and not facts<=train[key]:
            missing.append({'split':split,'category':key[0],'phase':key[1],
                            'missing_training_conditions':sorted(facts-train[key],key=str)})
    return {'passed':not missing,'missing_training_coverage':missing,
            'by_split_and_category':{s:{cat:dict(counts[(s,cat)]) for cat in CATEGORIES} for s in SPLITS},
            'overexposed_and_occluded_samples':'retained', 'scope':'saved RGB and visibility labels'}


def validate_saved_state(record, seq, joint_order):
    """Compare rendered metadata against the sealed original parent, not clip time."""
    f=seq['frames'][record['frame_index']]
    for key in ('source_frame_index','source_time_s','time_s','timestamp_ns','appearance','jaw_half_angle_deg'):
        if record[key] != f[key]: raise ValueError('Rendered state differs from planned clip: '+key)
    for key in ('trajectory_group_id','feature_profile_id','condition_bins','features','clip_start_frame','clip_start_time_s'):
        if record[key] != seq[key]: raise ValueError('Rendered sequence metadata changed: '+key)
    for group in ('psm','ecm'):
        for field,prefix in (('joint_positions','q'),('joint_velocities','v')):
            actual=[record[field][group][name] for name in joint_order[group]]
            if not np.allclose(actual,f[prefix+'_'+group],atol=1e-12,rtol=0):
                raise ValueError('Rendered joints/velocity differ from original parent')


def write_split_indexes(run, sequences, records=(), acceptance_passed=False):
    """Full planned indexes remain explicit about missing images during a smoke run."""
    import json
    from io_utils import write
    folder=run/'indices';folder.mkdir(exist_ok=True)
    available={(r['sequence_id'],r['frame_index']) for r in records if r['reason']=='dataset_frame'}
    if len(available)!=sum(r['reason']=='dataset_frame' for r in records):
        raise ValueError('Repeated full-generation frame receipt')
    inventory={};expected=set()
    for split in SPLITS:
        rows=[]
        for seq in sequences:
            if seq['split']!=split:continue
            for frame in seq['frames']:
                sid=seq['sequence_id'];i=frame['frame_index'];expected.add((sid,i))
                rows.append({'sequence_id':sid,'frame_index':i,'time_s':frame['time_s'],
                    'source_frame_index':frame['source_frame_index'],'source_time_s':frame['source_time_s'],
                    'split':split,'category':seq['category'],'trajectory_group_id':seq['trajectory_group_id'],
                    'feature_profile_id':seq['feature_profile_id'],'phase':frame['phase'],
                    'opening_state':jaw_state(frame['jaw_half_angle_deg']),'condition_bins':seq['condition_bins'],
                    'label':f'sequences/{sid}/labels/{i:06d}.json',
                    'image_left':f'sequences/{sid}/images/left/{i:06d}.png',
                    'image_right':f'sequences/{sid}/images/right/{i:06d}.png',
                    'rendered_and_verified':(sid,i) in available,
                    'category_equal_weight':1./(7*len(seq['frames']))})
        (folder/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        inventory[split]={'planned_pairs':len(rows),'verified_pairs':sum(r['rendered_and_verified'] for r in rows),
                          'sum_category_equal_weights':sum(r['category_equal_weight'] for r in rows)}
    if not available<=expected:raise ValueError('Unexpected frame outside planned split indexes')
    if acceptance_passed and available!=expected:raise ValueError('Full acceptance cannot omit planned frames')
    write(folder/'manifest.json',{'policy':POLICY,'splits':inventory,
        'dataset_complete':bool(acceptance_passed and available==expected),
        'window_rule':'Only consecutive frames within one sequence; maximum common window is 32 frames.',
        'weight_rule':'Each category mean receives 1/7; frame weights are 1/(7*category frame count).',
        'pending_rows':'Planned paths are not training data until rendered_and_verified=true.'})
