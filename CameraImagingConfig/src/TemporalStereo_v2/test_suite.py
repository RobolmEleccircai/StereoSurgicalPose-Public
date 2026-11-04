"""Failure injection and actual-asset geometry checks; no renderer is started."""
import tempfile,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh
from runtime import arguments,Scene
from io_utils import assert_mode,reserve_render,write,read,sha,frame_valid,commit_label,seal,require_seal
from safety import Safety,make_shape
from paths import Trajectory,jaw_positions
from kinematics import rigid_mm,usd_world,AXES

def rejected(fn):
    try:fn()
    except (ValueError,RuntimeError,FileNotFoundError):return
    raise AssertionError('Expected rejection did not happen')

def main():
    a=arguments();results=[]
    def test(name,fn):
        fn();results.append(name);print('TEST_PASS',name,flush=True)
    test('laptop_rejects_full_generation',lambda:rejected(lambda:assert_mode('generate',{'role':'laptop'})))
    with tempfile.TemporaryDirectory() as temporary:
        p=Path(temporary);machine={'role':'laptop','output_root':str(p)};c={'validation_render_limit':2,'validation_budget_scope':'test'}
        reserve_render(machine,c,'run_a','0');reserve_render(machine,c,'run_a','1')
        test('budget_survives_new_run_id',lambda:rejected(lambda:reserve_render(machine,c,'run_b','0')))
        ledger_path=p/'test.budget.json';ledger=read(ledger_path);original_attempts=copy.deepcopy(ledger['attempts'])
        ledger['additional_authorizations']=[{'id':'explicit-test-extension','run_id':'new_authorized_run','additional_attempts':1}]
        write(ledger_path,ledger)
        test('extension_cannot_be_used_by_other_run',lambda:rejected(lambda:reserve_render(machine,c,'wrong_run','0')))
        reserve_render(machine,c,'new_authorized_run','0')
        assert read(ledger_path)['attempts'][:2]==original_attempts
        test('additional_authorization_is_finite',lambda:rejected(lambda:reserve_render(machine,c,'new_authorized_run','1')))
        image=p/'image.png';mask=p/'mask.png';truth=p/'truth.json';image.write_bytes(b'image');mask.write_bytes(b'mask');write(truth,{'test':True})
        label=p/'label.json';commit_label(label,{'fingerprint':'f','camera_truth':'truth.json','camera_truth_sha256':sha(truth),
            'eyes':{'left':{'image':'image.png','image_sha256':sha(image),'mask':'mask.png','mask_sha256':sha(mask)},'right':{'image':'image.png','image_sha256':sha(image),'mask':'mask.png','mask_sha256':sha(mask)}}})
        assert frame_valid(label,'f');assert not frame_valid(label,'changed-code');mask.write_bytes(b'corrupt');assert not frame_valid(label,'f')
        results.append('resume_requires_all_hashes_and_same_fingerprint')
        trajectory=p/'trajectory.json';write(trajectory,{'q':[1,2,3]});seal(trajectory,'f');require_seal(trajectory,'f')
        write(trajectory,{'q':[1,2,4]});test('changed_trajectory_cache_rejected',lambda:rejected(lambda:require_seal(trajectory,'f')))
        label.write_text('{}');assert not frame_valid(label,'f');results.append('changed_label_commit_rejected')
    outer=trimesh.creation.box(extents=[10,10,10]);inner=trimesh.creation.box(extents=[1,1,1])
    big=make_shape('organ','','anatomy',outer.vertices,outer.faces);small=make_shape('tool','tool','psm',inner.vertices,inner.faces)
    assert Safety.pair(small,big)['reason']=='contained_inside';results.append('contained_tool_rejected_without_surface_intersection')
    T=np.eye(4);T[0,3]=5.25;small.move(T);assert not Safety.pair(small,big)['safe'];results.append('surface_penetration_rejected')
    T[0,3]=5.55;small.move(T);r=Safety.pair(small,big);assert r['safe'] and abs(r['distance_mm']-.05)<1e-6
    results.append('near_zero_gap_contact_accepted')
    small.levers=np.array([1.,1.])
    def move(qp,qe):
        T=np.eye(4);T[0,3]=qp[0];T[1,3]=qp[1];small.move(T)
    fake=SimpleNamespace(scene=SimpleNamespace(c={'max_subdivision_depth':20}),moving=[small],static=[big],adjacent=set(),move=move,pair=Safety.pair)
    fake.sweep=lambda path,a,b,fn,depth=0,cache=None:Safety.sweep(fake,path,a,b,fn,depth,cache)
    path=SimpleNamespace(speed_bounds=lambda a,b:{'psm':np.array([20.,0.])},at=lambda t:(np.array([20*t-10,0.]),None))
    def cross(t):move(*path.at(t));return Safety.pair(small,big)
    assert not fake.sweep(path,0,1,cross)['safe'];results.append('between_frame_crossing_rejected')
    path=SimpleNamespace(speed_bounds=lambda a,b:{'psm':np.array([0.,.1])},at=lambda t:(np.array([5.55,t*.1]),None))
    def slide(t):move(*path.at(t));return Safety.pair(small,big)
    assert fake.sweep(path,0,.01,slide)['safe'];results.append('contact_slide_interval_certified')
    path=SimpleNamespace(speed_bounds=lambda a,b:{'psm':np.array([.1*np.pi,0.])},at=lambda t:(np.array([7.55-.1*np.sin(np.pi*t),0.]),None))
    def near(t):
        move(*path.at(t));r=Safety.pair(small,big)
        return dict(r,safe=r['safe'] and r['distance_mm']>=2.,required_tissue_clearance_mm=2.)
    assert not fake.sweep(path,0,1,near)['safe'];results.append('between_frame_clearance_violation_rejected')
    q=np.array([[0.],[1.]]);curve=Trajectory.create([0,1],q,q)
    assert abs(curve.speed_bounds()['psm'][0]-1.875)<1e-10;results.append('quintic_peak_speed_bound')
    scene=Scene(a.machine_data,a.settings_data);safety=Safety(scene)
    from assets import inspect_assets
    broken=scene.stage.DefinePrim('/MissingReferenceFailureInjection','Xform')
    broken.GetReferences().AddReference('/nonexistent_temporal_v2_test_asset.usda')
    try:inspect_assets(scene.stage,a.machine_data['isaac_root'],a.run/'unused_failure_test_cache')
    except ValueError as error:assert 'USD composition' in str(error)
    else:raise AssertionError('A missing USD reference was accepted despite other valid geometry')
    scene.stage.RemovePrim('/MissingReferenceFailureInjection')
    results.append('missing_usd_reference_rejected_with_other_geometry_present')
    test('wrong_anatomy_root_rejected',lambda:rejected(lambda:Safety(Scene(a.machine_data,dict(a.settings_data,anatomy_root='/missing_anatomy')))))
    test('official_joint_limit_rejected',lambda:rejected(lambda:scene.psm.fk(scene.psm.upper+1)))
    assert len(safety.static)>0 and any(x.group=='ecm' for x in safety.moving);results.append('actual_organ_and_ecm_meshes_loaded')
    jaw=next(x for x in safety.moving if x.body.endswith('/psm_tool_gripper1_link'))
    shaft=next(x for x in safety.moving if x.body.endswith('/psm_tool_roll_link'))
    q=scene.psm.base.copy();safety.move(q,scene.ecm.base);first=safety.cached_pair(jaw,shaft);size=len(safety.pair_cache)
    q[0]+=.1;safety.move(q,scene.ecm.base);cached=safety.cached_pair(jaw,shaft);fresh=safety.pair(jaw,shaft)
    assert len(safety.pair_cache)==size and cached['safe']==fresh['safe'] and abs(cached['distance_mm']-fresh['distance_mm'])<1e-6
    q[scene.psm.names.index('psm_tool_gripper1_joint')]=-.1;safety.move(q,scene.ecm.base)
    cached=safety.cached_pair(jaw,shaft);fresh=safety.pair(jaw,shaft)
    assert len(safety.pair_cache)>size and cached['safe']==fresh['safe'] and abs(cached['distance_mm']-fresh['distance_mm'])<1e-6
    results.append('relative_geometry_cache_reuses_common_motion_and_invalidates_articulation')
    for alpha in (0.,15.,30.):
        qp=jaw_positions(scene.psm.base,scene.psm,np.deg2rad(alpha));scene.psm.fk(qp)
    results.append('official_jaw_endpoints_within_limits')
    times=np.linspace(0,149/60,41)
    initial=np.repeat(scene.psm.base[None,:],len(times),axis=0);ecm=np.repeat(scene.ecm.base[None,:],len(times),axis=0)
    for angles in ((0.,8.),(30.,22.)):
        exact=Trajectory.create(times,initial,ecm,continuous=True).with_jaw_curve(scene.psm,angles)
        exact.validate_bounds(scene)
        index=scene.psm.names.index('psm_tool_gripper2_joint')
        for t in np.linspace(times[0],times[-1],1001):
            value=exact.at(t)[0][index];assert -1e-12<=value<=np.pi/6+1e-12
    results.append('closed_and_fully_open_jaw_curves_never_overshoot_hard_stops')
    qp=scene.psm.base.copy();qe=scene.ecm.base.copy();qp[0]+=.02;qp[2]+=.003;qe[0]+=.01;qe[2]+=.002
    spec={'baseline_scale':1.03,'relative_yaw_delta_deg':.2};truth=scene.cameras(qe,spec);scene.apply(qp,qe,spec)
    for eye in ('left','right'):
        actual=rigid_mm(usd_world(scene.stage,scene.rig+'/'+eye.capitalize()))@AXES
        assert np.max(abs(actual-truth[eye]))<1e-6
    for group,q in [('psm',qp),('ecm',qe)]:
        for body,T in getattr(scene,group).fk(q).items():assert np.max(abs(rigid_mm(usd_world(scene.stage,body))-T))<1e-6
    results.append('authored_psm_ecm_and_camera_poses_match_independent_fk')
    from quality_labels import image_quality,TipRegions
    rgb=np.zeros((10,10,3),np.uint8);mask=np.ones((10,10),np.uint8)
    assert not image_quality(rgb,mask)['overexposed']
    rgb[0,0]=255
    assert image_quality(rgb,mask)['overexposed']
    rgb[0,0]=[255,255,100]
    assert not image_quality(rgb,mask)['overexposed']
    results.append('exposure_threshold_and_all_channel_condition')
    definitions=TipRegions(safety).definitions()
    assert definitions==TipRegions(safety).definitions()
    for region in definitions['jaws']:
        samples=np.array(region['sample_points_body_mm'])
        shape=next(s for s in safety.moving if s.body==region['body'])
        _,dist,_=shape.mesh.nearest.on_surface(samples)
        assert len(samples)==128 and max(dist)<1e-8 and min(samples[:,1])>=region['local_Y_cutoff_mm']-1e-9
    results.append('distal_region_samples_fixed_on_actual_jaw_surface')
    from pose_truth import jaw_definitions,attach_body_truth,validate_body_truth
    from pxr import UsdGeom
    hinges=jaw_definitions(scene.stage,scene.c['psm_root'])
    ground=scene.labels(qp,qe,spec)
    record={'eyes':{eye:{'K':scene.profile[eye]['K'],'D':scene.profile[eye]['D'],'tool':ground[eye]}
                    for eye in ('left','right')}}
    cameras={'T_world_from_optical_mm':{k:v.tolist() for k,v in truth.items()},
             'T_right_from_left_mm':(np.linalg.inv(truth['right'])@truth['left']).tolist()}
    attach_body_truth(record,scene.psm.fk(qp),scene.roll,hinges,cameras,sha(scene.paths['source_scene']))
    validate_body_truth(record,cameras)
    for eye,e in record['eyes'].items():
        for name,frame in e['local_frames'].items():
            actual=np.linalg.inv(truth[eye])@rigid_mm(usd_world(scene.stage,frame['body']))
            assert np.allclose(actual[:3,:3],np.array(frame['T_optical_from_local_mm'])[:3,:3],atol=1e-6,rtol=0)
    results.append('full_local_rotation_truth_matches_authored_USD_in_both_eyes')
    corrupt=copy.deepcopy(record)
    corrupt['eyes']['left']['local_frames']['jaw1']['T_optical_from_local_mm'][0][0]+=.1
    test('corrupt_full_local_rotation_rejected',lambda:rejected(lambda:validate_body_truth(corrupt,cameras)))
    write(a.run/'validation/unit_tests.json',{'passed':True,'tests':results,'rendered_pairs':0})
    print('TESTS_COMPLETE',len(results),'NO_RENDER',flush=True)

if __name__=='__main__':main()
