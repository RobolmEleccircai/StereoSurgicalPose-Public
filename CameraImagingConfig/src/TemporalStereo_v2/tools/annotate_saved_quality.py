"""Derive separate visibility/quality sidecars and previews from saved RGB; no renderer."""
import argparse,os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from io_utils import read,write,sha,load_machine,frame_valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('machine','settings','run','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--worker',action='store_true');a=p.parse_args();m=load_machine(a.machine)
    if not a.worker:
        isaac=Path(m['isaac_root']);lib=list((isaac/'extscache').glob('omni.usd.libs-*'))
        if len(lib)!=1:raise ValueError('Expected one USD library')
        env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
        env['PYTHONPATH']=os.pathsep.join([str(lib[0]),m['deps_root'],str(ROOT),env.get('PYTHONPATH','')])
        env['LD_LIBRARY_PATH']=os.pathsep.join([str(lib[0]/'bin'),env.get('LD_LIBRARY_PATH','')])
        subprocess.run([str(isaac/'python.sh'),str(Path(__file__)),*sys.argv[1:],'--worker'],env=env,check=True);return
    import numpy as np
    from PIL import Image
    from runtime import Scene
    from safety import Safety
    from quality_labels import TipRegions,image_quality,validate_quality_fields
    from pose_truth import attach_body_truth,jaw_definitions
    from live_preview import LivePreview,preview_font
    if a.output.resolve().is_relative_to((a.run/'sequences').resolve()):raise ValueError('Do not overwrite raw data')
    before={str(p):sha(p) for p in (a.run/'sequences').rglob('*') if p.is_file()}
    c=read(a.settings);scene=Scene(m,c);safety=Safety(scene);regions=TipRegions(safety,c.get('tip_region'));hinges=jaw_definitions(scene.stage,c['psm_root'])
    publisher=LivePreview(a.output/'previews',preview_font(m));write(a.output/'tip_regions.json',regions.definitions());summary=[]
    scene_hash=sha(scene.paths['source_scene'])
    for lp in sorted(a.run.glob('sequences/*/labels/*.json')):
        r=read(lp)
        if not frame_valid(lp,r['fingerprint']):raise ValueError('Source pair corrupt')
        qp=np.array([r['joint_positions']['psm'][n] for n in scene.psm.names]);qe=np.array([r['joint_positions']['ecm'][n] for n in scene.ecm.names])
        safety.move(qp,qe);truth=read(lp.parent/r['camera_truth']);images={}
        for eye,e in r['eyes'].items():
            with Image.open(lp.parent/e['image']) as im:rgb=np.array(im.convert('RGB'))
            with Image.open(lp.parent/e['mask']) as im:mask=np.array(im)
            e['fixed_tip_visibility']=e['jaw_visibility']
            e['jaw_tip_region_visibility']=regions.evaluate(truth['T_world_from_optical_mm'][eye],scene.profile,eye,mask)
            e['image_quality']=image_quality(rgb,mask,c.get('overexposure'));images[eye]=Image.fromarray(rgb)
            for region in e['jaw_tip_region_visibility']:
                if region['region_visible'] and region['mask_supported_visible_sample_count']==0:raise ValueError('Region projection has no visible-mask support')
        r['quality_flags']={'overexposed_any_eye':any(e['image_quality']['overexposed'] for e in r['eyes'].values())};validate_quality_fields(r)
        attach_body_truth(r,scene.psm.fk(qp),scene.roll,hinges,truth,scene_hash)
        sidecar={'source_label':str(lp.resolve()),'source_label_sha256':sha(lp),'source_scene_sha256':scene_hash,
            'derivation_code_sha256':sha(ROOT/'quality_labels.py'),'new_rgb_renders':0,'quality_flags':r['quality_flags'],
            'eyes':{eye:{k:e[k] for k in ('fixed_tip_visibility','jaw_tip_region_visibility','image_quality')} for eye,e in r['eyes'].items()}}
        write(a.output/'labels'/f"{r['sequence_id']}_{r['job_id']}.json",sidecar)
        publisher.publish(r,truth,images,sha(lp))
        summary.append({'sequence_id':r['sequence_id'],'job_id':r['job_id'],'category':r['category'],
            'fixed_visible':sum(v['visible'] for e in r['eyes'].values() for v in e['fixed_tip_visibility']),
            'region_visible':sum(v['region_visible'] for e in r['eyes'].values() for v in e['jaw_tip_region_visibility']),
            'overexposed_any_eye':r['quality_flags']['overexposed_any_eye']})
        print('QUALITY_PAIR',len(summary),r['sequence_id'],r['job_id'],summary[-1],flush=True)
    if before!={str(p):sha(p) for p in (a.run/'sequences').rglob('*') if p.is_file()}:raise ValueError('Source dataset changed')
    report={'passed':True,'pairs':len(summary),'observations':4*len(summary),'fixed_point_visible':sum(r['fixed_visible'] for r in summary),
            'tip_region_visible':sum(r['region_visible'] for r in summary),'overexposed_pairs':sum(r['overexposed_any_eye'] for r in summary),
            'source_files_unchanged':len(before),'new_rgb_renders':0,'records':summary}
    write(a.output/'report.json',report);write(a.output/'overexposed_frames.json',[r for r in summary if r['overexposed_any_eye']])
    print('QUALITY_COMPLETE',report,flush=True)


if __name__=='__main__':main()
