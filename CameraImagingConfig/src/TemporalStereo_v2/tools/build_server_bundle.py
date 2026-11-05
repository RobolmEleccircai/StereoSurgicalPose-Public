"""Create a new balanced release using verified assets from an older package.

Copies bytes only; does not import Isaac Sim, alter the old bundle, or start jobs.
"""
import argparse,shutil,sys,tarfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from io_utils import read,write,sha,source_hash,validate_settings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-bundle',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--source-commit',default='working-tree-see-source-hash')
    a=p.parse_args();base=a.base_bundle.resolve();dest=a.destination.resolve()
    if dest.exists() or dest.is_relative_to(base):raise ValueError('Use a new directory outside the old package')
    manifest=read(base/'assets/bundle_manifest.json')
    for relative,checksum in manifest['bundle_files'].items():
        path=(base/'assets'/relative).resolve()
        if not path.is_relative_to(base/'assets') or sha(path)!=checksum:raise ValueError('Old asset bundle changed: '+relative)
    c=read(ROOT/'settings.balanced21.json');validate_settings(c)
    old=read(base/'assets/settings.json')
    for key in ('width','height','fps','renderer','samples_per_pixel_per_frame','rt_subframes',
                'global_linear_multiplier','vignette_minimum_linear_gain','native_camera','jaw_speed_rad_s','jaw_half_angle_deg'):
        if c[key]!=old[key]:raise ValueError('Approved capture parameter changed: '+key)
    dest.mkdir(parents=True)
    for name in ('assets','fonts','wheelhouse'):
        shutil.copytree(base/name,dest/name)
    shutil.copytree(ROOT,dest/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.local.json','*.log',
        'runs','.deps','server_assets*','.git','checks','outputs'))
    (dest/'config').mkdir();shutil.copy2(base/'config/machine.server.example.json',dest/'config/machine.server.example.json')
    shutil.copy2(ROOT/'SERVER_MIGRATION.md',dest/'SERVER_MIGRATION.md')
    c.update(source_scene='scene.usdc',calibration='calibration.json',appearance='appearance.json')
    write(dest/'assets/settings.json',c)
    manifest['previous_bundle_manifest_sha256']=sha(base/'assets/bundle_manifest.json')
    manifest['bundle_files']={str(f.relative_to(dest/'assets')):sha(f) for f in sorted((dest/'assets').rglob('*'))
                              if f.is_file() and f.name!='bundle_manifest.json'}
    write(dest/'assets/bundle_manifest.json',manifest)
    write(dest/'release.json',{'source_commit':a.source_commit,'code_sha256':source_hash(),
        'dataset_policy':c['dataset_policy'],'expected_pairs':2100,'expected_sequences':21,
        'split_pairs':{'train':1400,'validation':350,'test':350},
        'previous_release_sha256':sha(base/'release.json'),'new_rgb_renders_during_packaging':0,
        'full_generation_started':False,'server_validation_pending':True,
        'omitted':['Isaac Sim installation','machine credentials','generated datasets','runtime caches']})
    (dest/'MANIFEST.sha256').write_text(''.join(sha(f)+'  '+str(f.relative_to(dest))+'\n'
        for f in sorted(dest.rglob('*')) if f.is_file() and f.name!='MANIFEST.sha256'))
    archive=dest.with_suffix('.tar.gz')
    if archive.exists():raise ValueError('Archive already exists')
    with tarfile.open(archive,'w:gz') as tar:tar.add(dest,arcname=dest.name)
    archive.with_name(archive.name+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(archive,sha(archive),archive.stat().st_size,flush=True)


if __name__=='__main__':main()
