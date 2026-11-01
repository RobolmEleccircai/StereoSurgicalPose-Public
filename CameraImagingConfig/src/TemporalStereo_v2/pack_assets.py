"""Create a new relocatable scene bundle without editing original USD or textures.

Run with Isaac Sim Python and its USD extension on PYTHONPATH, as configured by generate.py.
"""
import argparse,hashlib,shutil
from pathlib import Path
from pxr import Usd,Sdf
from runtime import dependency_audit,source_paths
from io_utils import read,write,load_machine,sha,HERE

def main():
    p=argparse.ArgumentParser();p.add_argument('--machine',type=Path,required=True);p.add_argument('--settings',type=Path,default=HERE/'settings.json');p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();m=load_machine(a.machine);c=read(a.settings);out=a.destination.resolve()
    out.mkdir(parents=True,exist_ok=False);inventory=dependency_audit(m,c);sources=source_paths(m,c)
    source=Usd.Stage.Open(str(sources['source_scene']));flat=Usd.Stage.Open(source.Flatten());copied={}
    for record in inventory['active_asset_attributes']:
        authored=record['authored'];files=[Path(p) for p in record['resolved_files']]
        if authored.endswith('.mdl') and all(f.resolve().is_relative_to(Path(m['isaac_root']).resolve()) for f in files):
            # SDK modules are supplied by the pinned Isaac Sim installation.
            continue
        if authored.startswith(('http://','https://')):
            cache=Path(m['output_root'])/'asset_cache';relative=files[0].relative_to(cache)
            src_root=cache/relative.parts[0];dst_root=out/'network'/relative.parts[0]
            if not dst_root.exists():shutil.copytree(src_root,dst_root)
            target=out/'network'/relative
            replacement='./'+str(target.relative_to(out))
        else:
            parents={f.parent for f in files}
            if len(parents)!=1:raise ValueError('UDIM tiles must share an asset directory')
            folder=out/'assets'/hashlib.sha256(str(files[0].parent).encode()).hexdigest()[:16];folder.mkdir(parents=True,exist_ok=True)
            if files[0].suffix=='.mdl':shutil.copytree(files[0].parent,folder,dirs_exist_ok=True)
            for original in files:
                target=folder/original.name
                if not target.exists():shutil.copy2(original,target)
                copied[str(target.relative_to(out))]=sha(target)
            replacement='./'+str(folder.relative_to(out))+'/'+(Path(authored).name if '<UDIM>' in authored else files[0].name)
        flat.GetAttributeAtPath(record['attribute']).Set(Sdf.AssetPath(replacement))
    flat.GetRootLayer().Export(str(out/'scene.usdc'))
    shutil.copy2(sources['calibration'],out/'calibration.json');shutil.copy2(sources['appearance'],out/'appearance.json')
    settings=dict(c,source_scene='scene.usdc',calibration='calibration.json',appearance='appearance.json')
    write(out/'settings.json',settings)
    # Open the exported scene and verify composed asset paths resolve after relocation.
    probe=dict(m,asset_root=str(out));check=dependency_audit(probe,settings)
    write(out/'bundle_manifest.json',{'source_inventory':inventory,'bundle_files':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
        'dependencies_after_export':check,'required_isaac_version':m['expected_isaac_version']})
    print('ASSET_BUNDLE_COMPLETE',out,flush=True)

if __name__=='__main__':main()
