"""Audit composed rendering inputs; ignore editor icons and shadowed legacy assets."""
from pathlib import Path
import glob,re,urllib.request,urllib.parse,hashlib
from pxr import Usd,UsdGeom,UsdShade,Sdf
from io_utils import sha

def inspect_assets(stage,isaac_root,cache):
    errors=[]
    for prim in stage.Traverse():
        index=prim.GetPrimIndex();errors.extend(index.localErrors)
        if index.rootNode:errors.extend(index.rootNode.layerStack.localErrors)
    if errors:raise ValueError('USD composition contains unresolved references or payloads: '+str(errors))
    files={Path(l.realPath).resolve() for l in stage.GetUsedLayers() if l.realPath};todo=[];seen=set();overrides={};records=[]
    mdl_index={}
    for p in Path(isaac_root).rglob('*.mdl'):
        mdl_index.setdefault(p.name,[]).append(p)
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    fetched=set()
    inspected_local=set()
    def inspect_local_mdl(path):
        path=Path(path).resolve()
        if path in inspected_local or path.is_relative_to(Path(isaac_root).resolve()):return
        inspected_local.add(path)
        for relative in re.findall(r'"([^"\n]+\.(?:png|jpg|jpeg|dds|exr|tif|mdl))"',path.read_text(),re.I):
            if '.thumbs/' in relative:continue
            if relative.startswith(('http://','https://')):fetch(relative);continue
            target=(path.parent/relative).resolve()
            if not target.is_file():raise FileNotFoundError('Missing custom MDL dependency: '+str(target))
            files.add(target)
            if target.suffix=='.mdl':inspect_local_mdl(target)
    def fetch(url):
        url=url.replace('http://','https://',1);parts=urllib.parse.urlparse(url)
        target=cache/parts.netloc/parts.path.lstrip('/')
        if '..' in target.parts:raise ValueError('Unsafe dependency path')
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with urllib.request.urlopen(url,timeout=45) as response:data=response.read()
            temp=target.with_name(target.name+'.tmp');temp.write_bytes(data);temp.replace(target)
        files.add(target)
        if url in fetched:return target
        fetched.add(url)
        if target.suffix=='.mdl':
            text=target.read_text()
            for relative in re.findall(r'"([^"\n]+\.(?:png|jpg|jpeg|dds|exr|tif|mdl))"',text,re.I):
                if '.thumbs/' in relative:continue # material-browser preview, not a rendering input
                if relative.startswith('http'):fetch(relative)
                else:fetch(urllib.parse.urljoin(url,relative))
            for module in re.findall(r'import\s+::([\w:]+)::\*',text):
                head=module.split('::')[0]
                if head in ('anno','base','df','math','state','tex','limits','scene','debug'):continue
                module_path=module.replace('::','/')+'.mdl'
                if head=='vMaterials_2':
                    prefix=url.split('/Materials/')[0]+'/Materials/'
                    fetch(prefix+module_path)
                elif Path(module_path).name in mdl_index:
                    files.add(sorted(mdl_index[Path(module_path).name])[0])
                else:raise FileNotFoundError('Unresolved MDL import '+module)
        return target
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputeVisibility()!='invisible':
            mat,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            if mat:
                for out in mat.GetOutputs():todo.extend(str(x.GetPrimPath()) for x in out.GetAttr().GetConnections())
    while todo:
        path=todo.pop()
        if path in seen:continue
        seen.add(path);p=stage.GetPrimAtPath(path)
        for a in p.GetAttributes():
            if a.GetName().startswith('ui:'):continue
            todo.extend(str(x.GetPrimPath()) for x in a.GetConnections())
            value=a.Get()
            if not isinstance(value,Sdf.AssetPath) or not value.path:continue
            original=value.path;resolved=value.resolvedPath;matches=[]
            if resolved and Path(resolved).is_file():matches=[Path(resolved)]
            elif original.startswith(('http://','https://')):
                target=fetch(original);matches=[target];overrides[str(a.GetPath())]=str(target)
            elif original.endswith('.mdl') and Path(original).name in mdl_index:
                candidates=mdl_index[Path(original).name]
                exact=[p for p in candidates if str(p).endswith(original)]
                matches=[sorted(exact or candidates)[0]]
            else:
                stack=a.GetPropertyStack();author=next((x.layer for x in stack if x.HasDefaultValue()),None)
                if author:
                    anchored=Sdf.ComputeAssetPathRelativeToLayer(author,original)
                    pattern=anchored.replace('<UDIM>','[1-9][0-9][0-9][0-9]')
                    matches=[Path(p) for p in glob.glob(pattern) if Path(p).is_file()]
            if not matches:raise FileNotFoundError('Unresolved active rendering asset: '+str(a.GetPath())+' = '+original)
            files.update(p.resolve() for p in matches)
            for match in matches:
                if match.suffix=='.mdl':inspect_local_mdl(match)
            records.append({'attribute':str(a.GetPath()),'authored':original,'resolved_files':[str(p) for p in matches]})
    if not records:raise ValueError('No material assets found')
    return files,records,overrides
