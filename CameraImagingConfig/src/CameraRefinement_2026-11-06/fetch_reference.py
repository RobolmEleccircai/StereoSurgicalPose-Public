"""Fetch individual public ZIP members with HTTP Range; never download the full multi-GB dataset."""
import io,json,zipfile,requests,hashlib
from pathlib import Path
root=Path(__file__).parent
url='https://zenodo.org/records/17598453/files/DRENDS_ExVivo_Sequences.zip?download=1'
meta=requests.get('https://zenodo.org/api/records/17598453',timeout=30).json()
size=next(f['size'] for f in meta['files'] if f['key']=='DRENDS_ExVivo_Sequences.zip')
class RangeFile(io.RawIOBase):
 def __init__(self):self.pos=0;self.size=size;self.transferred=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,off,whence=0):self.pos=off if whence==0 else self.pos+off if whence==1 else self.size+off;return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n==0:return b''
  with requests.get(url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=45,stream=True) as r:
   if r.status_code!=206:raise RuntimeError(f'Range not supported: {r.status_code}')
   content=r.content
  if len(content)!=n:raise RuntimeError(f'Unexpected range length {len(content)} expected {n}')
  self.transferred+=n;self.pos+=n;return content
f=RangeFile()
with zipfile.ZipFile(f) as z:
 names=z.namelist();(root/'sources/sequence_archive_index.json').write_text(json.dumps(names,indent=2))
 print('entries',len(names),'index bytes',f.transferred,flush=True)
 choices=[n for n in names if 'Seq01' in n and (n.endswith('/left.mp4') or n.endswith('/right.mp4'))]
 print('matches',len(choices),choices[:8],flush=True)
 if not choices:
  print('\n'.join(names[:65]));raise SystemExit(0)
 folder=root/'sources/tissue_reference';folder.mkdir(exist_ok=True)
 selected=choices;records=[]
 for name in selected:
  info=z.getinfo(name);print('FETCHING',name,'compressed bytes',info.compress_size,flush=True)
  assert info.compress_size<200_000_000
  data=z.read(name);file=folder/Path(name).name;file.write_bytes(data)
  records.append({'archive_entry':name,'file':str(file.relative_to(root)),'sha256':hashlib.sha256(data).hexdigest()})
  print('EXTRACTED',name,len(data),flush=True)
 (folder/'source.json').write_text(json.dumps({'source_url':url,'archive_size':size,'downloaded_bytes':f.transferred,'images':records},indent=2))
