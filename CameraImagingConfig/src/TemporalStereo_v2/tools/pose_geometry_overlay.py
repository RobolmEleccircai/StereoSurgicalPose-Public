"""Project actual CAD contours onto existing RGB; no simulator or RGB rendering."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from reannotate import sha, project


def prepare(vertices, faces):
    v, inverse = np.unique(np.round(vertices, 7), axis=0, return_inverse=True)
    f = inverse[faces]
    normals = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
    good = np.linalg.norm(normals, axis=1) > 1e-12
    f, normals = f[good], normals[good]
    edges = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(len(f)), 3)
    keys = edges[:, 0]*len(v)+edges[:, 1]
    order = np.argsort(keys)
    _, first, count = np.unique(keys[order], return_index=True, return_counts=True)
    return dict(v=v, f=f, normals=normals, edges=edges[order[first]],
                a=owner[order[first]], b=owner[order[first+count-1]], boundary=count==1)


def project_many(p, e):
    x, y = p[:, 0]/p[:, 2], p[:, 1]/p[:, 2]
    k1, k2, p1, p2, k3 = e['D']; r = x*x+y*y
    radial = 1+k1*r+k2*r*r+k3*r*r*r; K = e['K']
    return np.c_[K[0][0]*(x*radial+2*p1*x*y+p2*(r+2*x*x))+K[0][2],
                 K[1][1]*(y*radial+p1*(r+2*y*y)+2*p2*x*y)+K[1][2]]


def clipped_edges(p):
    # Clip to a conservative camera frustum before distortion projection.
    # This avoids projecting geometry behind the lens or near z=0.
    start, delta = p[:, 0], p[:, 1]-p[:, 0]
    lo, hi = np.zeros(len(p)), np.ones(len(p)); valid = np.ones(len(p), bool)
    for n, offset in [(np.array([0,0,1.]), -.1), (np.array([1.,0,1.5]),0),
                      (np.array([-1.,0,1.5]),0), (np.array([0,1.,1.5]),0), (np.array([0,-1.,1.5]),0)]:
        value, slope = start@n+offset, delta@n
        still = np.abs(slope)<1e-12; valid &= ~(still & (value<0))
        t = np.divide(-value, slope, out=np.zeros_like(value), where=~still)
        lo = np.where(slope>1e-12, np.maximum(lo,t), lo)
        hi = np.where(slope< -1e-12, np.minimum(hi,t), hi)
    valid &= lo<=hi
    return np.stack([start[valid]+lo[valid,None]*delta[valid], start[valid]+hi[valid,None]*delta[valid]], axis=1)


def draw_panel(rgb, mask, label, frame, eye, meshes, geometry, font_path):
    e = label['eyes'][eye]; w,h = rgb.size
    camera = np.linalg.inv(frame['camera_truth']['T_world_from_optical_mm'][eye])
    lines = Image.new('L',(w,h)); pen = ImageDraw.Draw(lines); edge_count = 0
    for mesh, info in zip(meshes, geometry['meshes']):
        T = camera @ frame['T_world_from_body_mm'][info['body']]
        origin = np.linalg.inv(T)[:3,3]
        facing = np.einsum('ij,ij->i', mesh['normals'], origin-mesh['v'][mesh['f'][:,0]]) > 0
        select = mesh['boundary'] | (facing[mesh['a']] != facing[mesh['b']])
        points = mesh['v'][mesh['edges'][select]]
        points = points@T[:3,:3].T+T[:3,3]
        points = clipped_edges(points)
        if not len(points):
            continue
        # Densify edges to preserve the curvature induced by lens distortion.
        t = np.linspace(0,1,17)
        samples = points[:,0,None,:]+t[None,:,None]*(points[:,1,None,:]-points[:,0,None,:])
        uv = project_many(samples.reshape(-1,3),e).reshape(-1,17,2)
        on_screen = (uv[:,:,0].max(1)>=0)&(uv[:,:,0].min(1)<w)&(uv[:,:,1].max(1)>=0)&(uv[:,:,1].min(1)<h)
        for curve in uv[on_screen]:
            pen.line([tuple(x) for x in curve],fill=255,width=1)
        edge_count += int(on_screen.sum())
    line = np.array(lines)>0
    # Use the saved visible-instrument mask for display-region classification.
    # This is a contour visualization, not a new per-facet occlusion annotation.
    visible_region = np.array(mask.filter(ImageFilter.MaxFilter(5)))>0
    visible = line & visible_region
    interior = visible & (np.array(mask.filter(ImageFilter.MinFilter(9)))>0)
    exterior = visible & ~interior
    yy,xx = np.indices((h,w)); hidden = line & ~visible_region & ((xx+yy)%20<11)
    image = np.array(rgb).copy()
    image[interior] = np.rint(.20*np.array([40,235,245])+.80*image[interior]).astype(np.uint8)
    image[exterior] = np.rint(.95*np.array([40,235,245])+.05*image[exterior]).astype(np.uint8)
    image[hidden] = np.rint(.9*np.array([255,90,215])+.1*image[hidden]).astype(np.uint8)
    head,foot = 50,100
    canvas=Image.new('RGB',(w,h+head+foot),'#101820');canvas.paste(Image.fromarray(image),(0,head))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(font_path),25);small=ImageFont.truetype(str(font_path),22)
    title='左目 LEFT' if eye=='left' else '右目 RIGHT'
    draw.text((16,10),f"{title} · 器械几何姿态投影   {label['sequence_id']} / {label['job_id']}",font=font,fill='white')
    g=e['tool'];C=np.array(g['shaft_reference_position_mm']);d=np.array(g['shaft_direction_unit'])
    axis=project_many(C+np.linspace(0,15,61)[:,None]*d,e);axis[:,1]+=head
    draw.line([tuple(p) for p in axis],fill='#171717',width=7)
    draw.line([tuple(p) for p in axis],fill='#ffe956',width=3)
    end=axis[-1];direction=end-axis[-5];direction/=np.linalg.norm(direction)
    perp=np.array([-direction[1],direction[0]])
    draw.polygon([tuple(end),tuple(end-17*direction+7*perp),tuple(end-17*direction-7*perp)],fill='#ffe956')
    # Named leaders only: no circle or crosshair substitutes for tool geometry.
    def name(text, point, offset, color):
        q=np.array([point[0],point[1]+head]);bbox=draw.textbbox((0,0),text,font=font)
        bw=bbox[2]-bbox[0]+18;bh=42
        x=max(10,min(w-bw-10,q[0]+offset[0]));y=max(head+10,min(head+h-bh-10,q[1]+offset[1]))
        near=(max(x,min(x+bw,q[0])),max(y,min(y+bh,q[1])))
        draw.line((tuple(q),near),fill=color,width=2)
        draw.rounded_rectangle((x,y,x+bw,y+bh),radius=5,fill='#101820',outline=color,width=1)
        draw.text((x+9,y+6),text,font=font,fill=color)
    name('C / d 杆轴',g['shaft_reference_pixel'],(55,90),'#ffe956')
    name('T 两尖端中点',g['tip_pixel'],(-300,-100),'#ff9de2')
    draw.text((16,head+h+10),'青色：器械 CAD 投影（内部线框半透明）    紫色虚线：遮挡区域投影    黄色：从 C 出发的杆轴 d',font=small,fill='white')
    draw.text((16,head+h+47),'使用同一帧的原 CAD、关节与相机真值；不是模型预测。原始 RGB 保持不变，未重新渲染。',font=small,fill='#c4ced8')
    return canvas,dict(projected_edges=edge_count,visible_region_line_pixels=int(visible.sum()),hidden_region_line_pixels=int((line&~visible_region).sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--font',type=Path,required=True)
    a=p.parse_args();geometry=json.loads((a.geometry/'geometry.json').read_text())
    if sha(a.geometry/'meshes.npz')!=geometry['meshes_sha256']:
        raise ValueError('Geometry checksum mismatch')
    archive=np.load(a.geometry/'meshes.npz');meshes=[prepare(archive[m['key']+'_vertices'],archive[m['key']+'_faces']) for m in geometry['meshes']]
    source_hashes={str(f.relative_to(a.run)):sha(f) for f in (a.run/'sequences').rglob('*') if f.is_file()}
    a.output.mkdir(parents=True,exist_ok=True);results=[]
    for frame in geometry['frames']:
        seq,job=frame['sequence_id'],frame['job_id'];path=a.run/'sequences'/seq/'labels'/f'{job}.json'
        if sha(path)!=frame['label_sha256']:raise ValueError('Label changed since geometry export')
        label=json.loads(path.read_text());panels=[];stats={}
        for eye in ('left','right'):
            e=label['eyes'][eye];source=path.parent/e['image'];mask=path.parent/e['mask']
            if sha(source)!=e['image_sha256'] or sha(mask)!=e['mask_sha256']:raise ValueError('Original image/mask changed')
            panel,stats[eye]=draw_panel(Image.open(source).convert('RGB'),Image.open(mask).convert('L'),label,frame,eye,meshes,geometry,a.font)
            panels.append(panel)
        w,h=panels[0].size;canvas=Image.new('RGB',(2*w,h))
        canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(w,0))
        target=a.output/f'{seq}_{job}.jpg';canvas.save(target,quality=95,subsampling=0)
        results.append({'file':target.name,'sha256':sha(target),'statistics':stats})
    after={str(f.relative_to(a.run)):sha(f) for f in (a.run/'sequences').rglob('*') if f.is_file()}
    if after!=source_hashes:raise RuntimeError('Original data modified')
    report={'new_rgb_renders':0,'simulation_app_started':False,'source_files_unchanged':True,
            'source_file_count':len(source_hashes),'pairs':len(results),'geometry_sha256':sha(a.geometry/'geometry.json'),
            'annotation_code_sha256':sha(Path(__file__)),'source':'actual_CAD_and_saved_full_joint_truth',
            'visibility_display':'Projected contour classified by saved visible-tool mask dilated by 2 pixels; not a new per-facet visibility label. Interior line alpha 0.20; boundary-region alpha 0.95, using a 4-pixel eroded mask only to set opacity, never to move or synthesize projected edges.',
            'files':results}
    (a.output/'overlay_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('new_rgb_renders','source_files_unchanged','source_file_count','pairs')}))


if __name__=='__main__':main()
