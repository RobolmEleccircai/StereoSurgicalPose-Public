"""First completed pair, using the same approved final-image processing."""
from pathlib import Path
import json
import argparse
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--sample',default='0000');sid=p.parse_args().sample
r=Path(__file__).resolve().parent;n=r/'render/native';out=r/'progress';out.mkdir(exist_ok=True)
cfg=json.loads((n/'config/initial_parameters.json').read_text());intr=json.loads((n/'config/camera_intrinsics.json').read_text())
gt=json.loads((n/f'ground_truth/{sid}.json').read_text())['tools']['right_tool']
v=json.loads((r.parent/'SingleInstrument_2026-11-07/preview_final/config/dark_border_parameters.json').read_text())
yy,xx=np.indices((1080,1920),dtype=np.float64);cx,cy=v['center_pixel'];rx,ry=v['semiaxes_pixel'];radius=np.sqrt(((xx-cx)/rx)**2+((yy-cy)/ry)**2)
par=v['profiles']['border_3p5_global075'];t=np.clip((radius-par['start_radius'])/(par['end_radius']-par['start_radius']),0,1);gain=(1-(1-par['outer_linear_gain'])*t*t*(3-2*t)).astype(np.float32)
pair=Image.new('RGB',(1600,448),'#142033');pd=ImageDraw.Draw(pair);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
for ei,eye in enumerate(['left','right']):
    raw=np.load(n/f'linear/{sid}_{eye}.npy').astype(np.float32)
    cy=intr[eye]['K'][1][2];cx=intr[eye]['K'][0][2];yy,xx=np.indices((1080,1920),dtype=np.float32)
    far=max(np.hypot(px-cx,py-cy) for px,py in [(0,0),(1919,0),(0,1079),(1919,1079)])
    radius=np.clip(np.hypot(xx-cx,yy-cy)/far,0,1);sigma=np.sqrt(.3**2+(.8*radius*radius)**2)
    levels=np.linspace(float(sigma.min()),float(sigma.max()),17);step=levels[1]-levels[0];blur=np.zeros_like(raw,dtype=np.float32)
    for s in levels:blur+=np.maximum(1-np.abs(sigma-s)/step,0).astype(np.float32)[:,:,None]*cv2.GaussianBlur(raw,(0,0),float(s),borderType=cv2.BORDER_REFLECT_101)
    base=blur*np.asarray(cfg['illumination']['combined_rgb_scales'][eye],dtype=np.float32)
    atten=base*gain[:,:,None]*float(v['global_linear_multiplier'])
    encoded=np.clip(np.where(atten<=.0031308,12.92*atten,1.055*np.maximum(atten,0)**(1/2.4)-.055),0,1)
    y=encoded@np.array([.2126,.7152,.0722],dtype=np.float32);rng=np.random.default_rng(int(cfg['noise']['seed_base'])+int(sid)*2+ei)
    pixel=rng.standard_normal(encoded.shape,dtype=np.float32)*np.sqrt(1+6*y)[:,:,None];row=rng.normal(0,.3,(1080,1,1)).astype(np.float32)
    im=Image.fromarray(np.clip(np.rint(encoded*255+pixel+row),0,255).astype(np.uint8));im.save(out/f'{sid}_{eye}.png')
    draw=ImageDraw.Draw(im);tool=gt[eye];line=[tuple(p) for p in tool['shaft_polyline_pixels_raw']]
    draw.line(line,fill='black',width=7);draw.line(line,fill=(255,225,35),width=3)
    for key,col,letter in [('shaft_reference_pixel_raw',(25,235,255),'C'),('tip_pixel_raw',(255,75,170),'T')]:
        x,y=tool[key];draw.ellipse((x-6,y-6,x+6,y+6),outline=col,width=3);draw.text((x-10,y-40),letter,font=font,fill=col,stroke_width=2,stroke_fill='black')
    pts=np.array(line+[tool['shaft_reference_pixel_raw'],tool['tip_pixel_raw']]);cx,cy=(pts.min(0)+pts.max(0))/2
    x0=int(np.clip(round(cx)-400,0,1120));y0=int(np.clip(round(cy)-200,0,680))
    pair.paste(im.crop((x0,y0,x0+800,y0+400)),(ei*800,48));pd.text((ei*800+18,11),sid+' / '+eye.upper(),font=font,fill='white')
pair.save(out/f'{sid}_detail.jpg',quality=96,subsampling=0)
print('PAIR_READY',sid)
