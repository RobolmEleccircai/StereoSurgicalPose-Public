"""Image-only, ROI-assisted jaw-front/ring extraction for one requested pair.

The ROIs are manually specified from the displayed images. This is not an
automatic detector for unseen tools. No camera extrinsics or CAD projections
are read here.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.optimize import least_squares
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
settings = {
    'left': {'jaw_roi': [1185, 617, 1260, 679], 'ring_seed': [1316.0, 651.0]},
    'right': {'jaw_roi': [1278, 610, 1355, 674], 'ring_seed': [1403.0, 643.0]},
}
results = {}
canvas = Image.new('RGB', (1600, 520), '#142033')
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)

for eye_index, (eye, cfg) in enumerate(settings.items()):
    im = cv2.imread(str(root/'input_images'/eye/'00.png'))
    x0,y0,x1,y1 = cfg['jaw_roi']
    patch = im[y0:y1,x0:x1].astype(float)
    # Gray-blue metal against red tissue; the tip edge is measured from this mask.
    metric = patch[:,:,0] - patch[:,:,2]
    mask = np.uint8(metric > -8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    label = 1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    mask = np.uint8(labels==label)*255
    ys,xs = np.where(mask>0)
    # Fit the distal end edge in the middle of its visible span; avoid corners.
    near = xs < xs.min()+8
    low,high=np.percentile(ys[near],[10,90])
    rows=np.arange(int(np.ceil(low+0.15*(high-low))),int(np.floor(high-0.15*(high-low)))+1)
    edge=[]
    for y in rows:
        xx=np.flatnonzero(mask[y]>0)
        if len(xx):
            x=int(xx.min())
            if x>0 and metric[y,x]!=metric[y,x-1]:
                x=x-1+(-8-metric[y,x-1])/(metric[y,x]-metric[y,x-1])
            edge.append([float(x+x0),float(y+y0)])
    edge=np.array(edge)
    coef=np.polyfit(edge[:,1],edge[:,0],1)
    av=float((low+high)/2+y0)
    A=np.array([np.polyval(coef,av),av])

    # Outer ring boundary from image gradients; ellipse parameters are local fits.
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray=cv2.GaussianBlur(gray,(0,0),0.7)
    sx,sy=cfg['ring_seed']
    xx,yy=np.meshgrid(np.arange(int(sx)-20,int(sx)+21),np.arange(int(sy)-20,int(sy)+21))
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag=np.hypot(gx,gy)
    theta=np.linspace(0,2*np.pi,120,endpoint=False)
    rad=np.linspace(9,16,141)
    px=(sx+np.cos(theta)[:,None]*rad).astype(np.float32)
    py=(sy+np.sin(theta)[:,None]*rad).astype(np.float32)
    samples=cv2.remap(mag,px,py,cv2.INTER_LINEAR)
    ix=samples.argmax(1)
    boundary=np.column_stack([px[np.arange(len(theta)),ix],py[np.arange(len(theta)),ix]])
    strength=samples[np.arange(len(theta)),ix]
    boundary=boundary[strength>np.percentile(strength,20)]
    def err(p):
        cx,cy,rx,ry,a=p
        v=boundary-[cx,cy]
        rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        v=v@rot
        return (np.sqrt((v[:,0]/rx)**2+(v[:,1]/ry)**2)-1)*np.sqrt(rx*ry)
    fit=least_squares(err,[sx,sy,12,13,0],bounds=([sx-4,sy-4,8,8,-1.57],[sx+4,sy+4,18,18,1.57]),loss='soft_l1',f_scale=.65)
    B=fit.x[:2]
    results[eye]={'A':A.tolist(),'B':B.tolist(),'pixel_distance':float(np.linalg.norm(B-A)),
        'jaw_edge_points':edge.tolist(),'jaw_mask_threshold_B_minus_R':-8,
        'ring_ellipse':fit.x.tolist(),'ring_fit_median_residual_px':float(np.median(abs(err(fit.x)))),
        'manual_roi_and_seed':cfg}
    cropbox=(1160,590,1450,730) if eye=='left' else (1255,580,1545,720)
    cx,cy,cx1,cy1=cropbox
    crop=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop(cropbox).resize((783,378))
    d=ImageDraw.Draw(crop)
    scale=783/(cx1-cx)
    def uv(p): return ((p[0]-cx)*scale,(p[1]-cy)*scale)
    for p in edge: d.ellipse((uv(p)[0]-1,uv(p)[1]-1,uv(p)[0]+1,uv(p)[1]+1),fill='#00ff60')
    for p in boundary: d.ellipse((uv(p)[0]-1,uv(p)[1]-1,uv(p)[0]+1,uv(p)[1]+1),fill='#00ff60')
    d.line([uv(A),uv(B)],fill='#00eaff',width=3)
    for label,p in [('A',A),('B',B)]:
        u,v=uv(p);d.ellipse((u-5,v-5,u+5,v+5),outline='#fff000',width=2);d.text((u+8,v+8),label,font=font,fill='#fff000')
    canvas.paste(crop,(eye_index*800+8,55))
    draw.text((eye_index*800+12,12),f'{eye.upper()}  A-B = {results[eye]["pixel_distance"]:.2f} px',font=font,fill='white')
    draw.text((eye_index*800+12,448),f'A=({A[0]:.2f}, {A[1]:.2f})   B=({B[0]:.2f}, {B[1]:.2f})',font=font,fill='white')

(root/'points_detected.json').write_text(json.dumps({'method':'ROI-assisted color edge midpoint and local ring ellipse fit; image-only coordinates, not projected ground truth','fully_automatic':False,'eyes':results},indent=2))
canvas.save(root/'measured_AB.png')
print(json.dumps({e:{k:v for k,v in p.items() if k in ['A','B','pixel_distance','ring_fit_median_residual_px']} for e,p in results.items()},indent=2))
