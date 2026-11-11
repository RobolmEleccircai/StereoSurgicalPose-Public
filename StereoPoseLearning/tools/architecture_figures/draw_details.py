"""Editable vector figures for the code-audited stereo_pose_unified_c_v1 model.

All positions are shared by the SVG and PDF backends. No generative imagery.
"""
from pathlib import Path
import argparse, math, html, json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import toColor as HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth

PROJECT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=PROJECT/'docs/figures/model')
parser.add_argument('--audit', type=Path, default=PROJECT/'docs/evidence/model_shape_audit_20261112.json')
args = parser.parse_args()
OUT=args.output.resolve(); OUT.mkdir(parents=True,exist_ok=True)
W,H=2560,2220
INK='#243343';MUTED='#596979';LINE='#586979';BORDER='#ccd5df'
BLUE='#2863a1';BLUE_BG='#edf4fc';GREEN='#27765d';GREEN_BG='#edf7f1'
ORANGE='#b36b26';ORANGE_BG='#fff5e8';PURPLE='#72569c';PURPLE_BG='#f4f0fa'
GRAY_BG='#f6f8fa';RED='#a04c58'
SCALE=.4
PDF=canvas.Canvas(str(OUT/'detailed_structure.pdf'),pagesize=(W*SCALE,H*SCALE))
PDF.setTitle('StereoPoseNet: current stereo-temporal articulated pose architecture')
PDF.setAuthor('Code-audited architecture, stereo_pose_unified_c_v1')

class Figure:
    def __init__(self,name):
        self.name=name;self.items=[];self.text_bounds=[]
        self.items.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
        self.items.append('<rect width="100%" height="100%" fill="white"/>')
        PDF.saveState();PDF.scale(SCALE,SCALE);PDF.setFillColor(HexColor('#ffffff'));PDF.rect(0,0,W,H,fill=1,stroke=0)
    def rect(self,x,y,w,h,fill='white',stroke=BORDER,r=10,lw=1.8,dash=False):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
        PDF.setFillColor(HexColor(fill));PDF.setStrokeColor(HexColor(stroke));PDF.setLineWidth(lw);PDF.setDash([8,6] if dash else [])
        PDF.roundRect(x,H-y-h,w,h,r,fill=1,stroke=1);PDF.setDash([])
    def text(self,x,y,s,size=23,bold=False,color=INK,anchor='start',font=None):
        f=font or ('Helvetica-Bold' if bold else 'Helvetica')
        width=stringWidth(s,f,size)
        xx=x if anchor=='start' else x-width/2 if anchor=='middle' else x-width
        if xx<0 or xx+width>W or y-size<0 or y>H:raise ValueError(('out of page',s,x,y,width))
        self.text_bounds.append((xx,y-size,width,size,s))
        self.items.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{html.escape(s)}</text>')
        PDF.setFillColor(HexColor(color));PDF.setFont(f,size);PDF.drawString(xx,H-y,s)
    def lines(self,x,y,lines,size=22,gap=29,anchor='start',color=MUTED):
        for i,line in enumerate(lines):self.text(x,y+i*gap,line,size,color=color,anchor=anchor)
    def line(self,pts,color=LINE,lw=2.6,arrow=True,dash=False):
        self.items.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in pts)+f'" fill="none" stroke="{color}" stroke-width="{lw}" stroke-linejoin="round"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
        PDF.setStrokeColor(HexColor(color));PDF.setLineWidth(lw);PDF.setDash([8,6] if dash else []);p=PDF.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
        for x,y in pts[1:]:p.lineTo(x,H-y)
        PDF.drawPath(p);PDF.setDash([])
        if arrow:
            x,y=pts[-1];a,b=pts[-2];theta=math.atan2(y-b,x-a);length=11;width=5
            verts=[(x,y),(x-length*math.cos(theta)+width*math.sin(theta),y-length*math.sin(theta)-width*math.cos(theta)),(x-length*math.cos(theta)-width*math.sin(theta),y-length*math.sin(theta)+width*math.cos(theta))]
            self.items.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in verts)+f'" fill="{color}"/>')
            p=PDF.beginPath();p.moveTo(verts[0][0],H-verts[0][1])
            for a,b in verts[1:]:p.lineTo(a,H-b)
            p.close();PDF.setFillColor(HexColor(color));PDF.drawPath(p,fill=1,stroke=0)
    def box(self,x,y,w,h,title,body=(),color=BLUE,fill='white',ts=25,fs=21):
        if stringWidth(title,'Helvetica-Bold',ts)>w-22:raise ValueError(('title too wide',title,w,ts))
        self.rect(x,y,w,h,fill=fill,stroke=color)
        self.text(x+w/2,y+34,title,ts,True,color,anchor='middle')
        for i,line in enumerate(body):
            if stringWidth(line,'Helvetica',fs)>w-22:raise ValueError(('text too wide',title,line,w))
            self.text(x+w/2,y+65+i*27,line,fs,color=INK,anchor='middle')
    def panel(self,x,y,w,h,title,color,fill):
        self.rect(x,y,w,h,fill=fill,stroke=color,r=14,lw=2.2)
        self.text(x+24,y+40,title,28,True,color)
    def tag(self,x,y,label,color,width=170):
        self.rect(x-width/2,y-22,width,38,fill='white',stroke=color,r=19)
        self.text(x,y+4,label,21,True,color,anchor='middle')
    def finish(self):
        self.items.append('</svg>');(OUT/f'{self.name}.svg').write_text('\n'.join(self.items))
        PDF.restoreState();PDF.showPage()
        return self.text_bounds

f=Figure('network_overview')
f.text(46,62,'StereoPoseNet: stereo-temporal articulated instrument pose',43,True)
f.text(48,101,'Current implementation: stereo_pose_unified_c_v1  |  17.945 M parameters  |  ConvNeXtV2 Nano backbone',23,color=MUTED)
f.text(48,137,'Notation: N = 2BT; feature maps = [N, C, H, W].  B: batch, T: time, 2: left/right.  Train T=8; streaming T=1.',22,color=MUTED)

f.panel(45,172,2470,560,'(a) Shared visual localization and per-camera calibration',PURPLE,PURPLE_BG)
f.box(75,245,280,118,'Stereo RGB',('[N,3,720,1280]','uint8; retain raw image R'),PURPLE,ts=26,fs=20)
f.line([(215,363),(215,401)],PURPLE)
f.box(75,402,280,106,'Resize + normalize',('[N,3,360,640]',),PURPLE,ts=23)
f.line([(355,451),(385,451),(385,379),(407,379)],PURPLE)
f.text(835,267,'ConvNeXtV2 Nano: one shared backbone for both views',25,True,PURPLE,anchor='middle')
stage_x=[410,625,840,1055]
for i,(x,c,h,w,blocks) in enumerate(zip(stage_x,[80,160,320,640],[90,45,22,11],[160,80,40,20],[2,2,8,2])):
    f.box(x,310,192,144,f'Stage {i+1}',(f'{blocks} blocks',f'[N,{c},{h},{w}]'),PURPLE,ts=24,fs=20)
    if i<3:f.line([(x+192,379),(x+215,379)],PURPLE)
    f.line([(x+96,454),(x+96,505)],PURPLE,arrow=False)
f.line([(506,505),(1280,505),(1280,402),(1310,402)],PURPLE)
f.text(864,537,'Four-scale feature outputs',20,color=PURPLE,anchor='middle')
f.box(1310,278,445,264,'Top-down FPN',('1x1 lateral projections: C -> 128','Bilinear resize + sum: 4 scales','3x3 Conv 128 -> 128','GroupNorm(8) + GELU','F: [N,128,90,160]'),PURPLE,ts=28,fs=22)
f.line([(1755,402),(1792,402),(1792,298),(1835,298)],PURPLE)
f.line([(1792,402),(1835,402)],PURPLE)
f.box(1835,248,640,93,'Foreground head: 1x1 Conv 128 -> 1',('Mask logits M: [N,1,90,160]',),PURPLE,ts=24)
f.box(1835,365,640,93,'Landmark head: 1x1 Conv 128 -> 5',('Heatmaps: [N,5,90,160] for C, H1, H2, J1, J2',),PURPLE,ts=24,fs=21)
f.line([(2155,458),(2155,491)],PURPLE)
f.box(1835,492,640,124,'Heatmap coordinate decoding',('Sharpened sigmoid expectation, power = 4','U: [N,5,2] pixels; confidence: [N,5]'),PURPLE,ts=25)
f.box(75,558,510,111,'Per-camera K and distortion D',('K: [B,T,2,3,3]; D: [B,T,2,5]','Camera vector k: [BT,2,9]'),PURPLE,ts=25,fs=21)
f.lines(635,588,['k = [fx/W, fy/H, cx/W, cy/H, k1, k2, p1, p2, k3]',
                    'Known per-eye imaging parameters; no stereo baseline input.'],22,32,color=INK)
f.text(635,655,'Ports below reuse these tensors; the original RGB bypasses the backbone for local crops.',21,color=MUTED)
f.tag(440,715,'F + M + U',BLUE,200);f.tag(1280,715,'R + M + U',GREEN,200);f.tag(2120,715,'R + M + U',ORANGE,200)
for x,col in [(440,BLUE),(1280,GREEN),(2120,ORANGE)]:f.line([(x,734),(x,790)],col)

f.panel(45,790,790,1015,'(b) Shaft / body branch',BLUE,BLUE_BG)
f.box(80,865,720,128,'Foreground + five landmark tokens',('1 mask-pooled descriptor + C/H1/H2/J1/J2 patches','8x8 sampling -> local encoder -> 4x4 pool','Six ordered tokens: [N,6,128]'),BLUE,ts=26,fs=21)
f.line([(440,993),(440,1018)],BLUE)
f.box(80,1018,720,88,'Add camera and view embeddings',('Camera MLP 9 -> 128 -> 128; reshape [BT,2,6,128]',),BLUE,ts=25,fs=21)
f.line([(440,1106),(440,1130)],BLUE)
f.box(80,1130,720,99,'Bidirectional cross-attention x 2',('4 heads, 32 dimensions/head; FFN 128 -> 256 -> 128','Input / output: [BT,2,6,128]'),BLUE,ts=26,fs=21)
f.line([(440,1229),(440,1254)],BLUE)
f.box(80,1254,720,78,'Flatten six tokens; FC 768 -> 128',('Reshape: [2B,T,128]',),BLUE,ts=25,fs=21)
f.line([(440,1332),(440,1357)],BLUE)
f.box(80,1357,720,95,'Causal shaft GRU: 128 -> 256',('Output [2B,T,256]; h_shaft [1,2B,256]','Shared weights across eyes; separate L/R histories'),BLUE,ts=26,fs=20)
f.line([(440,1452),(440,1472),(260,1472),(260,1491)],BLUE)
f.line([(440,1472),(620,1472),(620,1491)],BLUE)
f.box(80,1491,345,112,'Rotation heads',('Per eye: 256 -> 256 -> 6','6D -> R: [B,T,2,3,3]'),BLUE,ts=25,fs=20)
f.box(455,1491,345,112,'Depth heads',('[h(256), C token(128)]','384 -> 256 -> 1; softplus'),BLUE,ts=25,fs=19)
f.line([(800,1182),(817,1182),(817,1478),(628,1478),(628,1491)],BLUE,dash=True)
f.text(775,1467,'C token',17,color=BLUE,anchor='end')
f.line([(628,1603),(628,1627)],BLUE)
f.box(455,1627,345,119,'C = Z [xn, yn, 1]',('C pixel from U + K,D','Brown inverse -> ray','C: [B,T,2,3]'),BLUE,ts=24,fs=20)
f.line([(253,1603),(253,1759),(440,1759)],BLUE,arrow=False)
f.line([(628,1746),(628,1759),(440,1759)],BLUE,arrow=False)
f.tag(440,1783,'C_L, R_L | C_R, R_R',BLUE,365)

for x,col,bg,letter,name,roi,output,k in [
    (885,GREEN,GREEN_BG,'c','Wrist','C + hinge + both tips','pitch, yaw',2),
    (1725,ORANGE,ORANGE_BG,'d','Paired-jaw','hinge + both tips','jaw half-angle',1)]:
    center=x+395
    f.panel(x,790,790,1015,f'({letter}) {name} branch',col,bg)
    f.box(x+35,865,720,128,'Predicted foreground crop from raw RGB',('Detach mask and ROI coordinates; mask RGB before sampling',f'ROI uses {roi}', 'RGB + mask: [N,4,160,160]'),col,ts=25,fs=20)
    f.line([(center,993),(center,1018)],col)
    f.box(x+35,1018,720,139,'Independent lightweight CNN',('4 -> 32 -> 48 -> 64 channels; strides 2,2,1','[N,32,80,80] -> [N,48,40,40] -> [N,64,40,40]','Depthwise + pointwise residual refinement'),col,ts=26,fs=20)
    f.line([(center,1157),(center,1182)],col)
    f.box(x+35,1182,720,119,'Spatial descriptor + local metadata',('Pool 4x4 -> 1024 -> FC 128','Add metadata MLP 15 -> 128 -> 128','Paired descriptors: [BT,2,128]'),col,ts=25,fs=21)
    f.line([(center,1301),(center,1326)],col)
    f.box(x+35,1326,720,119,'Local stereo fusion: concatenate + MLP',('[own(128), other(128)] -> 256 -> 128 -> 128','Add own-view descriptor; [BT,2,128]','Reshape to temporal sequences: [2B,T,128]'),col,ts=25,fs=21)
    f.line([(center,1445),(center,1470)],col)
    f.box(x+35,1470,720,95,f'Causal {"wrist" if k==2 else "jaw"} GRU: 128 -> 256',('Output [2B,T,256]; hidden state [1,2B,256]','Own parameters; shared across L/R, separate L/R histories'),col,ts=25,fs=20)
    f.line([(center,1565),(center,1590)],col)
    f.box(x+35,1590,720,139,f'Concatenate L/R states -> angle head',('256 + 256 = 512 per timestamp',f'FC 512 -> 256 -> {k}; sigmoid into physical joint limits',f'{output}: [B,T,{k}]'),col,ts=25,fs=21)
    f.line([(center,1729),(center,1760)],col)
    f.tag(center,1783,output,col,290)

for x,col in [(440,BLUE),(1280,GREEN),(2120,ORANGE)]:f.line([(x,1803),(x,1848)],col,arrow=False)
f.line([(440,1848),(2120,1848)],LINE,arrow=False)
f.line([(1280,1848),(1280,1880),(715,1880),(715,1910)],LINE)
f.text(46,1882,'(e) Pose output and deterministic articulated geometry',29,True,INK)
f.box(65,1910,1300,158,'Exported articulated pose: [B,T,17]',('C_L(3) + quaternion_L(4) + C_R(3) + quaternion_R(4)','+ wrist pitch(1) + wrist yaw(1) + jaw half-angle(1)','Quaternion order: xyzw; C: mm; joint angles: radians.'),BLUE,ts=29,fs=23)
f.line([(1365,1989),(1410,1989)],LINE)
f.box(1410,1910,1080,158,'Known CAD / USD kinematic chain',('Derive d, H1/H2, J1/J2, T and three local XYZ frames','Project each eye with its own K,D -> arrows and panels','Symmetric jaw joints: q1 = -half, q2 = +half'),MUTED,ts=28,fs=23)
f.text(50,2120,'Three independent temporal modules; state pack [3,2B,256] = [shaft, wrist, jaw]. No future frames are used.',23,True,INK)
f.text(50,2155,'R = original RGB; F = FPN features; M = mask logits; U = predicted landmark pixels. Dashed route = reused C token.',21,color=MUTED)
f.text(50,2188,'Tensor dimensions checked at B=1, T=1; the architecture supports causal streaming with three independent states.',20,color=MUTED)
f.finish()

g=Figure('fusion_and_operator_details')
g.text(46,62,'How stereo evidence and temporal history are fused',43,True)
g.text(48,102,'Operator-level details of the same model. Body attention and local-part MLP fusion are distinct implementations.',23,color=MUTED)
g.text(48,137,'N = 2BT. Every branch shares weights across views but keeps the left and right recurrent states separate.',22,color=MUTED)
g.panel(45,172,1210,920,'(a) Shaft: bidirectional token attention',BLUE,BLUE_BG)
g.box(80,246,510,92,'Left tokens L',('[BT,6,128]',),BLUE,ts=27)
g.box(710,246,510,92,'Right tokens R',('[BT,6,128]',),BLUE,ts=27)
for x in [335,965]:g.line([(x,338),(x,367)],BLUE)
g.box(80,367,510,77,'LayerNorm(128)',('Same normalization parameters for L/R',),BLUE,ts=25,fs=19)
g.box(710,367,510,77,'LayerNorm(128)',('Same normalization parameters for L/R',),BLUE,ts=25,fs=19)
g.line([(230,444),(230,510)],BLUE)
g.line([(1070,444),(1070,510)],BLUE)
g.line([(445,444),(445,467),(845,467),(845,510)],BLUE,dash=True)
g.line([(850,444),(850,487),(455,487),(455,510)],BLUE,dash=True)
g.text(140,487,'Q = L',18,color=BLUE);g.text(1100,487,'Q = R',18,color=BLUE)
g.box(80,510,510,130,'Left <- right attention',('Q: L; K,V: R','4 heads x 32; sequence length = 6','Residual: L + attention output'),BLUE,ts=25,fs=22)
g.box(710,510,510,130,'Right <- left attention',('Q: R; K,V: L','4 heads x 32; sequence length = 6','Residual: R + attention output'),BLUE,ts=25,fs=22)
for x in [335,965]:g.line([(x,640),(x,667)],BLUE)
g.box(80,667,510,112,'Residual feed-forward network',('LN128 -> FC128:256 -> GELU','-> FC256:128; add residual'),BLUE,ts=23,fs=21)
g.box(710,667,510,112,'Residual feed-forward network',('LN128 -> FC128:256 -> GELU','-> FC256:128; add residual'),BLUE,ts=23,fs=21)
g.text(650,812,'Two independently parameterized blocks; output [BT,2,6,128]',22,True,BLUE,anchor='middle')
g.line([(335,779),(335,834),(650,834),(650,860)],BLUE)
g.line([(965,779),(965,834),(650,834)],BLUE,arrow=False)
g.box(80,860,1140,100,'Six-token pooling -> causal shaft GRU',('Flatten 768 -> FC128 -> LN128 -> GELU -> [2B,T,128]','GRU128:256 -> per-eye rotation heads; state + C token -> depth heads'),BLUE,ts=26,fs=22)
g.lines(85,1000,['Both attention directions read the original pair inside each block.',
                    'Token order: foreground, C, H1, H2, J1, J2. C token is also routed to Z.'],22,31,color=INK)

g.panel(1305,172,1210,920,'(b) Wrist and jaw: two independent local branches',GREEN,GREEN_BG)
g.box(1340,246,510,119,'Local left descriptor fL',('CNN descriptor(128) + metadata(128)','[BT,128]'),GREEN,ts=25)
g.box(1970,246,510,119,'Local right descriptor fR',('CNN descriptor(128) + metadata(128)','[BT,128]'),GREEN,ts=25)
g.line([(1495,365),(1495,422)],GREEN);g.line([(2325,365),(2325,422)],GREEN)
g.line([(1740,365),(1740,389),(2110,389),(2110,422)],GREEN,dash=True)
g.line([(2110,365),(2110,405),(1740,405),(1740,422)],GREEN,dash=True)
g.box(1340,422,510,128,'[fL || fR] : 256',('LN256 -> FC256:128 -> GELU','-> FC128:128; add fL','Fused left descriptor: [BT,128]'),GREEN,ts=26,fs=21)
g.box(1970,422,510,128,'[fR || fL] : 256',('LN256 -> FC256:128 -> GELU','-> FC128:128; add fR','Fused right descriptor: [BT,128]'),GREEN,ts=26,fs=21)
for x in [1595,2225]:g.line([(x,550),(x,589)],GREEN)
g.box(1340,589,510,118,'Left causal GRU state',('Input [B,T,128] -> [B,T,256]','hL(t) = GRU(xL(t), hL(t-1))'),GREEN,ts=25,fs=21)
g.box(1970,589,510,118,'Right causal GRU state',('Input [B,T,128] -> [B,T,256]','hR(t) = GRU(xR(t), hR(t-1))'),GREEN,ts=25,fs=21)
g.text(1910,747,'Same GRU weights within a branch; no sharing between wrist and jaw',21,True,GREEN,anchor='middle')
g.line([(1595,707),(1595,769),(1910,769),(1910,794)],GREEN)
g.line([(2225,707),(2225,769),(1910,769)],GREEN,arrow=False)
g.box(1340,794,1140,115,'Late stereo-state concatenation',('[hL(t) || hR(t)] -> [B,T,512]','FC512:256 -> GELU -> FC256:k -> sigmoid joint bounds'),GREEN,ts=27,fs=23)
g.line([(1910,909),(1910,930),(1605,930),(1605,950)],GREEN)
g.line([(1910,930),(2215,930),(2215,950)],ORANGE)
g.box(1340,950,530,97,'Wrist: k = 2',('pitch / yaw: [B,T,2]',),GREEN,ts=26,fs=22)
g.box(1950,950,530,97,'Paired jaw: k = 1',('half-opening: [B,T,1]',),ORANGE,fill=ORANGE_BG,ts=26,fs=22)

g.panel(45,1140,1210,804,'(c) Building the six body tokens',BLUE,BLUE_BG)
g.box(80,1214,1140,88,'F: [N,128,90,160]  |  M: [N,1,90,160]  |  U: [N,5,2]',('U from sigmoid(heat)^4 weighted coordinate expectation; confidence = max sigmoid(heat)',),BLUE,ts=23,fs=20)
g.line([(650,1302),(650,1322),(300,1322),(300,1347)],BLUE)
g.line([(650,1322),(880,1322),(880,1347)],BLUE)
g.box(80,1347,440,143,'Whole-instrument token',('w = sigmoid(M)^2 / sum','Weighted spatial pooling of F','[N,1,128]'),BLUE,ts=24,fs=22)
g.box(560,1347,660,143,'Five landmark-centered feature patches',('Radius 24 network pixels; sample 8x8','grid_sample -> [5N,128,8,8]','align_corners=False; border padding'),BLUE,ts=23,fs=22)
g.line([(890,1490),(890,1518)],BLUE)
g.box(560,1518,660,151,'Lightweight local residual encoder',('DWConv3x3(128) -> GN8 -> GELU -> PWConv1x1','Add sampled patch; pool to [5N,128,4,4]','Flatten 2048 -> FC128 -> LN128 -> GELU','Local tokens: [N,5,128]'),BLUE,ts=24,fs=20)
g.line([(300,1490),(300,1700),(650,1700),(650,1730)],BLUE)
g.line([(890,1669),(890,1700),(650,1700)],BLUE,arrow=False)
g.box(80,1730,1140,133,'Concatenate + positional / camera embeddings',('Whole(1) + local(5) -> [N,6,128]; add region embedding [6,128]','Per-token metadata: 5 -> 128 -> 128; add camera: 9 -> 128 -> 128','Add learned view embedding [2,128]; reshape [BT,2,6,128]'),BLUE,ts=26,fs=22)
g.text(85,1902,'Whole metadata: center(2), spread(2), mean mask(1). Local metadata: uv(2), radius(2), confidence(1).',18,color=MUTED)

g.panel(1305,1140,1210,804,'(d) Native-resolution crop and local CNN',ORANGE,ORANGE_BG)
g.box(1340,1214,1140,112,'Predicted foreground routing: stop gradients through mask and ROI',('sigmoid(M) -> resize 720x1280 -> clamp((p-0.2)/0.8,0,1) -> max-pool3x3','Normalize source RGB; multiply by gated mask BEFORE bilinear ROI sampling'),ORANGE,ts=24,fs=21)
g.line([(1910,1326),(1910,1351)],ORANGE)
g.box(1340,1351,1140,109,'Wrist ROI: C,H,J1,J2  |  Jaw ROI: H,J1,J2  |  H=(H1+H2)/2',('Concatenate foreground RGB + mask; sample [N,4,160,160] for each branch','align_corners=False; zero padding; crop coordinates derived from predicted U'),ORANGE,ts=23,fs=21)
g.line([(1910,1460),(1910,1480),(1520,1480),(1520,1505)],ORANGE)
for x,title,lines in [(1340,'Conv3x3 / stride 2',('4 -> 32; GN8 + GELU','[N,32,80,80]')),
                     (1730,'Conv3x3 / stride 2',('32 -> 48; GN8 + GELU','[N,48,40,40]')),
                     (2120,'Conv3x3 / stride 1',('48 -> 64; GN8 + GELU','[N,64,40,40]'))]:
    g.box(x,1505,360,118,title,lines,ORANGE,ts=22,fs=21)
    if x<2120:g.line([(x+360,1564),(x+390,1564)],ORANGE)
g.line([(2300,1623),(2300,1648),(1910,1648),(1910,1673)],ORANGE)
g.box(1340,1673,1140,109,'Residual refinement: DWConv3x3(64) -> GN8 -> GELU -> PWConv1x1',('Feature remains [N,64,40,40]; adaptive pool4x4 -> [N,1024]','FC1024:128 -> LN128 -> GELU -> local descriptor [N,128]'),ORANGE,ts=22,fs=22)
g.line([(1630,1782),(1630,1810)],ORANGE)
g.line([(2225,1782),(2225,1810)],MUTED,dash=True)
g.box(1340,1810,565,96,'Add local metadata MLP 15:128:128',('K/D(9) + box(4) + visibility(1) + conf(1)',),ORANGE,ts=22,fs=19)
g.box(1945,1810,535,96,'Auxiliary heatmap: Conv1x1 64 -> 3',('[N,3,40,40]; training/debug output',),MUTED,fill=GRAY_BG,ts=21,fs=20)

g.rect(45,1980,2470,160,fill=GRAY_BG,stroke=BORDER)
g.text(70,2020,'Temporal state contract',28,True)
g.text(70,2056,'h_shaft [1,2B,256]   +   h_wrist [1,2B,256]   +   h_jaw [1,2B,256]   ->   [3,2B,256]',25,True,BLUE)
g.text(70,2092,'Each frame advances all three modules once. Reset on sequence / timestamp / calibration discontinuity; preserve state within a sequence.',22,color=INK)
g.text(70,2124,'Training uses 8-frame windows. Streaming supplies one image pair at a time, with history retained in the three states.',21,color=MUTED)
g.text(50,2182,'Solid arrows: feature flow. Dashed stereo arrows: other-view input. Auxiliary local heatmaps are skipped by the normal optimized runtime.',21,color=MUTED)
g.finish()
PDF.save()

audit=json.loads(args.audit.read_text())
assert audit['parameters']==17945453
assert audit['backbone']['output']==[[2,80,90,160],[2,160,45,80],[2,320,22,40],[2,640,11,20]]
assert audit['hidden']==[3,2,256] and audit['exported']==[1,1,17]
print(json.dumps({'pdf':str(OUT/'detailed_structure.pdf'),'figures':2,'parameter_count':audit['parameters'],'shape_assertions_passed':True}))
