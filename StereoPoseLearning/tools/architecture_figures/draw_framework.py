"""Publication framework: embedded typography, aligned branches, readable shapes.
The two existing operator-detail figures are deliberately left untouched.
"""
from pathlib import Path
from io import BytesIO
import argparse, base64, html, math, json, os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import toColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PROJECT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=PROJECT/'docs/figures/model')
parser.add_argument('--assets', type=Path, default=PROJECT/'docs/figures/model/assets')
parser.add_argument('--font-dir', type=Path, help='Directory with LiberationSans-Regular.ttf and LiberationSans-Bold.ttf')
args = parser.parse_args()
OUT=args.output.resolve(); OUT.mkdir(parents=True,exist_ok=True); ASSETS=args.assets.resolve()
W,H=3000,1500; S=.36
candidates = [args.font_dir, Path(os.environ['LIBERATION_FONT_DIR']) if os.environ.get('LIBERATION_FONT_DIR') else None,
              Path('/usr/share/fonts/truetype/liberation2'), Path('/usr/share/fonts/truetype/liberation'),
              Path('/Library/Fonts'), Path.home()/'.fonts']
FONTROOT = next((p for p in candidates if p and all((p/n).is_file() for n in ('LiberationSans-Regular.ttf','LiberationSans-Bold.ttf'))), None)
if FONTROOT is None: parser.error('Install fonts-liberation2 or pass --font-dir /path/to/LiberationSans')
FONTFILES={False:FONTROOT/'LiberationSans-Regular.ttf',True:FONTROOT/'LiberationSans-Bold.ttf'}
for bold,p in FONTFILES.items(): pdfmetrics.registerFont(TTFont('FigureBold' if bold else 'FigureRegular',str(p)))
C={'ink':'#172b40','muted':'#405266','line':'#607387','purple':'#654690','purple_bg':'#f4f0fa',
   'blue':'#2266a4','blue_bg':'#edf4fc','green':'#227451','green_bg':'#eef7f2','orange':'#ad5c16','orange_bg':'#fff6eb',
   'light':'#f5f7fa','border':'#cbd5df'}
pdf=canvas.Canvas(str(OUT/'framework.pdf'),pagesize=(W*S,H*S));pdf.setTitle('StereoPoseNet: stereo-temporal articulated pose framework');pdf.scale(S,S)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<defs><style>']
for bold,p in FONTFILES.items():
 svg.append('@font-face {font-family:FigureSans;font-weight:'+('700' if bold else '400')+';src:url(data:font/ttf;base64,'+base64.b64encode(p.read_bytes()).decode()+') format("truetype");}')
svg+=['</style></defs>','<rect width="100%" height="100%" fill="white"/>']
TEXT_AUDIT=[]

def text(x,y,s,size=28,bold=False,color=None,anchor='start',max_width=None):
 color=color or C['ink']; font='FigureBold' if bold else 'FigureRegular'
 width=pdfmetrics.stringWidth(s,font,size);xx=x if anchor=='start' else x-width/2 if anchor=='middle' else x-width
 assert xx>=0 and xx+width<=W,(s,xx,width)
 if max_width is not None: assert width<=max_width,(s,width,max_width)
 svg.append(f'<text x="{x}" y="{y}" font-family="FigureSans,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{html.escape(s)}</text>')
 pdf.setFillColor(toColor(color));pdf.setFont(font,size);pdf.drawString(xx,H-y,s)
 TEXT_AUDIT.append({'text':s,'font_size':size,'bold':bold,'x':xx,'y':y,'width':width})

def rect(x,y,w,h,fill='white',stroke=None,r=10,lw=2):
 stroke=stroke or C['border'];svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
 pdf.setFillColor(toColor(fill));pdf.setStrokeColor(toColor(stroke));pdf.setLineWidth(lw);pdf.roundRect(x,H-y-h,w,h,r,fill=1,stroke=1)

def poly(points,fill,stroke=None,lw=1.8):
 svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="{fill}" stroke="{stroke or fill}" stroke-width="{lw}"/>')
 pdf.setFillColor(toColor(fill));pdf.setStrokeColor(toColor(stroke or fill));pdf.setLineWidth(lw);p=pdf.beginPath();p.moveTo(points[0][0],H-points[0][1])
 for x,y in points[1:]:p.lineTo(x,H-y)
 p.close();pdf.drawPath(p,fill=1,stroke=1)

def line(points,color=None,lw=3,arrow=True,dash=False,halo=False):
 color=color or C['line']
 if halo:line(points,'white',lw+8,False)
 svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{lw}" stroke-linejoin="round"'+(' stroke-dasharray="9 7"' if dash else '')+'/>')
 pdf.setStrokeColor(toColor(color));pdf.setLineWidth(lw);pdf.setDash([9,7] if dash else []);p=pdf.beginPath();p.moveTo(points[0][0],H-points[0][1])
 for x,y in points[1:]:p.lineTo(x,H-y)
 pdf.drawPath(p);pdf.setDash([])
 if arrow:
  x,y=points[-1];a,b=points[-2];t=math.atan2(y-b,x-a);l=14;d=5.8
  poly([(x,y),(x-l*math.cos(t)+d*math.sin(t),y-l*math.sin(t)-d*math.cos(t)),(x-l*math.cos(t)-d*math.sin(t),y-l*math.sin(t)+d*math.cos(t))],color,lw=0)

def image(name,x,y,w,h,color=None,crop=None):
 im=Image.open(ASSETS/name).convert('RGB')
 if crop:im=im.crop(crop)
 im=im.resize((round(w*2),round(h*2)),Image.Resampling.LANCZOS);b=BytesIO();im.save(b,format='PNG');data=b.getvalue()
 svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{base64.b64encode(data).decode()}"/>')
 pdf.drawImage(ImageReader(BytesIO(data)),x,H-y-h,w,h)
 stroke=color or C['border'];svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{stroke}" stroke-width="2"/>');pdf.setStrokeColor(toColor(stroke));pdf.setLineWidth(2);pdf.rect(x,H-y-h,w,h,fill=0,stroke=1)

def module(x,y,w,h,titles,shapes,col):
 rect(x,y,w,h,'white',col,r=12,lw=2.2)
 for i,s in enumerate(titles):text(x+w/2,y+34+i*32,s,29,True,col,'middle',w-22)
 # Shapes stay inside the box, below the module name, never on the connector.
 sy=y+h-22-(len(shapes)-1)*31
 for i,s in enumerate(shapes):text(x+w/2,sy+i*31,s,26,False,C['muted'],'middle',w-20)

def loop(x,y,w,col):
 line([(x+w-18,y+28),(x+w+19,y+28),(x+w+19,y-32),(x+w/2,y-32),(x+w/2,y)],col,2.5)
 text(x+w/2,y-49,'previous state',25,False,col,'middle')

def plus(x,y,col):
 rect(x-16,y-16,32,32,'white',col,r=16);text(x,y+9,'+',28,True,col,'middle')

text(50,61,'Foreground-guided stereo-temporal articulated pose estimation',44,True)
text(52,106,'Shared visual features  •  Three task-specific branches  •  Pose in each camera coordinate system',28,color=C['muted'])

# Branch panels: equal width, common module baselines, stronger type hierarchy.
for y,h,title,col,bg in [(150,370,'SHAFT / BODY',C['blue'],C['blue_bg']),(615,330,'WRIST',C['green'],C['green_bg']),(1015,330,'PAIRED JAW',C['orange'],C['orange_bg'])]:
 rect(1395,y,1060,h,bg,col,18,2.2);text(1425,y+46,title,33,True,col)

# Stereo images, encoder and feature pyramid.
text(190,190,'Stereo RGB pair',31,True,anchor='middle')
image('input_left.png',50,230,280,157.5,C['purple']);image('input_right.png',50,420,280,157.5,C['purple'])
for y,s in [(230,'L'),(420,'R')]:
 rect(57,y+7,37,34,C['ink'],C['ink'],r=4,lw=0);text(75.5,y+33,s,27,True,'white','middle')
text(190,616,'1280 × 720 / eye',27,False,C['muted'],'middle')
rect(370,185,580,393,C['purple_bg'],C['purple'],18,2.2)
text(660,235,'SHARED VISUAL ENCODER',31,True,C['purple'],'middle')
line([(330,309),(349,309),(349,397),(408,397)],C['purple'])
line([(330,499),(349,499),(349,397)],C['purple'],arrow=False)
# The backbone is a solid volume; FPN has multiple resolutions.
poly([(408,302),(425,285),(655,285),(638,302)],'#e3daee',C['purple'])
poly([(638,302),(655,285),(655,475),(638,492)],'#c9b9df',C['purple'])
rect(408,302,230,190,'#e3daee',C['purple'],3)
text(523,350,'ConvNeXtV2',30,True,C['purple'],'middle')
text(523,389,'Nano',30,True,C['purple'],'middle')
text(523,444,'L/R weight sharing',25,False,C['ink'],'middle')
text(523,479,'640 × 360 input',26,False,C['ink'],'middle')
line([(661,397),(702,397)],C['purple'])
for size,x,y,fill in [(140,708,301,'#dcd3eb'),(106,744,318,'#cbbce0'),(76,783,338,'#bba4d4'),(48,821,358,'#a88ac6')]:
 rect(x,y,size,size*.9,fill,C['purple'],2,1.8)
 for j in range(1,4):
  line([(x+size*j/4,y),(x+size*j/4,y+size*.9)],C['purple'],.7,False)
  line([(x,y+size*.9*j/4),(x+size,y+size*.9*j/4)],C['purple'],.7,False)
text(801,471,'128-channel FPN',28,True,C['purple'],'middle')
text(660,548,'Feature map  [N, 128, 90, 160]',29,False,C['ink'],'middle')

# Predicted localization and visual region tokens.
text(1170,190,'Predicted localization',31,True,C['purple'],'middle')
line([(950,397),(980,397),(980,271),(1000,271)],C['purple'])
image('mask_left.png',1000,230,150,84.4,C['purple']);image('heat_left.png',1195,230,150,84.4,C['purple'])
text(1075,350,'1 mask',27,True,C['purple'],'middle');text(1270,350,'5 keypoints',27,True,C['purple'],'middle')
text(1170,386,'Each map: 90 × 160',27,False,C['muted'],'middle')
line([(1170,397),(1170,420)],C['blue'])
for row,side in enumerate(['L','R']):
 yy=425+row*57;text(1007,yy+30,side,27,True,C['blue'],'end')
 for i,label in enumerate(['F','C','H1','H2','J1','J2']):
  xx=1023+i*54;rect(xx,yy,46,40,'#dceafa',C['blue'],4,1.4);text(xx+23,yy+29,label,24,True,C['blue'],'middle')
text(1170,559,'6 × 128 tokens / eye',28,True,C['blue'],'middle')
text(1170,595,'+ camera / view embeddings',25,False,C['muted'],'middle')
line([(1347,472),(1374,472),(1374,385),(1435,385)],C['blue'],halo=True)
# FPN feeds the foreground-weighted and keypoint-guided token extraction.
line([(950,490),(980,490),(980,531),(1023,531)],C['blue'],2.4,dash=True)

# Body stereo fusion, temporal state and camera-frame pose.
module(1435,310,280,150,['Cross-view','attention'],['2 blocks · 4 heads','[BT, 2, 6, 128]'],C['blue'])
line([(1715,385),(1785,385)],C['blue'])
module(1785,310,245,150,['Body GRU'],['[2B, T, 128]','→ [2B, T, 256]'],C['blue']);loop(1785,310,245,C['blue'])
line([(2030,385),(2130,385)],C['blue'])
module(2130,310,285,150,['Rotation + depth'],['R6D (6) + Z (1)','C from pixel ray'],C['blue'])
line([(1575,310),(1575,236),(2273,236),(2273,310)],C['blue'],2.4,dash=True)
text(1710,225,'Fused C token (128)',25,False,C['blue'],'middle')
text(1435,499,'Pool: 6 × 128 → 128',27,False,C['blue'])
text(1810,499,'Two camera-frame poses: C + quaternion',27,False,C['blue'])

# Foreground ROI routing uses original-resolution images and localization heads.
line([(190,624),(190,665),(1038,665),(1038,723),(1070,723)],C['line'],2.7)
text(650,653,'Original-resolution RGB',28,False,C['muted'],'middle')
# One crossing is deliberately bridged with a white halo, not a graph junction.
line([(1345,272),(1360,272),(1360,682),(1350,682)],C['purple'],2.4,dash=True,halo=True)
rect(1070,645,280,105,C['light'],C['line'],12,2)
text(1210,687,'Foreground ROIs',29,True,anchor='middle')
text(1210,725,'Detached mask / boxes',25,False,C['muted'],'middle')
line([(1210,750),(1210,770)],C['green'])
line([(1090,750),(1000,750),(1000,1230),(1050,1230)],C['orange'],2.4)

for y,col,name,out in [(775,C['green'],'Wrist',2),(1175,C['orange'],'Jaw',1)]:
 image('wrist_left.png' if name=='Wrist' else 'jaw_left.png',1050,y,125,125,col)
 image('wrist_right.png' if name=='Wrist' else 'jaw_right.png',1210,y,125,125,col)
 for x,s in [(1112,'L'),(1272,'R')]:text(x,y+155,s,27,True,col,'middle')
 line([(1335,y+62),(1435,y+62)],col)
 module(1435,y-12,195,145,['Local CNN'],['4 × 160 × 160','→ 128-D'],col)
 line([(1630,y+62),(1690,y+62)],col);plus(1660,y+62,col)
 text(1660,y+25,'m',25,True,col,'middle')
 module(1690,y-12,215,145,['Stereo MLP'],['[L | R], [R | L]','256 → 128'],col)
 line([(1905,y+62),(1965,y+62)],col)
 module(1965,y-12,205,145,[name+' GRU'],['128 → 256','per-eye state'],col);loop(1965,y-12,205,col)
 line([(2170,y+62),(2230,y+62)],col)
 module(2230,y-12,195,145,['Angle head'],['[hL | hR]: 512','256 → '+str(out)],col)
 text(1435,y-67,'Branch-specific weights; shared across L/R.',26,False,col)

# Merge the 14 + 2 + 1 outputs. The neutral bus denotes concatenation.
line([(2415,385),(2480,385)],C['blue'],arrow=False)
line([(2425,837),(2480,837)],C['green'],arrow=False)
line([(2425,1237),(2480,1237)],C['orange'],arrow=False)
line([(2480,385),(2480,1237)],C['line'],3,False)
line([(2480,730),(2525,730)],C['line'])
text(2470,367,'14',27,True,C['blue'],'end');text(2470,815,'2',27,True,C['green'],'end');text(2470,1215,'1',27,True,C['orange'],'end')
rect(2525,560,425,355,'white',C['line'],14,2.6)
text(2737,608,'ARTICULATED POSE',31,True,anchor='middle')
text(2737,648,'[B, T, 17]',29,True,anchor='middle')
for i in range(17):
 col=C['blue'] if i<14 else C['green'] if i<16 else C['orange'];rect(2572+i*20,673,15,32,col,col,2,1)
text(2737,752,'Left: C (3) + quaternion (4)',28,False,C['blue'],'middle')
text(2737,792,'Right: C (3) + quaternion (4)',28,False,C['blue'],'middle')
text(2737,839,'Wrist pitch + yaw (2)',28,False,C['green'],'middle')
text(2737,879,'Jaw half-opening angle (1)',28,False,C['orange'],'middle')
line([(2737,915),(2737,957)],C['line'])
module(2525,957,425,106,['Instrument kinematics'],['d, H1/H2, J1/J2, T, local XYZ'],C['line'])
line([(2737,1063),(2737,1130)],C['line'])
rect(2538,1080,399,32,'white','white',0,0)
text(2737,1105,'Per-eye projection with K and D',27,True,anchor='middle')
image('output_pair.png',2525,1130,425,239,C['line'],crop=(0,0,1280,720))
text(2737,1404,'Three vectors + local XYZ panels',26,False,C['muted'],'middle')

# Supporting facts are grouped in two readable cards, not scattered near edges.
rect(50,759,427,520,C['light'],C['border'],14,2)
text(78,807,'CAMERA CONDITIONING',29,True)
text(78,859,'Known K + Brown distortion D',27)
text(78,905,'Global token embedding',28,True,C['blue'])
text(78,943,'9 → 128 dimensions',27)
text(78,999,'Local metadata m',28,True)
for yy,s in [(1040,'Camera (9) + ROI (4)'),(1079,'+ mask fraction (1)'),(1118,'+ keypoint confidence (1)')]:text(78,yy,s,27)
text(78,1172,'15 → 128 dimensions',27,True)
text(78,1233,'No stereo extrinsic input',27,True,C['muted'])
rect(520,759,430,520,C['blue_bg'],C['blue'],14,2)
text(548,807,'C FROM A SINGLE RAY',29,True,C['blue'])
text(548,863,'Predicted C pixel',29,True,C['blue'])
text(548,903,'+ K, D → undistorted ray',27)

# True typographic subscripts, drawn identically in PDF and SVG.
formula=[('C = Z [x',36,0),('n',24,8),(', y',36,0),('n',24,8),(', 1]',36,0)]
fw=sum(pdfmetrics.stringWidth(s,'FigureBold',size) for s,size,dy in formula)
fx=735-fw/2
for fragment,size,dy in formula:
 text(fx,984+dy,fragment,size,True,C['blue'])
 fx+=pdfmetrics.stringWidth(fragment,'FigureBold',size)
poly([(553,1114),(580,1128),(553,1142)],C['blue'])
line([(580,1128),(867,1077)],C['blue'],3)
rect(855,1065,18,18,C['blue'],C['blue'],9,0)
text(884,1079,'C',29,True,C['blue'])
text(624,1060,'Learn Z',28,True,C['blue'])
text(548,1194,'X, Y follow the predicted ray.',27)
text(548,1233,'No second lateral regression.',27,True,C['blue'])
text(50,1333,'F: foreground token   H1/H2: hinges   J1/J2: jaw tips',27,color=C['muted'])
text(50,1373,'Foreground ROIs use the original 720p images.',27,color=C['muted'])

line([(50,1432),(2950,1432)],C['border'],2,False)
text(50,1475,'N = 2BT   •   B: batch   T: time   •   Training T = 8   •   Streaming T = 1 + history   •   Three GRU states: [3, 2B, 256]',29,color=C['muted'])
text(2945,1475,'stereo_pose_unified_c_v1',25,color=C['muted'],anchor='end')

svg.append('</svg>');(OUT/'framework.svg').write_text('\n'.join(svg));pdf.save()
(OUT/'typography_audit.json').write_text(json.dumps({'font':'Liberation Sans, embedded regular and bold','minimum_size':min(x['font_size'] for x in TEXT_AUDIT),'text':TEXT_AUDIT,'unchanged_model':True},indent=2))
print('Created framework PDF/SVG; fonts embedded; minimum font size:',min(x['font_size'] for x in TEXT_AUDIT))
