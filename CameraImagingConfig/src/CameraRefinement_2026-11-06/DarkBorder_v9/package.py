from pathlib import Path
import argparse, json, zipfile
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
root = a.run
project = root.parent
pv = root / 'preview'
pv.mkdir(exist_ok=True)
m = json.loads((root / 'manifest.json').read_text())
report = json.loads((root / 'effect_report.json').read_text())
profile = json.loads((root/'config/dark_border_parameters.json').read_text())
selected = profile['profiles'][profile['selected']]
start, end = selected['start_radius'], selected['end_radius']
floor = selected['outer_linear_gain']
cx, cy = profile['center_pixel']
ax, ay = profile['semiaxes_pixel']
fonts = ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
font = next((ImageFont.truetype(f, 22) for f in fonts if Path(f).exists()), ImageFont.load_default())

def row(paths, labels, name):
    canvas = Image.new('RGB', (1600, 490), '#152033')
    d = ImageDraw.Draw(canvas)
    for i, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as im:
            canvas.paste(im.resize((800, 450), Image.Resampling.LANCZOS), (800*i, 40))
        d.text((800*i+12, 7), label, font=font, fill='white')
    canvas.save(pv / name, quality=96)

cards = []
for sid in m['samples']:
    row([root/f'images/{sid}_{eye}.png' for eye in ['left', 'right']],
        [f'{sid} / LEFT / border 3.5% / global 0.75', f'{sid} / RIGHT / border 3.5% / global 0.75'], f'{sid}_pair.jpg')
    cards.append(f'<section><h2>双目 {sid}</h2><img src="preview/{sid}_pair.jpg"><p><a href="images/{sid}_left.png">左目原图</a> · <a href="images/{sid}_right.png">右目原图</a> · <a href="ground_truth/{sid}.json">几何真值</a></p></section>')
row([project/'run13_border_2p5percent/images/0000_left.png', root/'images/0000_left.png'],
    ['PREVIOUS / border 2.5% / global 1.0', 'NEW / border 3.5% / global 0.75'], 'before_after.jpg')

applied=json.loads((root/'config/applied_parameters.json').read_text())
scales=applied['illumination']['combined_rgb_scales']
notes = f'''# 暗边 3.5% + 整体线性亮度降低 25%

日期：2026-11-07。四组双目、八张 1920×1080 PNG。复用 run05 路径追踪 HDR；未重新渲染。

按用户指定的两项设置执行：**暗边最低线性增益 3.5%，再将全图在线性 RGB 中乘以 0.75**。椭圆轮廓和渐变范围保持 run13 原值。25% 的下降作用在线性光度域，不是在已有 8-bit PNG 上乘 0.75；sRGB 编码后像素值不会统一下降 25%。

| 参数 | 上一版 run13 | 本轮实际值 |
|---|---:|---:|
| 暗边最低线性增益（相对中心） | 2.5% | **3.5%** |
| 全局线性亮度系数 | 1.0 | **0.75** |
| 全局等效 EV 调整 | 0 | **−0.4150374993 EV** |
| 中央相对 run05 的有效线性增益 | 1.0 | **0.75** |
| 最外围相对 run05 的有效线性增益 | 2.5% | **2.625% = 3.5%×0.75** |
| 遮罩中心 / px | (959.5,539.5) | (959.5,539.5) |
| 椭圆半轴 / px | (840,470) | (840,470) |
| 渐变半径 | 0.30→1.10 | 0.30→1.10 |
| 渐变函数 | Smoothstep | Smoothstep |
| 最低增益椭圆半轴 / px | (924,517) | (924,517) |

```text
r = sqrt(((u-959.5)/840.0)^2 + ((v-539.5)/470.0)^2)
t = clip((r-0.30)/0.80, 0, 1)
mask_gain = 1 - 0.965*t*t*(3-2*t)
RGB_linear_final = RGB_linear_source_response * mask_gain * 0.75
```

全局调暗应用于新设为 3.5% 的遮罩结果。与上一版两项设置共同比较时，中央变为原来的 75%，最外围则为原来的 105%（2.625%/2.5%）；这不是两项变化合计后每个像素都下降 25%。椭圆边界仍是 x=35.5、1883.5 px 和 y=22.5、1056.5 px；横向渐变宽度 672 px、纵向 376 px 均保留。

## 最终有效响应数值

源 run05 线性响应补偿为 16 倍（+4 EV），加入全局 0.75 后有效倍率为 **12 倍**，等效 **+3.5849625007 EV**。这只是输出光度响应，不是硬件曝光时间或灯光功率调整。

| 眼 | 最终有效 RGB 系数（已含全局 0.75） |
|---|---|
| 左 | {scales['left']} |
| 右 | {scales['right']} |

config/applied_parameters.json 的 illumination.combined_rgb_scales 与 post_hdr_linear_multiplier 已折入 0.75，使用这些最终有效值时不要再乘一次全局系数。global_brightness_adjustment 保存源系数、0.75 和等效 EV，用于说明计算过程。生成脚本从 run05 源系数计算，再应用一次全局系数。

## 来源和成像顺序

3.5% 与降低 25% 来自本次用户明确指定；按已有线性光度流程实现。遮罩轮廓见 [run13](../run13_border_2p5percent/RUN_NOTES.md)，渲染和相机参数来源见 [run05](../run05_initial/RUN_NOTES.md)。这些效果数值不是厂家或 grasp5 实测暗角、曝光数据。

原路径追踪 HDR（原生景深、畸变映射已在缓存）→ 原空间虚化 → 源固定 RGB 响应（含 ×16）→ 3.5% 椭圆遮罩 → 全局 ×0.75 → sRGB → 原噪声模型与种子 → PNG。

噪声模型和随机种子保持原设置，信号相关噪声幅度随新亮度计算。相机 K/D、R/t、场景及器械位姿、几何真值保留。有效焦距 4 mm、f/5.6、100 mm 对焦、路径追踪每帧 64 样本与总采样设置 256 保持原值。原标签 inside_image 仅表示几何投影在画幅内。

## 检查与复现

八张图中央有效线性增益为 0.75，与独立生成的仅全局调暗参考在中央逐像素一致；外围有效线性增益为 0.02625。原 HDR、源图像及真值散列保持一致。外侧 12 px 的 RGB 通道合并中位数为 {min(f['border_12px_median_dn'] for f in report['frames']):g}～{max(f['border_12px_median_dn'] for f in report['frames']):g}/255。详细数据见 effect_report.json。

原图 images/；参数 config/；几何真值 ground_truth/。从工程根目录运行，指定不存在的新目录：

```bash
python DarkBorder_v9/generate.py --source run05_initial --out run15_border_3p5_global075
python DarkBorder_v9/package.py --run run15_border_3p5_global075
```

Ubuntu 本轮目录：`/home/leorobot/PoseProject/CameraImagingConfig/CameraRefinement_2026-11-06/{root.name}/`。
'''
if profile.get('configuration_acceptance',{}).get('confirmed'):
    notes=notes.replace('\n\n日期：','\n\n**已确认采用：2026-11-07，用户确认本组数值与画面效果。后续生成默认沿用本配置；如需修改，另建版本并记录差异。**\n\n日期：',1)
(root/'RUN_NOTES.md').write_text(notes)
html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>暗边 3.5% + 整体调暗 25% · 四组双目</title><style>body{max-width:1550px;margin:32px auto;padding:0 22px;background:#101725;color:#e7edf5;font:17px/1.7 system-ui}a{color:#8cd3ff}img{width:100%;height:auto}section{padding:22px 0;border-top:1px solid #354153}</style><h1>暗边 3.5% + 整体调暗 25%</h1><p>暗边最低线性增益 3.5%，再将整幅线性 RGB 乘以 0.75；相对于源响应，中央有效增益 75%，外围 2.625%。椭圆中心、半轴和渐变范围均保持上一版。这是用户指定的效果设置。</p><p><a href="RUN_NOTES.md">实际参数、生效顺序与来源</a> · <a href="config/applied_parameters.json">完整配置 JSON</a> · <a href="stereo_images_and_labels.zip">下载原图与标签</a></p><h2>上一版与本轮</h2><img src="preview/before_after.jpg">'''+''.join(cards)+'</html>'
if profile.get('configuration_acceptance',{}).get('confirmed'):
    html=html.replace('</h1>','</h1><p><strong>已确认采用的配置 · 2026-11-07</strong>：后续生成默认沿用这些数值。</p>',1)
(root/'index.html').write_text(html)
with zipfile.ZipFile(root/'stereo_images_and_labels.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for folder in ['images','config','ground_truth']:
        for f in sorted((root/folder).iterdir()):
            if f.is_file(): z.write(f,str(f.relative_to(root)))
    for f in ['RUN_NOTES.md','effect_report.json','manifest.json']: z.write(root/f,f)
print(root/'index.html')
