"""Publish approved XYZ overlays after a pair is committed; never render RGB."""
from pathlib import Path
import os
import shutil
from PIL import Image, ImageFont
from io_utils import read, write, sha
from pose_truth import validate_body_truth
from tools.reannotate import vector_panel


def preview_font(machine):
    configured = machine.get('preview_font')
    candidates = [Path(configured)] if configured else [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
        Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')]
    for path in candidates:
        if path.is_file():
            ImageFont.truetype(str(path), 25)
            return path
    raise FileNotFoundError('Set machine.preview_font to an existing Chinese-capable font before capture')


class LivePreview:
    def __init__(self, output, font):
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.font = Path(font)
        ImageFont.truetype(str(font), 25)
        self.code_sha256 = sha(Path(__file__).parent/'tools/reannotate.py')
        self.font_sha256 = sha(self.font)
        self._page()

    def _page(self):
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>双目采集预览</title>
<style>body{background:#101820;color:#eef3f7;font:18px/1.6 system-ui;margin:20px}img{width:100%;height:auto}</style>
<h2>双目采集 · 最新完成的一对图像</h2>
<p>生成完成一对后更新。图中显示该帧仿真真值；60 Hz 是仿真采样频率，非渲染或页面刷新速度。</p>
<p id="state">等待已完成的图像…</p><img id="pair" alt="等待首对图像">
<script>let busy=false;function refresh(){if(busy)return;busy=true;const p=new Image();
p.onload=()=>{document.getElementById('pair').src=p.src;document.getElementById('state').textContent='预览已刷新；序列、帧号及坐标见图。';busy=false;};
p.onerror=()=>{busy=false;};p.src='latest.jpg?t='+Date.now();}refresh();setInterval(refresh,2000);</script></html>'''
        target = self.output/'live.html'; temp = target.with_suffix('.tmp')
        temp.write_text(page); os.replace(temp, target)

    def publish(self, record, camera_truth, images, source_label_sha256):
        axes = validate_body_truth(record, camera_truth)
        panels=[]; projections={}
        for eye in ('left', 'right'):
            panel, projections[eye] = vector_panel(images[eye], record, eye, self.font,
                                                   split_jaws=True, coordinate_axes=axes)
            panels.append(panel)
        w,h=panels[0].size
        canvas=Image.new('RGB',(2*w,h))
        for i,panel in enumerate(panels): canvas.paste(panel,(i*w,0))
        target=self.output/f"{record['sequence_id']}_{record['job_id']}.jpg"
        temp=target.with_suffix('.partial.jpg');canvas.save(temp,quality=95,subsampling=0);os.replace(temp,target)
        entry={'sequence_id':record['sequence_id'],'job_id':record['job_id'],'time_s':record['time_s'],
               'source_label_sha256':source_label_sha256,'image':target.name,'image_sha256':sha(target),
               'annotation_code_sha256':self.code_sha256,'font_sha256':self.font_sha256,
               'projection_results':projections,'new_rgb_renders':0}
        write(target.with_suffix('.json'),entry)
        temp=self.output/'latest.partial.jpg';shutil.copyfile(target,temp);os.replace(temp,self.output/'latest.jpg')
        write(self.output/'latest.json',entry)
        return target

    def publish_saved(self, label):
        label=Path(label);record=read(label)
        truth_path=label.parent/record['camera_truth']
        if sha(truth_path)!=record['camera_truth_sha256']:raise ValueError('Camera truth hash mismatch')
        images={}
        for eye,e in record['eyes'].items():
            path=label.parent/e['image']
            if sha(path)!=e['image_sha256']:raise ValueError('Source image hash mismatch')
            with Image.open(path) as image: images[eye]=image.convert('RGB')
        return self.publish(record,read(truth_path),images,sha(label))
