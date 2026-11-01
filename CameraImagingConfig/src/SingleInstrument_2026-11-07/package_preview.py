"""Verify and assemble the newly rendered single-instrument stereo preview."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path('/home/leorobot/PoseProject/CameraImagingConfig/SingleInstrument_2026-11-07')
native = root / 'preview/native'
final = root / 'preview_final'
reference = root.parent / 'CameraRefinement_2026-11-06/run14_border_3p5_global075'

def read(path):
    return json.loads(path.read_text())

m = read(native / 'manifest.json')
assert m['completed'] and m['samples'] == ['0000']
assert m['instrument_configuration']['count'] == 1
assert m['instrument_configuration']['active_tools'] == ['right_tool']
assert m['renderer'] == 'PathTracing'
assert m['render_sampling']['spp_per_frame'] == 64
assert m['render_sampling']['total_spp_setting'] == 256
gt = read(final / 'ground_truth/0000.json')
assert list(gt['tools']) == ['right_tool']
assert sorted(gt['cameras']) == ['left', 'right']
assert read(final/'config/camera_intrinsics.json') == read(reference/'config/camera_intrinsics.json')
assert read(final/'ground_truth/stereo_extrinsics.json') == read(reference/'ground_truth/stereo_extrinsics.json')
profile = read(final/'config/dark_border_parameters.json')
assert profile['global_linear_multiplier'] == 0.75
assert profile['profiles']['border_3p5_global075']['outer_linear_gain'] == 0.035
assert read(final/'effect_report.json')['all_checks_passed']

fm = read(final/'manifest.json')
fm.update(source_run='../preview', native_scene='../preview/native/scene/sample_0000.usda',
          new_scene_render_performed=True,
          instrument_configuration=m['instrument_configuration'],
          native_render_sampling=m['render_sampling'])
(final/'manifest.json').write_text(json.dumps(fm, indent=2)+'\n')
pair = Image.new('RGB', (3840, 1136), '#152033')
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
draw = ImageDraw.Draw(pair)
for i, eye in enumerate(['left', 'right']):
    with Image.open(final/f'images/0000_{eye}.png') as image:
        image.load()
        assert image.size == (1920, 1080)
        pair.paste(image.convert('RGB'), (1920*i, 56))
    draw.text((1920*i+22, 12), f'{eye.upper()} / SINGLE INSTRUMENT', font=font, fill='white')
pair.save(final/'stereo_pair.jpg', quality=96, subsampling=0)
report = dict(completed=True, new_isaac_sim_render=True, instrument_count=1,
              retained_instrument='right_tool', resolution_per_eye=[1920,1080],
              renderer=m['renderer'], render_sampling=m['render_sampling'],
              intrinsics_distortion_stereo_unchanged=True,
              outer_linear_gain=0.035, global_linear_multiplier=0.75,
              images=['images/0000_left.png', 'images/0000_right.png'])
(final/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
