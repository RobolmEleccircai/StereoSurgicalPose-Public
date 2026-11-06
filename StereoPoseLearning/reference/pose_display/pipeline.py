"""Public frame-to-overlay API. No USD, world pose or camera extrinsics needed."""
from dataclasses import dataclass
from pathlib import Path
import copy
import time
import numpy as np
from PIL import Image, ImageDraw
from .decoding import decode
from .geometry import build_geometry
from .projection import project
from .renderer import vector_panel, font_cached


@dataclass
class StereoFrame:
    frame_id: str
    timestamp_ns: int
    left_rgb: np.ndarray
    right_rgb: np.ndarray


@dataclass
class PosePrediction:
    frame_id: str
    timestamp_ns: int
    values17: object
    source: str = 'prediction'
    # Optional measured/predicted visibility. None means unknown, never visible.
    visibility: object = None
    visibility_source: str = 'unknown'


def find_font(path=None):
    candidates = [Path(path)] if path else [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
        Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')]
    for item in candidates:
        if item.is_file():
            font_cached(str(item), 25)
            return str(item)
    raise FileNotFoundError('Pass --font with a Chinese-capable font')


class StereoOverlay:
    def __init__(self, instrument_model, calibration, font=None, compact=True):
        self.model = copy.deepcopy(instrument_model)
        self.cameras = copy.deepcopy(calibration)
        if calibration['image'].get('space') != 'raw_distorted':
            raise ValueError('This profile displays raw distorted images; supply the matching K/D')
        self.width = int(calibration['image']['width']); self.height = int(calibration['image']['height'])
        if self.width < 1280 or self.height < 720:
            raise ValueError('The approved display layout requires at least 1280x720 per eye')
        for eye in ('left', 'right'):
            K = np.asarray(calibration[eye]['K'], float); D = np.asarray(calibration[eye]['D'], float)
            if K.shape != (3, 3) or D.shape != (5,) or not np.isfinite(K).all() or not np.isfinite(D).all():
                raise ValueError('Expected finite K (3x3) and five distortion coefficients')
            if K[0, 0] <= 0 or K[1, 1] <= 0 or max(abs(K[0, 1]), abs(K[1, 0])) > 1e-12 or not np.allclose(K[2], [0, 0, 1]):
                raise ValueError('Unsupported camera matrix')
        self.font = find_font(font); self.compact = compact

    def _images(self, frame):
        images = {}
        for eye in ('left', 'right'):
            image = np.asarray(getattr(frame, eye+'_rgb'))
            if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
                raise ValueError('RGB size/type disagrees with calibration; update K when resizing')
            images[eye] = Image.fromarray(image)
        return images

    def render(self, frame, prediction, quality=None):
        """Return a new RGB stereo canvas and projection metadata; inputs stay unchanged."""
        started = time.perf_counter(); images = self._images(frame)
        if (frame.frame_id, frame.timestamp_ns) != (prediction.frame_id, prediction.timestamp_ns):
            raise ValueError('Prediction belongs to another frame; stale overlays are rejected')
        if prediction.source not in ('prediction', 'ground_truth_replay'):
            raise ValueError('Explicit prediction/replay provenance required')
        poses = decode(prediction.values17, self.model)
        geometry = build_geometry(poses, self.cameras, self.model)
        record = {'display_source': prediction.source, 'sequence_id': str(frame.frame_id)[:48],
                  'job_id': f't={frame.timestamp_ns/1e9:.4f}s', 'jaw_half_angle_deg': float(np.rad2deg(prediction.values17[16])), 'eyes': {}}
        for eye, g in poses.items():
            camera = self.cameras[eye]; points = [g[k] for k in ('C', 'T', 'J1', 'J2')]
            uv = project(points, camera['K'], camera['D'])
            inside = [bool(0 <= p[0] < self.width and 0 <= p[1] < self.height) for p in uv]
            if prediction.visibility is None:
                visibility = [{'visible': None, 'occluder_kind': 'unknown', 'in_frame': inside[i+2]} for i in (0, 1)]
            else:
                if prediction.visibility_source == 'unknown': raise ValueError('Visibility needs explicit provenance')
                visibility = copy.deepcopy(prediction.visibility[eye])
                if len(visibility) != 2: raise ValueError('Expected two fixed-tip visibility records')
                for i, item in enumerate(visibility):
                    if item['visible'] not in (True, False, None): raise ValueError('Invalid visibility state')
                    item['in_frame'] = inside[i+2]
                    if not item['in_frame']: item['visible'] = False
            tool = {'shaft_reference_position_mm': g['C'].tolist(), 'shaft_direction_unit': g['d'].tolist(),
                'tip_position_mm': g['T'].tolist(), 'jaw1_tip_position_mm': g['J1'].tolist(), 'jaw2_tip_position_mm': g['J2'].tolist(),
                'shaft_reference_pixel': uv[0].tolist(), 'tip_pixel': uv[1].tolist(), 'jaw_tip_pixels': uv[2:].tolist()}
            record['eyes'][eye] = {'K': camera['K'], 'D': camera['D'], 'tool': tool, 'jaw_visibility': visibility}
            if quality is not None:
                for name in ('jaw_tip_region_visibility', 'image_quality'):
                    record['eyes'][eye][name] = quality[eye][name]
        panels = []; projections = {}
        for eye in ('left', 'right'):
            panel, projections[eye] = vector_panel(images[eye], record, eye, self.font, True, geometry, compact=self.compact)
            panels.append(panel)
        canvas = Image.new('RGB', (panels[0].width*2, panels[0].height))
        for i, panel in enumerate(panels): canvas.paste(panel, (i*panel.width, 0))
        return np.asarray(canvas), {'frame_id': frame.frame_id, 'timestamp_ns': frame.timestamp_ns,
            'source': prediction.source, 'visibility_source': prediction.visibility_source, 'valid': True,
            'values17': np.asarray(prediction.values17, float).tolist(),
            'projection_results': projections, 'postprocess_ms': (time.perf_counter()-started)*1000}

    def unavailable(self, frame, reason):
        """Display the current images with an error, without reusing a previous pose."""
        images = self._images(frame); canvas = Image.new('RGB', (self.width*2, self.height+50), '#461820')
        for i, eye in enumerate(('left', 'right')): canvas.paste(images[eye], (i*self.width, 50))
        ImageDraw.Draw(canvas).text((16, 10), '无有效预测 / NO VALID PREDICTION: '+str(reason)[:110],
                                   font=font_cached(self.font, 25), fill='white')
        return np.asarray(canvas)
