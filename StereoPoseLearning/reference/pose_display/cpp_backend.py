"""Thin adapter to the C++ renderer. No Python geometry or per-arrow drawing.

Native callers can use native/overlay.h directly. This adapter keeps the current
PyTorch predictor API; ctypes releases the GIL during the native render call.
"""
import ctypes as ct
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
from .pipeline import find_font


class CppStereoOverlay:
    def __init__(self, instrument_model, calibration, font=None, *, library=None,
                 geometry_log=False, display_mode='vectors'):
        if display_mode not in ('vectors','full'): raise ValueError('Invalid display mode')
        self.display_mode=display_mode
        native = Path(__file__).parent / 'native'
        path = Path(library or os.environ.get('SURGICAL_OVERLAY_LIBRARY',
                    str(native / 'build' / 'libsurgical_overlay.so')))
        if not path.is_file():
            raise FileNotFoundError(f'Build the C++ renderer first; library missing: {path}')
        self._lib = lib = ct.CDLL(str(path.resolve()))
        lib.sd_abi_version.restype = ct.c_int
        if lib.sd_abi_version() != 2:
            raise RuntimeError('Incompatible C++ overlay ABI')
        lib.sd_create.argtypes = [ct.c_char_p, ct.c_char_p]; lib.sd_create.restype = ct.c_void_p
        lib.sd_destroy.argtypes = [ct.c_void_p]; lib.sd_destroy.restype = None
        lib.sd_last_error.restype = ct.c_char_p
        lib.sd_compute.argtypes = [ct.c_void_p, ct.c_void_p]; lib.sd_compute.restype = ct.c_int
        lib.sd_render.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_ssize_t,
            ct.c_void_p, ct.c_ssize_t, ct.c_void_p, ct.c_ssize_t, ct.c_void_p,
            ct.c_char_p, ct.c_int64, ct.c_int]
        lib.sd_render.restype = ct.c_int
        lib.sd_inspect.argtypes = [ct.c_void_p, ct.c_int]; lib.sd_inspect.restype = ct.c_char_p
        self.width = int(calibration['image']['width']); self.height = int(calibration['image']['height'])
        self.header_height=0 if display_mode=='vectors' else 50
        self.canvas_shape=(self.height+self.header_height,self.width*2,3)
        self.font = find_font(font)
        config = json.dumps({'model': instrument_model, 'calibration': calibration,'display_mode':display_mode}, allow_nan=False)
        self._handle = lib.sd_create(config.encode(), self.font.encode())
        if not self._handle:
            raise ValueError(self._error())
        self._lock = threading.Lock()
        self.geometry_log = geometry_log

    def _error(self):
        return self._lib.sd_last_error().decode('utf-8', errors='replace')

    def close(self):
        if getattr(self, '_handle', None):
            with self._lock:
                if self._handle:
                    self._lib.sd_destroy(self._handle); self._handle = None

    def __del__(self):
        if getattr(self, '_handle', None):
            self._lib.sd_destroy(self._handle); self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _pose(self, values):
        a = np.ascontiguousarray(values, dtype=np.float64)
        if a.shape != (17,) or not np.isfinite(a).all():
            raise ValueError('Expected 17 finite pose values')
        return a

    def _images(self, frame, out=None):
        images = []
        for eye in ('left', 'right'):
            a = np.asarray(getattr(frame, eye + '_rgb'))
            if a.shape != (self.height, self.width, 3) or a.dtype != np.uint8:
                raise ValueError('RGB size/type disagrees with calibration')
            # Native API supports padded rows, but not reversed/channel-strided arrays.
            if a.strides[2] != 1 or a.strides[1] != 3 or a.strides[0] < self.width * 3:
                a = np.ascontiguousarray(a)
            images.append(a)
        if out is None:
            out = np.empty(self.canvas_shape, dtype=np.uint8)
        if (not isinstance(out, np.ndarray) or out.shape != self.canvas_shape
                or out.dtype != np.uint8 or not out.flags.c_contiguous or not out.flags.writeable):
            raise ValueError('Output must be a writable contiguous RGB stereo canvas')
        if any(np.shares_memory(out, a) for a in images):
            raise ValueError('Output must not overlap input images')
        return images, out

    def _inspect(self, detail):
        data = self._lib.sd_inspect(self._handle, int(detail))
        if data is None:
            raise ValueError(self._error())
        return json.loads(data)

    def compute_geometry(self, values, *, detailed=True):
        """CPU-only diagnostic; no image rendering or model inference."""
        values = self._pose(values)
        with self._lock:
            if self._lib.sd_compute(self._handle, values.ctypes.data):
                raise ValueError(self._error())
            return self._inspect(detailed)

    def render(self, frame, prediction, quality=None, *, out=None):
        """Render compact RGB; an explicit out buffer is overwritten on the next call.

        Without out, the returned image owns independent memory. Detailed native
        geometry is returned only when geometry_log=True (schema native_v1).
        """
        start = time.perf_counter()
        if (frame.frame_id, frame.timestamp_ns) != (prediction.frame_id, prediction.timestamp_ns):
            raise ValueError('Prediction belongs to another frame; stale overlays are rejected')
        if prediction.source not in ('prediction', 'ground_truth_replay'):
            raise ValueError('Explicit prediction/replay provenance required')
        if not isinstance(frame.timestamp_ns, int) or not -(2**63) <= frame.timestamp_ns < 2**63:
            raise ValueError('Timestamp must fit signed int64 nanoseconds')
        values = self._pose(prediction.values17)
        images, out = self._images(frame, out)
        visibility = np.full(4, -1, np.int32)
        if prediction.visibility is not None:
            if prediction.visibility_source == 'unknown':
                raise ValueError('Visibility needs explicit provenance')
            for e, eye in enumerate(('left', 'right')):
                records = prediction.visibility[eye]
                if len(records) != 2:
                    raise ValueError('Expected two fixed-tip visibility records')
                for i, item in enumerate(records):
                    v = item['visible']
                    if v is not None and not isinstance(v, (bool, np.bool_)):
                        raise ValueError('Invalid visibility state')
                    visibility[e * 2 + i] = (-1 if v is None else 1 if v else
                        {'tissue': 2, 'instrument': 3}.get(item.get('occluder_kind'), 4))
        with self._lock:
            result = self._lib.sd_render(self._handle, values.ctypes.data,
                images[0].ctypes.data, images[0].strides[0], images[1].ctypes.data, images[1].strides[0],
                out.ctypes.data, out.strides[0], visibility.ctypes.data,
                str(frame.frame_id).encode(), frame.timestamp_ns,
                int(prediction.source == 'ground_truth_replay'))
            if result:
                raise ValueError(self._error())
            projection = self._inspect(True) if self.geometry_log else None
        meta = {'frame_id': frame.frame_id, 'timestamp_ns': frame.timestamp_ns,
            'source': prediction.source, 'visibility_source': prediction.visibility_source,
            'valid': True, 'values17': values.tolist(), 'backend': 'cpp',
            'display_mode': self.display_mode,
            'geometry_schema': 'native_v1', 'postprocess_ms': (time.perf_counter() - start) * 1000}
        if projection is not None:
            meta['projection_results'] = projection
        if quality is not None:
            meta['quality'] = quality  # Independent observations, never inferred from pose.
        return out, meta

    def unavailable(self, frame, reason):
        # Error-only path: show current RGB and an error, never old arrows.
        from PIL import Image, ImageDraw
        from .renderer import font_cached
        images, out = self._images(frame)
        out[:] = (70, 24, 32)
        out[self.header_height:, :self.width] = images[0]; out[self.header_height:, self.width:] = images[1]
        canvas = Image.fromarray(out)
        ImageDraw.Draw(canvas).text((16, 10), '无有效预测 / NO VALID PREDICTION: ' + str(reason)[:110],
                                   font=font_cached(self.font, 25), fill='white')
        return np.asarray(canvas)
