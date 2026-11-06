"""Image-pair prediction and the existing, approved surgical vector overlay.

The callback returned by build_predictor is also accepted by
``python -m pose_display.video --predictor stereo_pose.predict:build_predictor``.
Only RGB and individual-camera K/D enter the network. Ground-truth labels,
inter-camera extrinsics, baseline and ground-truth visibility are never read.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch

from .data import resize_intrinsics
from .model import StereoPoseNet, joint_bounds_from_model


def _display_api():
    """Load the copied renderer; this file does not redefine its drawing rules."""
    reference = Path(__file__).resolve().parents[1] / "reference"
    if str(reference) not in sys.path:
        sys.path.insert(0, str(reference))
    try:
        return importlib.import_module("pose_display.pipeline")
    except ModuleNotFoundError as error:
        raise RuntimeError("Copy the approved pose_display package into reference/pose_display") from error


def _camera_inputs(calibration: dict) -> dict:
    """Copy only imaging calibration, explicitly excluding camera relation truth."""
    image = calibration["image"]
    if image.get("space") != "raw_distorted":
        raise ValueError("Expected raw_distorted RGB and matching Brown5 intrinsics/distortion")
    width, height = int(image["width"]), int(image["height"])
    if min(width, height) <= 0:
        raise ValueError("Invalid calibration image size")
    result = {"image": {"width": width, "height": height, "space": "raw_distorted"}}
    for eye in ("left", "right"):
        K = np.asarray(calibration[eye]["K"], dtype=np.float64)
        D = np.asarray(calibration[eye]["D"], dtype=np.float64)
        if K.shape != (3, 3) or D.shape != (5,) or not np.isfinite(K).all() or not np.isfinite(D).all():
            raise ValueError("Expected finite 3x3 K and Brown5 D=[k1,k2,p1,p2,k3]")
        if (K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])
                or abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12):
            raise ValueError("Unsupported intrinsic matrix")
        result[eye] = {"K": K.tolist(), "D": D.tolist()}
    return result


def _frame_identity(frame_id: str):
    match = re.fullmatch(r"(.*?)(\d+)", str(frame_id))
    return (match.group(1), int(match.group(2))) if match else (str(frame_id), None)


class StereoPredictor:
    """A causal stream callback; call reset() when starting a new capture.

    A saved checkpoint always determines the model and geometric definition.
    Runtime config only chooses the device and stream-reset policy. An isolated
    image pair has an empty recurrent history; video receives past frames only.
    """

    def __init__(self, checkpoint, *, device="auto", allow_diagnostic=False, fps=60.0):
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        for key in ("model", "config", "instrument_model", "diagnostic", "epoch"):
            if key not in checkpoint:
                raise ValueError(f"Checkpoint is missing {key}")
        self.diagnostic = bool(checkpoint["diagnostic"])
        if self.diagnostic and not allow_diagnostic:
            raise ValueError("Diagnostic checkpoint is not a trained predictor; use --allow-diagnostic only for a smoke test")
        self.training_config = checkpoint["config"]
        self.instrument_model = checkpoint["instrument_model"]
        self.epoch = int(checkpoint["epoch"])
        self.image_size = tuple(map(int, self.training_config["image_size"]))
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("Checkpoint has an invalid network image size")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Stream fps must be positive")
        self.period_ns = 1e9 / float(fps)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = StereoPoseNet(self.training_config["model"],
                                   joint_bounds_from_model(self.instrument_model), pretrained=False)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device).eval()
        self.pipeline = _display_api()
        self.reset_count = 0
        self.reset()

    def reset(self):
        self.hidden = None
        self.last_timestamp_ns = None
        self.last_frame_identity = None
        self.last_camera_fingerprint = None
        self.reset_count += 1

    def _preprocess(self, frame, calibration):
        height, width = self.image_size
        source_size = (calibration["image"]["height"], calibration["image"]["width"])
        images, intrinsics, distortions = [], [], []
        mean = np.asarray([0.485, 0.456, 0.406], np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], np.float32)[:, None, None]
        for eye in ("left", "right"):
            rgb = np.asarray(getattr(frame, eye + "_rgb"))
            if rgb.dtype != np.uint8 or rgb.shape != source_size + (3,):
                raise ValueError("Frame RGB dimensions/type disagree with its calibration")
            image = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR),
                               dtype=np.float32) / 255.0
            images.append((image.transpose(2, 0, 1) - mean) / std)
            intrinsics.append(resize_intrinsics(np.asarray(calibration[eye]["K"]), source_size, self.image_size))
            distortions.append(calibration[eye]["D"])
        return tuple(torch.as_tensor(np.stack(values), dtype=torch.float32, device=self.device)[None, None]
                     for values in (images, intrinsics, distortions))

    @torch.inference_mode()
    def __call__(self, frame, calibration):
        try:
            cameras = _camera_inputs(calibration)
            fingerprint = json.dumps(cameras, sort_keys=True, separators=(",", ":"))
            stamp = int(frame.timestamp_ns)
            identity = _frame_identity(frame.frame_id)
            discontinuous = self.last_timestamp_ns is not None and (
                stamp <= self.last_timestamp_ns
                or stamp - self.last_timestamp_ns > self.period_ns * 1.5 + 2
                or fingerprint != self.last_camera_fingerprint
                or identity[0] != self.last_frame_identity[0]
                or (identity[1] is not None and self.last_frame_identity[1] is not None
                    and identity[1] != self.last_frame_identity[1] + 1))
            if discontinuous:
                self.reset()
            inputs = self._preprocess(frame, cameras)
            kwargs = {}
            if self.training_config['model'].get('foreground_parts', False):
                raw = np.stack([np.asarray(getattr(frame, eye+'_rgb')).transpose(2,0,1)
                                for eye in ('left','right')])
                kwargs['hires_images'] = torch.as_tensor(raw.copy(), dtype=torch.uint8, device=self.device)[None,None]
            prediction, hidden = self.model(*inputs, hidden=self.hidden, **kwargs)
            value = self.model.export_values(prediction)[0, 0].detach().cpu().numpy()
            if value.shape != (17,) or not np.isfinite(value).all():
                raise ValueError("Model produced nonfinite or malformed values17")
            self.hidden = hidden.detach()
            self.last_timestamp_ns = stamp
            self.last_frame_identity = identity
            self.last_camera_fingerprint = fingerprint
            return self.pipeline.PosePrediction(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns,
                                                values17=value, source="prediction",
                                                visibility=None, visibility_source="unknown")
        except Exception:
            # A failed frame cannot leave stale recurrent state for the next one.
            self.reset()
            raise


def build_predictor(config: dict):
    """Factory for pose_display.video; config is JSON, not a truth-label file."""
    if "checkpoint" not in config:
        raise ValueError("Predictor config needs a checkpoint path")
    path = Path(config["checkpoint"]).expanduser()
    if not path.is_absolute():
        path = Path(config.get("_config_dir", ".")) / path
    return StereoPredictor(path, device=config.get("device", "auto"),
                           allow_diagnostic=config.get("allow_diagnostic", False),
                           fps=float(config.get("fps", 60.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "left", "right", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--font")
    parser.add_argument("--frame-id", default="000000")
    parser.add_argument("--timestamp-ns", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new empty prediction output directory")
    cameras = _camera_inputs(json.loads(args.calibration.read_text()))
    arrays = []
    for path in (args.left, args.right):
        with Image.open(path) as image:
            arrays.append(np.asarray(image.convert("RGB")))
    predictor = StereoPredictor(args.checkpoint, device=args.device, allow_diagnostic=args.allow_diagnostic)
    frame = predictor.pipeline.StereoFrame(args.frame_id, args.timestamp_ns, *arrays)
    prediction = predictor(frame, cameras)
    overlay = predictor.pipeline.StereoOverlay(predictor.instrument_model, cameras, args.font, compact=True)
    canvas, metadata = overlay.render(frame, prediction)
    metadata.update({"checkpoint": str(predictor.checkpoint_path), "checkpoint_epoch": predictor.epoch,
                     "diagnostic_checkpoint": predictor.diagnostic,
                     "prediction_accuracy_validated": False,
                     "network_image_size": list(predictor.image_size),
                     "coordinate_frame": "camera_optical_x_right_y_down_z_forward",
                     "position_unit": "mm", "angle_unit": "rad", "quaternion_order": "xyzw",
                     "note": "Single pair with empty recurrent history; no ground truth was read. "
                             "Diagnostic outputs, if enabled, test wiring only and are not usable pose estimates."})
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(args.output / "overlay.png")
    for name, data in (("prediction", metadata), ("calibration", cameras),
                       ("instrument_model", predictor.instrument_model)):
        (args.output / (name + ".json")).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                                               allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "source": "prediction",
                      "diagnostic_checkpoint": predictor.diagnostic, "values17": prediction.values17.tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
