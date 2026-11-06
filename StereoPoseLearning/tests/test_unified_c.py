"""Unified C has one image ray and learned depth, with end-to-end gradients.

CPU tests use a tiny deterministic backbone; no pretrained downloads or simulator.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from stereo_pose.data import StereoDataset, build_manifest
from stereo_pose.geometry import project_brown
from stereo_pose.model import StereoPoseNet, joint_bounds_from_model
from stereo_pose.translation import landmark_ray_depth_position

ROOT = Path(__file__).resolve().parents[1]


class TinyBackbone(nn.Module):
    """Only replaces the image backbone; all new routing/heads are real modules."""
    class Info:
        def channels(self):
            return [8, 16, 24, 32]

    def __init__(self):
        super().__init__()
        self.feature_info = self.Info()
        self.layers = nn.ModuleList([nn.Conv2d(3, 8, 4, stride=4), nn.Conv2d(8, 16, 2, stride=2),
                                     nn.Conv2d(16, 24, 2, stride=2), nn.Conv2d(24, 32, 2, stride=2)])

    def forward(self, x):
        out = []
        for layer in self.layers:
            x = torch.nn.functional.gelu(layer(x))
            out.append(x)
        return out


@pytest.fixture
def net(monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(93)
    monkeypatch.setattr("stereo_pose.model.timm.create_model", lambda *a, **k: TinyBackbone())
    config = json.loads((ROOT / "configs/default.json").read_text())["model"]
    config.update(feature_dim=16, hidden_dim=32, fusion_layers=1, attention_heads=4,
                  part_crop_size=32, part_mask_dilation_px=0,
                  c_translation_head="landmark_ray_depth_v1")
    instrument = json.loads((ROOT / "assets/instrument_model.json").read_text())
    return StereoPoseNet(config, joint_bounds_from_model(instrument), pretrained=False).eval()


def inputs(time=3):
    images = torch.randn(1, time, 2, 3, 64, 96)
    raw = torch.randint(0, 256, (1, time, 2, 3, 128, 192), dtype=torch.uint8)
    K = torch.tensor([[[84., 0., 45.7], [0., 91., 30.2], [0., 0., 1.]],
                      [[98., 0., 49.1], [0., 87., 32.6], [0., 0., 1.]]]).expand(1, time, 2, 3, 3).clone()
    D = torch.tensor([[-.25, .12, .001, -.002, .025],
                      [-.18, .06, -.002, .001, .012]]).expand(1, time, 2, 5).clone()
    return images, K, D, raw


def test_nonzero_distortion_different_intrinsics_round_trip_and_depth_invariance():
    # Distinct cameras and off-axis points ensure this cannot pass with K or D ignored.
    K = torch.tensor([[[530., 0., 301.2], [0., 528., 197.7], [0., 0., 1.]],
                      [[605., 0., 326.8], [0., 578., 171.6], [0., 0., 1.]]], dtype=torch.float64)
    D = torch.tensor([[-.308, .524, .0003, -.0021, .209],
                      [-.22, .13, -.0012, .0015, .04]], dtype=torch.float64)
    truth = torch.tensor([[25., -12., 85.], [-16., 21., 132.]], dtype=torch.float64)
    uv = project_brown(truth, K, D)
    C = landmark_ray_depth_position(uv, truth[:, 2:3], K, D)
    torch.testing.assert_close(C.double(), truth, atol=5e-6, rtol=1e-7)
    torch.testing.assert_close(project_brown(C.double(), K, D), uv, atol=1e-4, rtol=0)
    deeper = landmark_ray_depth_position(uv, truth[:, 2:3] * 1.7, K, D)
    torch.testing.assert_close(project_brown(deeper.double(), K, D), uv, atol=1e-4, rtol=0)
    torch.testing.assert_close(deeper, C * 1.7, atol=2e-5, rtol=1e-7)
    wrong = landmark_ray_depth_position(uv, truth[:, 2:3], K, D * 0)
    assert (wrong - C).abs().max() > .1


def test_decoder_has_gradients_to_uv_and_depth():
    uv = torch.tensor([[68., 21.], [30., 47.]], requires_grad=True)
    z = torch.tensor([[95.], [110.]], requires_grad=True)
    _, K, D, _ = inputs(1)
    C = landmark_ray_depth_position(uv, z, K[0, 0], D[0, 0])
    (C * torch.tensor([[.7, -.3, .2], [-.4, .8, .1]])).sum().backward()
    for t in (uv, z):
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert torch.all(t.grad.abs() > 1e-6)


def test_new_model_has_no_independent_xy_or_old_pose_head(net):
    assert not hasattr(net, "pose_heads")
    assert not any("pose_heads" in name or "ray_xy" in name for name, _ in net.named_parameters())
    for depth, rotation in zip(net.depth_heads, net.rotation_heads):
        assert depth[0].in_features == net.config["hidden_dim"] + net.config["feature_dim"]
        assert depth[-1].out_features == 1
        assert rotation[0].in_features == net.config["hidden_dim"]
        assert rotation[-1].out_features == 6


def test_c_projection_is_the_same_landmark_and_changing_depth_does_not_move_it(net):
    images, K, D, raw = inputs(1)
    with torch.no_grad():
        before, _ = net(images, K, D, hires_images=raw)
        for head in net.depth_heads:
            head[-1].bias.add_(.4)
        after, _ = net(images, K, D, hires_images=raw)
    torch.testing.assert_close(before["c_uv"], before["landmark_uv"][..., 0, :], atol=0, rtol=0)
    torch.testing.assert_close(before["c_uv"], after["c_uv"], atol=0, rtol=0)
    assert (before["c_depth"] - after["c_depth"]).abs().min() > 1
    for p in (before, after):
        torch.testing.assert_close(project_brown(p["C"].double(), K.double(), D.double()),
                                   p["c_uv"].double(), atol=1e-4, rtol=0)
    torch.testing.assert_close(before["R"], after["R"], atol=0, rtol=0)
    torch.testing.assert_close(before["joints"], after["joints"], atol=0, rtol=0)


def test_C_loss_reaches_C_heatmap_and_each_depth_head_without_detach(net):
    images, K, D, raw = inputs(1)
    captured = {}
    def hook(module, args, value):
        value.retain_grad()
        captured["heat"] = value
    handle = net.heatmap_head.register_forward_hook(hook)
    pred, _ = net(images, K, D, hires_images=raw)
    pred["c_depth"].retain_grad()
    target = pred["C"].detach() + torch.tensor([2., -1., 4.])
    (pred["C"] - target).square().mean().backward()
    handle.remove()
    grad = captured["heat"].grad
    assert grad is not None and torch.isfinite(grad).all()
    assert (grad[:, 0].abs().sum((-2, -1)) > 0).all()
    assert pred["c_depth"].grad is not None and (pred["c_depth"].grad.abs() > 0).all()
    for head in net.depth_heads:
        assert head[-1].weight.grad is not None and head[-1].weight.grad.abs().sum() > 0


def test_window_stream_causality_and_existing_17_value_constraints(net):
    images, K, D, raw = inputs(3)
    with torch.no_grad():
        pred, state = net(images, K, D, hires_images=raw)
        values = net.export_values(pred)
        streamed, hidden = [], None
        for t in range(3):
            step, hidden = net(images[:, t:t+1], K[:, t:t+1], D[:, t:t+1], hidden, raw[:, t:t+1])
            streamed.append(net.export_values(step))
        torch.testing.assert_close(values, torch.cat(streamed, 1), atol=3e-5, rtol=1e-5)
        altered_images, altered_raw = images.clone(), raw.clone()
        altered_images[:, 2] *= -2
        altered_raw[:, 2] = 255 - altered_raw[:, 2]
        altered, _ = net(altered_images, K, D, hires_images=altered_raw)
        torch.testing.assert_close(values[:, :2], net.export_values(altered)[:, :2], atol=1e-6, rtol=1e-6)
    assert values.shape == (1, 3, 17) and torch.isfinite(values).all()
    assert state.shape == (3, 2, net.config["hidden_dim"])
    assert torch.all(pred["C"][..., 2] > 0)
    for q in (values[..., 3:7], values[..., 10:14]):
        torch.testing.assert_close(q.norm(dim=-1), torch.ones(1, 3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pred["R"].transpose(-1, -2) @ pred["R"],
                               torch.eye(3).expand(1, 3, 2, 3, 3), atol=1e-6, rtol=1e-6)
    assert torch.all(torch.linalg.det(pred["R"]) > .99999)
    assert torch.all(pred["joints"] >= net.joint_lower)
    assert torch.all(pred["joints"] <= net.joint_upper)


def test_hidden_points_still_have_amodal_heatmap_and_C_targets(tmp_path):
    # Reuse only the existing real-schema tiny-file fixture; no rendering involved.
    spec = importlib.util.spec_from_file_location("_data_fixture", ROOT / "tests/test_data.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    source = tmp_path / "dataset"
    source.mkdir()
    fixture.save_plan(source, n=1)
    path = fixture.save_label(source, 0)
    label = json.loads(path.read_text())
    for eye in ("left", "right"):
        label["eyes"][eye]["fixed_tip_visibility"] = [{"visible": False}, {"visible": False}]
        label["eyes"][eye]["jaw_tip_region_visibility"] = [{"region_visible": False}, {"region_visible": False}]
    path.write_text(json.dumps(label))
    manifest = build_manifest([source], tmp_path / "manifest.json")
    data = StereoDataset(manifest, "train", window=1, image_size=(16, 32))
    assert len(data) == 1
    frame = data[0]
    assert frame["heatmap_valid"].bool().all()
    assert (frame["heatmaps"].flatten(-2).sum(-1) > 0).all()
    assert torch.isfinite(frame["C"]).all()
    assert all(not p["visible"] for p in frame["metadata"][0]["fixed_tip_visibility"]["left"])
