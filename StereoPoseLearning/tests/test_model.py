"""Causal output contract without downloading weights or using a simulator."""
import json
from pathlib import Path
import unittest
import torch
from stereo_pose.model import StereoPoseNet, joint_bounds_from_model

ROOT = Path(__file__).resolve().parents[1]


class ModelContractTest(unittest.TestCase):
    def test_causal_stream_and_17_value_geometry(self):
        torch.set_num_threads(2)
        torch.manual_seed(16)
        config = json.loads((ROOT / "configs/default.json").read_text())
        instrument = json.loads((ROOT / "assets/instrument_model.json").read_text())
        model = StereoPoseNet(config["model"], joint_bounds_from_model(instrument), pretrained=False).eval()
        images = torch.randn(1, 3, 2, 3, 64, 96)
        K = torch.tensor([[90., 0., 47.5], [0., 90., 31.5], [0., 0., 1.]]).expand(1, 3, 2, 3, 3)
        D = torch.zeros(1, 3, 2, 5)
        with torch.no_grad():
            pred, _ = model(images, K, D)
            values = model.export_values(pred)
            stream, state = [], None
            for t in range(3):
                p, state = model(images[:, t:t+1], K[:, t:t+1], D[:, t:t+1], state)
                stream.append(model.export_values(p))
            torch.testing.assert_close(values, torch.cat(stream, 1), atol=3e-5, rtol=1e-5)
            future_changed = images.clone(); future_changed[:, 2] *= -1
            changed, _ = model(future_changed, K, D)
            torch.testing.assert_close(values[:, :2], model.export_values(changed)[:, :2], atol=1e-6, rtol=1e-6)
            self.assertGreater(float((values[:, 2] - model.export_values(changed)[:, 2]).abs().max()), 1e-7)
        self.assertEqual(tuple(values.shape), (1, 3, 17))
        for q in (values[..., 3:7], values[..., 10:14]):
            torch.testing.assert_close(q.norm(dim=-1), torch.ones(1, 3), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.all(pred["joints"] >= model.joint_lower))
        self.assertTrue(torch.all(pred["joints"] <= model.joint_upper))
        self.assertTrue(torch.all(pred["C"][..., 2] > 0))
        torch.testing.assert_close(pred["R"].transpose(-1, -2) @ pred["R"],
                                   torch.eye(3).expand(1, 3, 2, 3, 3), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
