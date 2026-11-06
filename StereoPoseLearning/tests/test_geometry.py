"""Run with python -m unittest discover -s tests -p 'test_geometry.py'."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from stereo_pose.geometry import (FixedKinematics, matrix_to_quaternion_xyzw,
                                 matrix_to_rotation_6d, project_brown,
                                 quaternion_xyzw_to_matrix, rotation_6d_to_matrix,
                                 undistort_brown, values17)
from stereo_pose.losses import PoseLoss

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "instrument_model.json"
LEGACY = ROOT.parent / "CameraImagingConfig/src/TemporalStereo_v2"
if not MODEL_PATH.exists():
    MODEL_PATH = LEGACY / "validation_evidence/balanced_2100/new_calculations/instrument_model.json"


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.model = json.loads(MODEL_PATH.read_text())
        self.fk = FixedKinematics(self.model).double()

    def test_fk_matches_independent_scipy_reference(self):
        spec = importlib.util.spec_from_file_location("numpy_decoder", LEGACY / "pose_display/decoding.py")
        if not Path(spec.origin).exists():
            # The frozen independent reference is shipped with the new module
            # when it is deployed separately from the generator.
            spec = importlib.util.spec_from_file_location("numpy_decoder", ROOT / "reference" / "pose_display" / "decoding.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(113)
        for half in (0., .1, np.pi/6):
            rotation = Rotation.random(2, random_state=rng).as_matrix()
            C = torch.tensor(rng.normal(size=(2, 3))*5+[0, 0, 100], dtype=torch.float64)
            R = torch.tensor(rotation, dtype=torch.float64)
            joints = torch.tensor([-.42, .17, half], dtype=torch.float64)
            got = self.fk(C, R, joints)
            expected = module.decode(values17(C, R, joints).numpy(), self.model)
            for i, eye in enumerate(("left", "right")):
                points = np.stack([expected[eye][p] for p in ("C", "H1", "H2", "J1", "J2", "T")])
                np.testing.assert_allclose(got["points"][i], points, atol=1e-10, rtol=0)
                np.testing.assert_allclose(got["R_jaws"][i, 0], expected[eye]["R_jaw1"], atol=1e-12)

    def test_batch_and_geometry_gradients(self):
        C = torch.tensor([[[[2., 3., 100.], [1., 4., 101.]]]], dtype=torch.float64, requires_grad=True)
        r6 = torch.tensor([[[[1., .1, 0., 0., 1., .1], [1., 0., .2, .1, 1., 0.]]]],
                          dtype=torch.float64, requires_grad=True)
        q = torch.tensor([[[.2, -.1, .15]]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda c, r, j: self.fk(c, rotation_6d_to_matrix(r), j)["points"],
            (C, r6, q), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_rotation_roundtrip_and_degenerate_inputs(self):
        rotation = Rotation.from_rotvec([[0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0],
                                        [0, 0, np.pi], [.2, -.4, .7]]).as_matrix()
        R = torch.tensor(rotation, dtype=torch.float64)
        torch.testing.assert_close(rotation_6d_to_matrix(matrix_to_rotation_6d(R)), R, atol=1e-12, rtol=1e-12)
        q = matrix_to_quaternion_xyzw(R)
        torch.testing.assert_close(quaternion_xyzw_to_matrix(q), R, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(quaternion_xyzw_to_matrix(-q), R, atol=1e-12, rtol=1e-12)
        r6 = torch.zeros((2, 6), dtype=torch.float64, requires_grad=True)
        out = rotation_6d_to_matrix(r6)
        torch.testing.assert_close(torch.linalg.det(out), torch.ones(2, dtype=torch.float64))
        out.sum().backward()
        self.assertTrue(torch.isfinite(r6.grad).all())

    def test_distorted_projection_inverse_and_gradient(self):
        K = torch.tensor([[1059., 0., 573.5], [0., 1057., 425.5], [0., 0., 1.]], dtype=torch.float64)
        D = torch.tensor([-.3083, .5236, .000276, -.0021, .2094], dtype=torch.float64)
        points = torch.tensor([[-30., -20., 100.], [30., 20., 80.], [0., 0., 130.]],
                              dtype=torch.float64, requires_grad=True)
        uv = project_brown(points, K, D)
        rays = undistort_brown(uv, K, D)
        torch.testing.assert_close(rays, points[:, :2]/points[:, 2:], atol=1e-11, rtol=1e-11)
        self.assertTrue(torch.autograd.gradcheck(lambda p: undistort_brown(project_brown(p, K, D), K, D),
                                                (points,), atol=1e-5, rtol=1e-4))
        eyes = points[None].repeat(2, 1, 1)
        per_eye_K = K[None].repeat(2, 1, 1)
        per_eye_D = D[None].repeat(2, 1)
        self.assertEqual(project_brown(eyes, per_eye_K, per_eye_D).shape, (2, 3, 2))
        one = points[:2]
        self.assertEqual(project_brown(one, per_eye_K, per_eye_D).shape, (2, 2))

    def test_loss_backpropagates_through_fk(self):
        C = torch.tensor([[[[0., 0., 100.], [2., 0., 100.]]]], dtype=torch.float64, requires_grad=True)
        r6 = torch.tensor([[[[1., 0., 0., 0., 1., 0.], [1., 0., 0., 0., 1., 0.]]]],
                          dtype=torch.float64, requires_grad=True)
        joints = torch.tensor([[[.2, .1, .15]]], dtype=torch.float64, requires_grad=True)
        pred = {"C": C, "R": rotation_6d_to_matrix(r6), "joints": joints}
        target = {"C": C.detach()+1., "R": pred["R"].detach(), "joints": joints.detach()+.02}
        target.update(self.fk(target["C"], target["R"], target["joints"]))
        loss, stats = PoseLoss(self.fk)(pred, target)
        loss.backward()
        for p in (C, r6, joints):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertGreater(stats["tip_mm"].item(), 0)


if __name__ == "__main__":
    unittest.main()
