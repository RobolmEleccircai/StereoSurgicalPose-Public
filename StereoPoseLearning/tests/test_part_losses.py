"""Part supervision contracts and an optional anchored-geometry ablation."""
import json
from pathlib import Path
import unittest

import torch

from stereo_pose.geometry import FixedKinematics, project_brown
from stereo_pose.guidance import landmark_locations
from stereo_pose.losses import PoseLoss, local_heatmap_targets


ROOT = Path(__file__).resolve().parents[1]


class PartLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fk = FixedKinematics(json.loads((ROOT/'assets/instrument_model.json').read_text())).double()
        self.C = torch.tensor([[[[0., 0., 100.], [2., 0., 100.]]]], dtype=torch.float64)
        self.R = torch.eye(3, dtype=torch.float64).expand(1, 1, 2, 3, 3).clone()
        self.q = torch.tensor([[[.1, -.1, .15]]], dtype=torch.float64)
        self.target = {'C': self.C, 'R': self.R, 'joints': self.q,
            'K': torch.tensor([[100., 0., 49.5], [0., 100., 49.5], [0., 0., 1.]],
                              dtype=torch.float64).expand(1, 1, 2, 3, 3),
            'D': torch.zeros(1, 1, 2, 5, dtype=torch.float64),
            'image_size': torch.tensor([[[100., 100.]]], dtype=torch.float64)}
        self.target.update(self.fk(self.C, self.R, self.q))

    def weights(self, *enabled):
        return {k: float(k in enabled) for k in PoseLoss.DEFAULT_WEIGHTS}

    def test_align_corners_false_target_and_validity(self):
        # With extent 10 and width 20, u=14.5 maps exactly to column 4;
        # v=13.5 maps exactly to row 3. Both are pixel-centre coordinates.
        points = torch.tensor([[[[14.5, 13.5, 1.], [45., 13.5, 1.],
                                  [14.5, 13.5, -1.]]]], dtype=torch.float64).expand(1, 2, 3, 3)
        K = torch.eye(3, dtype=torch.float64).expand(1, 2, 3, 3)
        D = torch.zeros(1, 2, 5, dtype=torch.float64)
        centres = torch.tensor([20., 20.], dtype=torch.float64).expand(1, 2, 2, 2)
        extents = torch.full_like(centres, 10.)
        maps, valid = local_heatmap_targets(points, K, D, centres, extents,
                                             (20, 20), torch.tensor([[40., 40.]]))
        self.assertEqual(tuple(maps.shape), (1, 2, 2, 3, 20, 20))
        self.assertTrue(valid[..., 0].all())
        self.assertFalse(valid[..., 1:].any())
        torch.testing.assert_close(maps[..., 0, 3, 4], torch.ones(1, 2, 2, dtype=torch.float64))
        self.assertTrue(torch.isfinite(maps).all())
        self.assertEqual(float(maps[..., 1:, :, :].sum()), 0.)
        self.assertTrue(torch.equal(maps[..., 0, :, :].flatten(-2).argmax(-1),
                                    torch.full((1, 2, 2), 3*20+4)))

    def test_point_inside_roi_but_outside_image_is_invalid(self):
        points = torch.tensor([[[[45., 10., 1.]]]])
        K = torch.eye(3).expand(1, 1, 3, 3)
        D = torch.zeros(1, 1, 5)
        centres = torch.tensor([[[[45., 10.]]]])
        maps, valid = local_heatmap_targets(points, K, D, centres, torch.ones_like(centres)*5,
                                             (8, 8), torch.tensor([[40., 40.]]))
        self.assertFalse(valid.any())
        self.assertEqual(float(maps.sum()), 0.)

    def test_anchored_joint_gradient_does_not_depend_on_predicted_shaft(self):
        def joint_gradient(offset, decouple):
            q = (self.q + torch.tensor([.08, -.06, .04], dtype=torch.float64)).requires_grad_()
            pred = {'C': self.C + offset, 'R': self.R, 'joints': q}
            loss, _ = PoseLoss(self.fk, self.weights('geometry', 'part_rotation', 'projection'),
                               decouple_geometry=decouple)(pred, self.target)
            return torch.autograd.grad(loss, q)[0]
        offsets = (torch.zeros(3, dtype=torch.float64), torch.tensor([3., -2., 5.], dtype=torch.float64))
        anchor_a, anchor_b = [joint_gradient(x, True) for x in offsets]
        torch.testing.assert_close(anchor_a, anchor_b, atol=1e-12, rtol=1e-12)
        self.assertGreater(float(anchor_a.abs().sum()), 0)
        raw_a, raw_b = [joint_gradient(x, False) for x in offsets]
        self.assertGreater(float((raw_a-raw_b).abs().max()), 1e-5)

    def test_anchored_geometry_has_finite_body_and_joint_gradients(self):
        C = (self.C + torch.tensor([1., 2., 3.], dtype=torch.float64)).requires_grad_()
        q = (self.q + .03).requires_grad_()
        pred = {'C': C, 'R': self.R, 'joints': q}
        loss, _ = PoseLoss(self.fk, self.weights('geometry', 'part_rotation', 'projection'),
                           decouple_geometry=True)(pred, self.target)
        loss.backward()
        for value in (C, q):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.)

    def test_local_auxiliary_uses_existing_amodal_points_and_detaches_crop(self):
        logits = torch.zeros(1, 1, 2, 2, 3, 16, 16, requires_grad=True)
        centres = torch.full((1, 1, 2, 2, 2), 49.5, dtype=torch.float64, requires_grad=True)
        extents = torch.full_like(centres, 49.5, requires_grad=True)
        pred = {'C': self.C, 'R': self.R, 'joints': self.q,
                'part_heatmap_logits': logits, 'part_roi_centres': centres,
                'part_roi_half_extents': extents}
        # All-zero visible mask must NOT discard occluded amodal keypoints.
        target = {**self.target, 'masks': torch.zeros(1, 1, 2, 1, 25, 25)}
        loss, stats = PoseLoss(self.fk, self.weights('local_heatmap'))(pred, target)
        loss.backward()
        self.assertEqual(float(stats['local_heatmap_valid_points']), 12.)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        self.assertIsNone(centres.grad)
        self.assertIsNone(extents.grad)

    def test_new_losses_are_disabled_for_legacy_predictions(self):
        pred = {'C': self.C + 1., 'R': self.R, 'joints': self.q + .02}
        default, _ = PoseLoss(self.fk)(pred, self.target)
        explicit, _ = PoseLoss(self.fk, {'local_heatmap': 0.}, decouple_geometry=False)(pred, self.target)
        torch.testing.assert_close(default, explicit, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'predicted part maps'):
            PoseLoss(self.fk, {'local_heatmap': 1.})(pred, self.target)

    def test_detector_uv_loss_reaches_heatmaps_and_not_pose_outputs(self):
        logits = torch.randn(1, 1, 2, 5, 25, 25, dtype=torch.float64, requires_grad=True)
        xy, _ = landmark_locations(logits)
        uv = (xy+1)*50-.5
        C = (self.C+3).requires_grad_()
        q = (self.q+.1).requires_grad_()
        pred = {'C': C, 'R': self.R, 'joints': q, 'landmark_uv': uv}
        target = {**self.target, 'heatmap_valid': torch.ones(1, 1, 2, 5, dtype=torch.bool),
                  'masks': torch.zeros(1, 1, 2, 1, 25, 25)}
        loss, stats = PoseLoss(self.fk, self.weights('landmark_uv'))(pred, target)
        loss.backward()
        self.assertGreater(float(stats['loss/landmark_uv']), 0.)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        for value in (C, q):
            self.assertTrue(value.grad is None or torch.count_nonzero(value.grad) == 0)

    def test_detector_uv_uses_gt_and_ignores_inapplicable_points(self):
        expected = project_brown(self.target['points'], self.target['K'], self.target['D'])[..., :5, :]
        valid = torch.ones(1, 1, 2, 5, dtype=torch.bool)
        valid[..., 4] = False
        uv = expected.clone(); uv[..., 4, :] += 1000
        target = {**self.target, 'heatmap_valid': valid}
        criterion = PoseLoss(self.fk, self.weights('landmark_uv'))
        pred = {'C': self.C+20, 'R': self.R, 'joints': self.q+.4, 'landmark_uv': uv}
        loss, _ = criterion(pred, target)
        self.assertEqual(float(loss), 0.)
        changed = {**pred, 'C': self.C-10, 'joints': self.q-.2}
        other, _ = criterion(changed, target)
        torch.testing.assert_close(loss, other, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'detector coordinates'):
            criterion({k:v for k,v in pred.items() if k != 'landmark_uv'}, target)


if __name__ == '__main__':
    unittest.main()
