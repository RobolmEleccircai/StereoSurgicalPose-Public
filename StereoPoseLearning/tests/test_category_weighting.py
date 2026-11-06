"""Category loss weights stay exact with unequal visibility and microbatches."""
from types import SimpleNamespace
import unittest

import torch

from stereo_pose.loading import TrajectoryBalancedBatchSampler, category_weighted_loss, make_loader


CATEGORIES = ('pose', 'lighting', 'sensor', 'occlusion', 'ecm', 'stereo', 'combined', 'occlusion', 'ecm')


def dataset_fixture():
    records, windows = [], []
    for group, (category, count) in enumerate(zip(CATEGORIES, (8, 6, 5, 4, 3, 2, 1, 7, 4))):
        start = len(records)
        records.extend({'trajectory_group_id': f'trajectory_{group}', 'category': category,
                        'frame_index': frame} for frame in range(count + 2 - 1))
        windows.extend((start + i, start + i + 1) for i in range(count))
    return SimpleNamespace(split='train', records=records, windows=windows)


class CategoryWeightingTest(unittest.TestCase):
    def test_category_mass_and_sampling_counts(self):
        dataset = dataset_fixture()
        sampler = TrajectoryBalancedBatchSampler(dataset, 4, seed=11, category_balanced=True)
        old = TrajectoryBalancedBatchSampler(dataset, 4, seed=11)
        self.assertEqual(list(sampler), list(old))  # No extra training/render examples.
        report = sampler.coverage_report()
        self.assertEqual(report['logical_batch_size'], 9)
        self.assertEqual(report['microbatches_per_update'], 3)
        self.assertEqual(report['sampled_windows_per_epoch'], 72)
        for weight in report['category_loss_weights'].values():
            self.assertAlmostEqual(weight, 1/7)
        self.assertAlmostEqual(sampler.trajectory_weights['trajectory_3'], 1/14)
        self.assertAlmostEqual(sampler.trajectory_weights['trajectory_0'], 1/7)
        seen = set()
        batches = list(sampler)
        for step in range(sampler.rounds):
            indices = sum(batches[step*3:step*3+3], [])
            groups = [dataset.records[dataset.windows[i][0]]['trajectory_group_id'] for i in indices]
            self.assertEqual(len(set(groups)), 9)
            self.assertAlmostEqual(sum(sampler.trajectory_weights[name] for name in groups), 1)
            seen.update(indices)
        self.assertEqual(seen, set(range(len(dataset.windows))))

    def test_gradient_and_logged_mean_equal_exact_window_objective(self):
        dataset = dataset_fixture()
        sampler = TrajectoryBalancedBatchSampler(dataset, 4, seed=19, category_balanced=True)
        groups = tuple(sampler.groups)
        x = torch.arange(1, 28, dtype=torch.float64).reshape(9, 3)/7
        target = torch.arange(27, dtype=torch.float64).reshape(9, 3)/5
        # Valid point counts differ by window; pooled valid-point reduction is
        # deliberately NOT equivalent to a mean over trajectories.
        valid = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]]*3, dtype=torch.float64)
        weights = torch.tensor([sampler.trajectory_weights[g] for g in groups], dtype=torch.float64)
        full = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        per_window = (((full*x-target).square()*valid).sum(-1)/valid.sum(-1))
        expected_loss = (per_window*weights).sum()
        expected_loss.backward()

        def criterion(pred, truth):
            error = (pred['prediction']-truth['target']).square()
            value = (error*truth['valid']).sum()/truth['valid'].sum()
            return value, {'loss/total': value.detach(), 'probe': value.detach()*2}

        micro = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        accumulated_loss = 0.
        accumulated_probe = 0.
        mass_total = 0.
        for indices in ([3, 0, 8, 2], [5, 1, 4, 6], [7]):
            batch = {'images': torch.zeros(len(indices), 2, 1),
                     'target': target[indices], 'valid': valid[indices],
                     'metadata': [[{'trajectory_group_id': groups[i],
                                    'category': sampler.group_categories[groups[i]]}]*2 for i in indices]}
            loss, metrics, mass = category_weighted_loss(
                criterion, {'prediction': micro*x[indices]}, batch, sampler)
            (loss*mass).backward()
            accumulated_loss += float(loss.detach())*mass
            accumulated_probe += float(metrics['probe'])*mass
            mass_total += mass
        torch.testing.assert_close(micro.grad, full.grad, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(mass_total, 1.)
        self.assertAlmostEqual(accumulated_loss, float(expected_loss.detach()))
        self.assertAlmostEqual(accumulated_probe, float(expected_loss.detach())*2)

    def test_category_validation_and_loader_contract(self):
        dataset = dataset_fixture()
        settings = {'batch_size': 4, 'workers': 0, 'trajectory_balanced': True,
                    'category_balanced': True, 'accumulate': 3}
        self.assertTrue(make_loader(dataset, settings, shuffle=True).batch_sampler.category_balanced)
        dataset.split = 'validation'
        self.assertNotIsInstance(make_loader(dataset, settings, shuffle=False).batch_sampler,
                                 TrajectoryBalancedBatchSampler)
        with self.assertRaisesRegex(ValueError, 'requires trajectory-balanced'):
            make_loader(dataset, {**settings, 'trajectory_balanced': False}, shuffle=False)
        dataset = dataset_fixture()
        dataset.records[0]['category'] = 'other'
        with self.assertRaisesRegex(ValueError, 'one known category'):
            TrajectoryBalancedBatchSampler(dataset, 4, category_balanced=True)
        dataset = dataset_fixture()
        for record in dataset.records:
            record.pop('category')
        # Legacy defaults remain usable with old trajectory-only metadata.
        TrajectoryBalancedBatchSampler(dataset, 4)
        with self.assertRaisesRegex(ValueError, 'one known category'):
            TrajectoryBalancedBatchSampler(dataset, 4, category_balanced=True)


if __name__ == '__main__':
    unittest.main()
