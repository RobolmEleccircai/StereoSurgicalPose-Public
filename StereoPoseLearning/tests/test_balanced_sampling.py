"""Logical updates cover all trajectories without joining unrelated windows."""
from collections import Counter
from types import SimpleNamespace
import unittest

import torch

from stereo_pose.loading import TrajectoryBalancedBatchSampler, make_loader


def make_dataset(counts=(8, 6, 5, 4, 3, 2, 1), window=8):
    records, windows = [], []
    for group, count in enumerate(counts):
        start = len(records)
        records.extend({'trajectory_group_id': f'trajectory_{group}', 'frame_index': frame}
                       for frame in range(count + window - 1))
        windows.extend(tuple(range(start + i, start + i + window)) for i in range(count))
    return SimpleNamespace(split='train', records=records, windows=windows)


def window_group(dataset, index):
    groups = {dataset.records[i]['trajectory_group_id'] for i in dataset.windows[index]}
    assert len(groups) == 1
    return next(iter(groups))


class BalancedSamplingTest(unittest.TestCase):
    def test_logical_updates_each_group_once_every_native_window_covered(self):
        dataset = make_dataset()
        sampler = TrajectoryBalancedBatchSampler(dataset, 4, seed=18)
        batches = list(sampler)
        self.assertEqual(sampler.logical_batch_size, 7)
        self.assertEqual(sampler.microbatches_per_update, 2)
        self.assertEqual(sampler.logical_updates_per_epoch, 8)
        self.assertEqual(len(batches), 16)
        seen = Counter()
        for update in range(sampler.rounds):
            first, second = batches[2*update:2*update + 2]
            self.assertEqual((len(first), len(second)), (4, 3))
            names = [window_group(dataset, i) for i in first + second]
            self.assertEqual(len(set(names)), 7)
            seen.update(first + second)
            for index in first + second:
                frames = [dataset.records[i]['frame_index'] for i in dataset.windows[index]]
                self.assertEqual(frames, list(range(frames[0], frames[0] + 8)))
        self.assertEqual(set(seen), set(range(len(dataset.windows))))
        counts = Counter(window_group(dataset, index) for batch in batches for index in batch)
        self.assertEqual(set(counts.values()), {8})
        report = sampler.coverage_report()
        self.assertEqual(report['native_windows_total'], 29)
        self.assertEqual(report['sampled_windows_per_epoch'], 56)
        self.assertEqual(report['per_trajectory']['trajectory_6']['repeated_windows_planned'], 7)

    def test_epoch_reproducible_resume_and_changes_order(self):
        dataset = make_dataset()
        a = TrajectoryBalancedBatchSampler(dataset, 4, seed=20261110)
        b = TrajectoryBalancedBatchSampler(dataset, 4, seed=20261110)
        epoch0 = list(a)
        a.set_epoch(12); b.set_epoch(12)
        self.assertEqual(list(a), list(b))
        self.assertEqual(list(a), list(a))
        self.assertNotEqual(epoch0, list(a))
        self.assertEqual(a.coverage_report()['epoch_seed'], 20261122)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            a.set_epoch(-1)

    def test_no_cross_trajectory_windows_and_accumulation_contract(self):
        dataset = make_dataset()
        loader = make_loader(dataset, {'batch_size': 4, 'workers': 0, 'trajectory_balanced': True,
                                       'accumulate': 2, 'seed': 9}, shuffle=True)
        self.assertIsInstance(loader.batch_sampler, TrajectoryBalancedBatchSampler)
        self.assertEqual(len(loader), 16)
        with self.assertRaisesRegex(ValueError, 'accumulate'):
            make_loader(dataset, {'batch_size': 4, 'trajectory_balanced': True, 'accumulate': 1}, shuffle=True)
        bad = make_dataset()
        bad.windows[0] = (0, len(bad.records) - 1)
        with self.assertRaisesRegex(ValueError, 'crosses trajectory'):
            TrajectoryBalancedBatchSampler(bad, 4)
        dataset.split = 'validation'
        with self.assertRaisesRegex(ValueError, 'training-only'):
            TrajectoryBalancedBatchSampler(dataset, 4)
        # Validation stays sequential, despite shared training configuration.
        validation = make_loader(dataset, {'batch_size': 4, 'trajectory_balanced': True, 'accumulate': 2}, shuffle=False)
        self.assertNotIsInstance(validation.batch_sampler, TrajectoryBalancedBatchSampler)

    def test_four_plus_three_gradient_matches_one_seven_sample_update(self):
        x = torch.tensor([1., 2., 4., 5., 7., 11., 17.])
        truth = torch.tensor([1.5, 3., 3., 8., 12., 9., 10.])
        full = torch.tensor(0.3, requires_grad=True)
        ((full*x - truth).square().mean()).backward()
        micro = torch.tensor(0.3, requires_grad=True)
        for indices in (slice(0, 4), slice(4, 7)):
            losses = (micro*x[indices] - truth[indices]).square()
            (losses.mean() * losses.numel()/7).backward()
        torch.testing.assert_close(micro.grad, full.grad)


if __name__ == '__main__':
    unittest.main()
