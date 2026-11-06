"""CPU policy checks; no images, network downloads, or training jobs."""
import copy
import itertools
import math
from pathlib import Path
import tempfile
import unittest

from stereo_pose.training_policy import (validate_training_policy, training_epochs,
    should_early_stop, learning_rate_factor, restore_loss_bests, update_loss_bests,
    epoch_checkpoint_number, atomic_checkpoint_link, update_epoch_archive)


def finite():
    return dict(epochs=100, patience=25, warmup_epochs=3)


def continuous():
    return dict(finite(), epochs=None, patience=None,
                lr_schedule_epochs=100, min_lr_factor=.01)


class ContinuousPolicyTest(unittest.TestCase):
    def test_lazy_unlimited_epochs_and_no_patience_stop(self):
        settings = continuous()
        epochs = training_epochs(99, settings)
        self.assertIsInstance(epochs, itertools.count)
        values = list(itertools.islice(epochs, 250))
        self.assertEqual(values[0], 99)
        self.assertEqual(values[-1], 348)
        self.assertFalse(should_early_stop(200, None))
        self.assertFalse(should_early_stop(1000000, None))
        self.assertEqual(list(training_epochs(120, settings, smoke=True)), [120])

    def test_existing_finite_loop_and_early_stop_unchanged(self):
        self.assertEqual(list(training_epochs(98, finite())), [98, 99])
        self.assertEqual(list(training_epochs(100, finite())), [])
        self.assertFalse(should_early_stop(24, 25))
        self.assertTrue(should_early_stop(25, 25))
        self.assertEqual(list(training_epochs(100, finite(), smoke=True)), [100])

    def test_current_cosine_factors_identical_and_future_floor_positive(self):
        old, new = finite(), continuous()
        for epoch in range(0, 29):
            original = ((epoch+1)/3 if epoch < 3 else
                        .5*(1+math.cos(math.pi*(epoch-3)/97)))
            self.assertEqual(learning_rate_factor(epoch, old), original)
            self.assertEqual(learning_rate_factor(epoch, new), original)
        for epoch in (100, 101, 150, 1000, 1000000):
            self.assertEqual(learning_rate_factor(epoch, new), .01)
        anchored_old = dict(old, decay_anchor_epoch=25, decay_anchor_factor=.8)
        anchored_new = dict(new, decay_anchor_epoch=25, decay_anchor_factor=.8)
        self.assertEqual(learning_rate_factor(28, anchored_new), learning_rate_factor(28, anchored_old))
        self.assertEqual(learning_rate_factor(1000, anchored_new), .01)

    def test_continuous_policy_requires_horizon_and_positive_floor(self):
        for changes in ({'lr_schedule_epochs': None}, {'min_lr_factor': 0},
                        {'min_lr_factor': float('nan')}, {'patience': 0}, {'epochs': True}):
            settings = dict(continuous(), **changes)
            with self.assertRaises(ValueError):
                validate_training_policy(settings)

    def test_independent_best_losses_ties_resume_and_metadata(self):
        original = {'best_training_loss': .004, 'best_training_loss_epoch': 28,
                    'best_validation_loss': .005, 'best_validation_loss_epoch': 27}
        before = copy.deepcopy(original)
        state, writes = update_loss_bests(original, .003, .006, epoch=29)
        self.assertEqual(writes, ('best_train.pt',))
        self.assertEqual(state['best_training_loss_epoch'], 29)
        self.assertEqual(state['best_validation_loss_epoch'], 27)
        self.assertEqual(state['best_validation_loss'], .005)
        self.assertEqual(original, before)
        checkpoint = dict(model='current29', **state)
        restored = restore_loss_bests(checkpoint)
        self.assertEqual(restored, state)
        state, writes = update_loss_bests(restored, .004, .002, epoch=30)
        self.assertEqual(writes, ('best_val.pt',))
        self.assertEqual(state['best_training_loss'], .003)
        self.assertEqual(state['best_training_loss_epoch'], 29)
        self.assertEqual(state['best_validation_loss_epoch'], 30)
        tied, writes = update_loss_bests(state, .003, .002, epoch=31)
        self.assertEqual(tied, state)
        self.assertEqual(writes, ())
        both, writes = update_loss_bests(state, .001, .001, epoch=32)
        self.assertEqual(writes, ('best_train.pt', 'best_val.pt'))
        self.assertEqual(both['best_training_loss_epoch'], 32)
        self.assertEqual(both['best_validation_loss_epoch'], 32)

    def test_legacy_best_scores_and_smoke_without_validation(self):
        old = restore_loss_bests({'best': .3, 'best_tip': .4})
        self.assertEqual(old['best_training_loss'], math.inf)
        self.assertEqual(old['best_validation_loss'], math.inf)
        state, writes = update_loss_bests(old, .1, None, epoch=1)
        self.assertEqual(writes, ('best_train.pt',))
        self.assertEqual(state['best_validation_loss'], math.inf)
        with self.assertRaises(FloatingPointError):
            update_loss_bests(state, .1, float('nan'), epoch=2)

    def test_numeric_archive_retention_preserves_latest_and_root_bests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'checkpoints'; archive.mkdir()
            for number in (998, 999, 1000, 1001):
                (archive/f'epoch_{number:03d}.pt').write_text(str(number))
            for name in ('best.pt', 'best_tip.pt', 'best_train.pt', 'best_val.pt'):
                (root/name).write_text('retained')
            for old in sorted(archive.glob('epoch_*.pt'), key=epoch_checkpoint_number)[:-2]:
                old.unlink()
            self.assertEqual(sorted(int(p.read_text()) for p in archive.iterdir()), [1000, 1001])
            self.assertTrue(all((root/name).read_text() == 'retained'
                                for name in ('best.pt', 'best_tip.pt', 'best_train.pt', 'best_val.pt')))

    def test_zero_retention_does_not_create_or_delete_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); last = root/'last.pt'; last.write_text('epoch28')
            update_epoch_archive(last, 28, 0)
            self.assertFalse((root/'checkpoints').exists())
            (root/'checkpoints').mkdir()
            old = root/'checkpoints/epoch_027.pt'; old.write_text('historic')
            update_epoch_archive(last, 28, 0)
            self.assertEqual(list((root/'checkpoints').iterdir()), [old])
            self.assertEqual(old.read_text(), 'historic')

    def test_best_hardlinks_share_current_inode_but_keep_old_weights_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); last = root/'last.pt'; last.write_text('epoch28')
            train, val = root/'best_train.pt', root/'best_val.pt'
            atomic_checkpoint_link(last, train); atomic_checkpoint_link(last, val)
            self.assertEqual({p.stat().st_ino for p in (last, train, val)}, {last.stat().st_ino})
            checkpoint = root/'checkpoint.tmp'; checkpoint.write_text('epoch29')
            checkpoint.replace(last)
            self.assertEqual(last.read_text(), 'epoch29')
            self.assertEqual(train.read_text(), 'epoch28'); self.assertEqual(val.read_text(), 'epoch28')
            atomic_checkpoint_link(last, train)
            self.assertEqual(train.read_text(), 'epoch29'); self.assertEqual(val.read_text(), 'epoch28')
            self.assertEqual(train.stat().st_ino, last.stat().st_ino)
            self.assertNotEqual(val.stat().st_ino, last.stat().st_ino)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['best_train.pt','best_val.pt','last.pt'])

    def test_actual_archive_pruning_sorts_numerically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); last = root/'last.pt'
            for epoch in (998, 999, 1000, 1001):
                temp = root/'checkpoint.tmp'; temp.write_text(str(epoch)); temp.replace(last)
                update_epoch_archive(last, epoch, 2)
            self.assertEqual(sorted(p.read_text() for p in (root/'checkpoints').iterdir()), ['1000','1001'])

    def test_cpu_scheduler_resume_keeps_optimizer_moments_and_current_lr(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch CPU test runs in the training environment')
        settings = finite()
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([parameter], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda epoch: learning_rate_factor(epoch, settings))
        for _ in range(28):
            optimizer.zero_grad(); (parameter**2).sum().backward()
            optimizer.step(); scheduler.step()
        opt_state, sched_state = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
        resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
        resumed_opt = torch.optim.AdamW([resumed_parameter], lr=.01)
        resumed_sched = torch.optim.lr_scheduler.LambdaLR(resumed_opt,
            lambda epoch: learning_rate_factor(epoch, continuous()))
        resumed_opt.load_state_dict(opt_state); resumed_sched.load_state_dict(sched_state)
        self.assertEqual(resumed_opt.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
        self.assertTrue(torch.equal(resumed_opt.state[resumed_parameter]['exp_avg'], optimizer.state[parameter]['exp_avg']))
        self.assertEqual(resumed_sched.last_epoch, scheduler.last_epoch)
        for opt, sched, p in ((optimizer, scheduler, parameter), (resumed_opt, resumed_sched, resumed_parameter)):
            opt.zero_grad(); (p**2).sum().backward(); opt.step(); sched.step()
        self.assertTrue(torch.equal(resumed_parameter, parameter))
        self.assertEqual(resumed_opt.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    unittest.main()
