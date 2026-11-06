"""Epoch limits, positive learning-rate floor and independent loss bests.

These helpers contain no model, dataset, optimizer, or GPU operations.
An unlimited epoch policy is separate from the finite learning-rate schedule.
"""
import itertools
import math
import os
from pathlib import Path


def validate_training_policy(settings):
    keep = settings.get('keep_epoch_checkpoints')
    if keep is not None and (isinstance(keep, bool) or not isinstance(keep, int) or keep < 0):
        raise ValueError('keep_epoch_checkpoints must be a nonnegative integer or None')
    if not isinstance(settings.get('save_metric_best_checkpoints', True), bool):
        raise ValueError('save_metric_best_checkpoints must be Boolean')
    for key in ('epochs', 'patience'):
        value = settings[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f'{key} must be a positive integer or None')
    horizon = settings.get('lr_schedule_epochs', settings['epochs'])
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError('A positive lr_schedule_epochs is required when epochs is None')
    floor = settings.get('min_lr_factor', 0.0)
    if not isinstance(floor, (float, int)) or not math.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError('min_lr_factor must be finite and between zero and one')
    if settings['epochs'] is None and floor <= 0:
        raise ValueError('Continuous training requires a positive min_lr_factor')


def training_epochs(start, settings, smoke=False):
    """Return a lazy epoch iterator; smoke always performs exactly one epoch."""
    validate_training_policy(settings)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError('Start epoch must be a nonnegative integer')
    if smoke:
        return iter(range(start, start+1))
    if settings['epochs'] is None:
        return itertools.count(start)
    return iter(range(start, settings['epochs']))


def should_early_stop(stale, patience):
    """None explicitly disables automatic stopping for lack of improvement."""
    return patience is not None and stale >= patience


def learning_rate_factor(epoch, settings):
    """Keep the original cosine schedule, optionally clamped to a positive floor.

    lr_schedule_epochs controls only the schedule horizon, never the loop limit.
    Loading the saved optimizer/scheduler still restores the current LR exactly.
    """
    horizon = settings.get('lr_schedule_epochs', settings['epochs'])
    warmup = settings['warmup_epochs']
    if 'decay_anchor_epoch' in settings:
        anchor = settings['decay_anchor_epoch']
        progress = max(0, min(1, (epoch-anchor)/max(horizon-anchor, 1)))
        factor = settings['decay_anchor_factor'] * .5 * (1 + math.cos(math.pi*progress))
    elif epoch < warmup:
        factor = (epoch + 1) / max(warmup, 1)
    else:
        angle = math.pi * (epoch-warmup) / max(horizon-warmup, 1)
        # Beyond the schedule horizon, a continuous run stays at its floor;
        # it must not accidentally start another cosine cycle.
        if 'lr_schedule_epochs' in settings:
            angle = max(0, min(math.pi, angle))
        factor = .5 * (1 + math.cos(angle))
    return max(settings.get('min_lr_factor', 0.0), factor)


def restore_loss_bests(checkpoint=None):
    """Restore independent historical minima, with legacy checkpoint support."""
    checkpoint = checkpoint or {}
    result = {}
    for key in ('best_training_loss', 'best_validation_loss'):
        value = float(checkpoint.get(key, math.inf))
        if math.isnan(value) or value == -math.inf:
            raise ValueError('Invalid historical loss best: '+key)
        result[key] = value
        epoch = checkpoint.get(key+'_epoch')
        if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1):
            raise ValueError('Loss-best epochs must be one-based integers or None')
        result[key+'_epoch'] = epoch
    return result


def update_loss_bests(previous, training_loss, validation_loss, epoch=None):
    """Return new minima and checkpoint filenames to update, without mutation.

    Strict improvement prevents tied epochs from replacing the retained best.
    Validation may be None only when no full validation loss was evaluated.
    """
    scores = restore_loss_bests(previous)
    if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1):
        raise ValueError('Loss-best epoch must be a one-based positive integer')
    writes = []
    for key, value, filename in (
            ('best_training_loss', training_loss, 'best_train.pt'),
            ('best_validation_loss', validation_loss, 'best_val.pt')):
        if value is None:
            if key == 'best_training_loss':
                raise ValueError('Training epoch loss is required')
            continue
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite epoch loss: '+key)
        if value < scores[key]:
            scores[key] = value
            scores[key+'_epoch'] = epoch
            writes.append(filename)
    return scores, tuple(writes)


def epoch_checkpoint_number(path):
    """Numeric archive ordering remains correct after epoch999."""
    return int(Path(path).stem.removeprefix('epoch_'))


def atomic_checkpoint_link(source, destination):
    """Retain a completed immutable last checkpoint without another serialization.

    Subsequent atomic replacement of last.pt leaves this hardlink unchanged.
    Source and destination are within the same run filesystem.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Checkpoint source and destination must differ')
    temporary = destination.with_name(destination.name+'.tmp')
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def update_epoch_archive(last_checkpoint, epoch, keep):
    """keep=0 disables archive creation; existing historical archives stay intact."""
    if keep == 0:
        return
    if keep is not None and (isinstance(keep, bool) or not isinstance(keep, int) or keep < 0):
        raise ValueError('Archive retention must be nonnegative or None')
    archive = Path(last_checkpoint).parent/'checkpoints'
    archive.mkdir(exist_ok=True)
    atomic_checkpoint_link(last_checkpoint, archive/f'epoch_{epoch:03d}.pt')
    if keep is not None:
        for old in sorted(archive.glob('epoch_*.pt'), key=epoch_checkpoint_number)[:-keep]:
            old.unlink()
