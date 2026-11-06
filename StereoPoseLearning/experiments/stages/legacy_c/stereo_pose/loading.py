"""Trajectory-balanced training windows, CPU prefetch and asynchronous transfer."""
import math

import torch
from torch.utils.data import DataLoader, Sampler
from .data import collate_samples


def worker_init(_):
    torch.set_num_threads(1)


class TrajectoryBalancedBatchSampler(Sampler):
    """One window per trajectory per logical optimizer update, in microbatches.

    All native windows occur at least once in an epoch. Each trajectory gets
    ``rounds=max(native window counts)`` samples: shorter trajectories restart
    a freshly shuffled window cycle. No window is synthesized or concatenated.
    ``set_epoch`` makes resumed epoch order deterministic without depending on
    DataLoader worker scheduling or the global torch RNG.
    """

    def __init__(self, dataset, microbatch_size: int, seed: int = 0, category_balanced: bool = False):
        if dataset.split != 'train':
            raise ValueError('Trajectory-balanced sampling is training-only')
        if not isinstance(microbatch_size, int) or microbatch_size < 1:
            raise ValueError('microbatch_size must be a positive integer')
        groups = {}
        group_categories = {}
        self.category_balanced = bool(category_balanced)
        for index, window in enumerate(dataset.windows):
            if not window:
                raise ValueError('Cannot balance an empty temporal window')
            names = {dataset.records[i]['trajectory_group_id'] for i in window}
            if len(names) != 1:
                raise ValueError('A temporal window crosses trajectory groups')
            name = next(iter(names))
            groups.setdefault(name, []).append(index)
            if self.category_balanced:
                categories = {dataset.records[i].get('category') for i in window}
                if len(categories) != 1 or None in categories or 'unknown' in categories:
                    raise ValueError('Category-balanced windows require one known category')
                category = next(iter(categories))
                if name in group_categories and group_categories[name] != category:
                    raise ValueError('A trajectory group contains multiple categories')
                group_categories[name] = category
        if not groups:
            raise ValueError('Cannot balance a dataset with no training windows')
        self.groups = {name: tuple(groups[name]) for name in sorted(groups)}
        self.group_categories = group_categories
        category_counts = {}
        for category in group_categories.values():
            category_counts[category] = category_counts.get(category, 0) + 1
        self.category_counts = category_counts
        self.trajectory_weights = {
            name: (1.0 / (len(category_counts) * category_counts[group_categories[name]])
                   if self.category_balanced else 1.0 / len(self.groups))
            for name in self.groups
        }
        self.microbatch_size = microbatch_size
        self.seed = int(seed)
        self.epoch = 0
        self.logical_batch_size = len(self.groups)
        self.microbatches_per_update = math.ceil(self.logical_batch_size / self.microbatch_size)
        self.rounds = max(map(len, self.groups.values()))
        self.logical_updates_per_epoch = self.rounds

    def set_epoch(self, epoch: int):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError('Sampler epoch must be a nonnegative integer')
        self.epoch = epoch

    def __len__(self):
        return self.rounds * self.microbatches_per_update

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        names = tuple(self.groups)
        # Build each independent window stream through shuffled cycles. The
        # first complete cycle guarantees coverage of every native window.
        streams = {}
        for name, indices in self.groups.items():
            stream = []
            while len(stream) < self.rounds:
                order = torch.randperm(len(indices), generator=generator).tolist()
                stream.extend(indices[i] for i in order)
            streams[name] = stream[:self.rounds]
        for step in range(self.rounds):
            group_order = torch.randperm(len(names), generator=generator).tolist()
            logical_batch = [streams[names[i]][step] for i in group_order]
            for start in range(0, len(logical_batch), self.microbatch_size):
                yield logical_batch[start:start + self.microbatch_size]

    def coverage_report(self):
        """Planning-only counts; this does not read or claim to train images."""
        return {
            'strategy': 'one_window_per_trajectory_per_optimizer_update',
            'epoch_zero_based': self.epoch,
            'epoch_seed': self.seed + self.epoch,
            'trajectory_groups': len(self.groups),
            'native_windows_total': sum(map(len, self.groups.values())),
            'logical_batch_size': self.logical_batch_size,
            'microbatch_size': self.microbatch_size,
            'microbatches_per_update': self.microbatches_per_update,
            'logical_updates_per_epoch': self.rounds,
            'microbatches_per_epoch': len(self),
            'sampled_windows_per_epoch': self.rounds * self.logical_batch_size,
            'all_native_windows_covered': True,
            'per_trajectory': {
                name: {'native_windows': len(indices),
                       'unique_windows_planned': len(indices),
                       'sampled_windows_planned': self.rounds,
                       'repeated_windows_planned': self.rounds - len(indices),
                       'category': self.group_categories.get(name),
                       'optimizer_loss_weight': self.trajectory_weights[name]}
                for name, indices in self.groups.items()
            },
            'category_balanced': self.category_balanced,
            'category_loss_weights': {
                category: sum(self.trajectory_weights[name] for name in self.groups
                              if self.group_categories.get(name) == category)
                for category in sorted(self.category_counts)
            },
            'loss_normalization': (
                'per-window loss × 1/(category_count × trajectories_in_category); '
                'sum all trajectory contributions, then optimizer step'
                if self.category_balanced else
                'microbatch_sample_count / logical_batch_size; accumulate microbatches then step'),
        }


def make_loader(dataset, settings, shuffle=False):
    workers = int(settings.get('workers', 0))
    kwargs = dict(num_workers=workers, collate_fn=collate_samples,
                  pin_memory=settings.get('pin_memory', False))
    if settings.get('category_balanced', False) and not settings.get('trajectory_balanced', False):
        raise ValueError('Category-balanced loss requires trajectory-balanced sampling')
    balanced = settings.get('trajectory_balanced', False) and shuffle and dataset.split == 'train'
    if balanced:
        sampler = TrajectoryBalancedBatchSampler(dataset, settings['batch_size'], settings.get('seed', 0),
                                                 settings.get('category_balanced', False))
        if settings.get('accumulate', 1) != sampler.microbatches_per_update:
            raise ValueError('Trajectory-balanced accumulate must equal microbatches_per_update '
                             f'({sampler.microbatches_per_update}); received {settings.get("accumulate", 1)}')
        kwargs['batch_sampler'] = sampler
    else:
        kwargs.update(batch_size=settings['batch_size'], shuffle=shuffle)
    if workers:
        kwargs.update(prefetch_factor=settings.get('prefetch_factor', 2),
                      persistent_workers=settings.get('persistent_workers', True),
                      multiprocessing_context='spawn', worker_init_fn=worker_init)
    return DataLoader(dataset, **kwargs)


def category_weighted_loss(criterion, predictions, batch, sampler):
    """Return the exact weighted window loss, metrics and microbatch mass.

    The expensive visual forward is still batched. Each window's small loss/FK
    calculation is reduced separately because visible/in-frame target counts
    can differ between windows. Pooling those targets first would overweight
    windows with more valid points, despite equal category coefficients.
    Metadata stays on CPU; tensor slicing is a view and does not read the GPU.
    The returned mean must be multiplied by ``mass`` before backward. Over one
    logical update the masses sum to one, with every category contributing
    1 / number_of_categories and each of its independent trajectories equal.
    """
    if not sampler.category_balanced:
        raise ValueError('Category-weighted loss requires a category-balanced sampler')
    metadata = batch['metadata']
    if len(metadata) != batch['images'].shape[0]:
        raise ValueError('Metadata batch size mismatch')
    weights = []
    seen = set()
    for window in metadata:
        names = {frame['trajectory_group_id'] for frame in window}
        categories = {frame['category'] for frame in window}
        if len(names) != 1 or len(categories) != 1:
            raise ValueError('Training window crosses trajectory/category boundaries')
        name = next(iter(names))
        if name not in sampler.trajectory_weights or next(iter(categories)) != sampler.group_categories[name]:
            raise ValueError('Training window metadata disagrees with sampler')
        if name in seen:
            raise ValueError('A trajectory occurs twice in one logical microbatch')
        seen.add(name)
        weights.append(sampler.trajectory_weights[name])
    if not weights:
        raise ValueError('Cannot weight an empty batch')
    mass = sum(weights)
    loss_sum = None
    metric_sums = {}
    for i, weight in enumerate(weights):
        # All prediction and target tensors follow [B,T,...]; RGB is unused by
        # the criterion and excluded to avoid holding unnecessary subviews.
        prediction = {name: value[i:i+1] for name, value in predictions.items()}
        target = {name: value[i:i+1] for name, value in batch.items()
                  if isinstance(value, torch.Tensor) and name not in ('images', 'hires_images')}
        loss, metrics = criterion(prediction, target)
        term = loss * weight
        loss_sum = term if loss_sum is None else loss_sum + term
        for name, value in metrics.items():
            metric_sums[name] = metric_sums.get(name, 0) + value.detach() * weight
    return loss_sum / mass, {name: value / mass for name, value in metric_sums.items()}, mass


def device_batches(loader, device, asynchronous=False):
    def transfer(batch):
        return {k: v.to(device, non_blocking=asynchronous) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
    if device.type != 'cuda' or not asynchronous:
        for batch in loader:
            yield transfer(batch)
        return
    stream = torch.cuda.Stream(device=device)
    iterator = iter(loader)
    def preload():
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            return None
        with torch.cuda.stream(stream):
            return transfer(cpu_batch)
    pending = preload()
    while pending is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        batch = pending
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(torch.cuda.current_stream(device))
        pending = preload()
        yield batch
