"""CPU prefetch and pinned, asynchronous CUDA transfer; sample order is unchanged."""
import torch
from torch.utils.data import DataLoader
from .data import collate_samples


def worker_init(_):
    torch.set_num_threads(1)


def make_loader(dataset, settings, shuffle=False):
    workers = int(settings.get('workers', 0))
    kwargs = dict(batch_size=settings['batch_size'], shuffle=shuffle,
                  num_workers=workers, collate_fn=collate_samples,
                  pin_memory=settings.get('pin_memory', False))
    if workers:
        kwargs.update(prefetch_factor=settings.get('prefetch_factor', 2),
                      persistent_workers=settings.get('persistent_workers', True),
                      multiprocessing_context='spawn', worker_init_fn=worker_init)
    return DataLoader(dataset, **kwargs)


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
