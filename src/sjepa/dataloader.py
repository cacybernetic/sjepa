"""A resumable DataLoader adapter for in-epoch checkpointing.

A plain `torch.utils.data.DataLoader` with `shuffle=True` reshuffles the index
list at the start of every epoch (its internal `RandomSampler`) and follows that
order to form the batches. The order is not exposed, so a run that crashes in the
middle of an epoch cannot resume at the right batch: it must restart the epoch,
and the shuffle is different, breaking the "every sample once per epoch"
guarantee.

`ResumableDataLoader` wraps a dataset and reproduces that behaviour while
exposing the position through the PyTorch `state_dict()` / `load_state_dict()`
contract. Two facts make a tiny state enough to resume exactly:

  * the epoch order is **deterministic** from `(seed, epoch)` (a seeded
    `torch.Generator`), so we never store the index list, only the epoch number;
  * we only store how many batches were already consumed in the epoch.

On resume `__iter__` rebuilds the same order and starts an inner `DataLoader`
on the **remaining** batches only, so the already-seen batches are never loaded
again (no wasted audio decoding). This is the fault-tolerant / in-epoch
checkpointing used by frameworks such as PyTorch Lightning.

This is a plain adapter, not an `nn.Module`: it holds no learnable parameter and
only needs the serialization contract, which it implements directly.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


class _FixedBatchSampler:
    """Yield a fixed, precomputed list of index batches.

    Passed as the `batch_sampler` of an inner `DataLoader` so the worker
    processes load exactly the batches we kept (the ones not yet consumed).
    """

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class ResumableDataLoader:
    """A DataLoader wrapper whose iteration position can be saved and restored."""

    def __init__(self, dataset, batch_size, *, shuffle, seed, collate_fn=None,
                 num_workers=0, drop_last=False, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self._epoch = 0
        self._batches_done = 0

    # ----- epoch control -----

    def set_epoch(self, epoch):
        """Start a fresh epoch: change the order and reset the position."""
        self._epoch = epoch
        self._batches_done = 0

    def _order(self):
        """Return the index order for the current epoch (deterministic)."""
        total = len(self.dataset)
        if not self.shuffle:
            return list(range(total))
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        return torch.randperm(total, generator=generator).tolist()

    def _batched(self, order):
        """Split an index order into batches, honouring `drop_last`."""
        size = self.batch_size
        batches = [order[i:i + size] for i in range(0, len(order), size)]
        if self.drop_last and batches and len(batches[-1]) < size:
            batches.pop()
        return batches

    # ----- iteration -----

    def __iter__(self):
        """Iterate the remaining batches of the current epoch."""
        batches = self._batched(self._order())
        remaining = batches[self._batches_done:]
        loader = DataLoader(
            self.dataset, batch_sampler=_FixedBatchSampler(remaining),
            collate_fn=self.collate_fn, num_workers=self.num_workers,
            pin_memory=self.pin_memory)
        for batch in loader:
            self._batches_done += 1
            yield batch

    def __len__(self):
        """Return the total number of batches in one full epoch."""
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    # ----- serialization (in-epoch checkpoint) -----

    def state_dict(self):
        """Return the position so an interrupted epoch can resume exactly."""
        return {"epoch": self._epoch, "batches_done": self._batches_done,
                "seed": self.seed}

    def load_state_dict(self, state):
        """Restore the epoch and the number of batches already consumed."""
        self._epoch = int(state["epoch"])
        self._batches_done = int(state["batches_done"])
        self.seed = int(state.get("seed", self.seed))

    @property
    def batches_done(self):
        """How many batches of the current epoch were already consumed."""
        return self._batches_done
