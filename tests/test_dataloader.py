"""Tests for the resumable DataLoader adapter (in-epoch checkpointing)."""

import torch
from torch.utils.data import Dataset

from sjepa.dataloader import ResumableDataLoader


class _RangeDataset(Dataset):
    """A dataset whose item is just its index, to track the visit order."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return index


def _collate(batch):
    """Collate a list of indices into a tensor."""
    return torch.tensor(batch)


def _make(n=23, batch_size=4, shuffle=True, seed=0, drop_last=False):
    return ResumableDataLoader(
        _RangeDataset(n), batch_size=batch_size, shuffle=shuffle, seed=seed,
        collate_fn=_collate, num_workers=0, drop_last=drop_last)


def _all_indices(loader):
    """Flatten one full epoch into the list of visited indices."""
    visited = []
    for batch in loader:
        visited.extend(batch.tolist())
    return visited


def test_order_is_deterministic_per_epoch():
    """The same (seed, epoch) gives the same order across loaders."""
    a, b = _make(), _make()
    a.set_epoch(3)
    b.set_epoch(3)
    assert _all_indices(a) == _all_indices(b)


def test_epoch_changes_order_and_resets_position():
    """A new epoch reshuffles and starts at batch 0."""
    loader = _make()
    loader.set_epoch(0)
    order0 = _all_indices(loader)
    loader.set_epoch(1)
    assert loader.batches_done == 0
    order1 = _all_indices(loader)
    assert order0 != order1
    # Every index is still visited exactly once.
    assert sorted(order0) == sorted(order1) == list(range(23))


def test_no_shuffle_is_sequential():
    """Without shuffle the order is the natural range, every epoch."""
    loader = _make(shuffle=False)
    loader.set_epoch(0)
    assert _all_indices(loader) == list(range(23))
    loader.set_epoch(7)
    assert _all_indices(loader) == list(range(23))


def test_resume_skips_consumed_batches_without_loss():
    """State round-trip resumes exactly: union of seen indices = full epoch."""
    loader = _make()
    loader.set_epoch(2)
    seen = []
    for stop_after, batch in enumerate(loader):
        seen.extend(batch.tolist())
        if stop_after == 2:  # crash after 3 batches
            break
    state = loader.state_dict()
    assert state["batches_done"] == 3

    resumed = _make()
    resumed.load_state_dict(state)
    rest = _all_indices(resumed)

    # No batch is replayed and none is missing.
    assert set(seen).isdisjoint(rest)
    assert sorted(seen + rest) == list(range(23))


def test_drop_last_drops_the_partial_batch():
    """drop_last removes the final short batch from the epoch."""
    loader = _make(n=23, batch_size=4, drop_last=True)
    loader.set_epoch(0)
    assert len(loader) == 5            # 23 // 4
    assert len(_all_indices(loader)) == 20


def test_len_matches_iteration():
    """__len__ equals the number of batches actually yielded."""
    loader = _make(n=23, batch_size=4, drop_last=False)
    loader.set_epoch(0)
    count = sum(1 for _ in loader)
    assert count == len(loader) == 6   # ceil(23 / 4)
