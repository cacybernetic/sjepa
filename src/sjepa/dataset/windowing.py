"""Split each clip into overlapping windows that cover all of its frames.

A long recording holds far more speech than a single training window. Rather
than keep only one crop per file (and throw the rest away each epoch), we slide
a window of `window` units across every clip with a fixed `hop`, so every frame
is seen and neighbouring windows overlap.

The helpers are unit-agnostic: pass seconds for the on-the-fly reader, or raw
sample counts for the HDF5 reader. `plan_windows` turns a list of clip lengths
into a flat list of `(clip_index, start)` pairs, one entry per window.
"""

from __future__ import annotations

# Largest overlap we allow, so the hop stays strictly positive (an overlap of
# 1.0 would place infinitely many windows at the same spot).
_MAX_OVERLAP = 0.95


def hop_from_overlap(window, overlap):
    """Turn an overlap fraction into a hop in the same units as `window`.

    Args:
        window: the window length (seconds or samples).
        overlap: the fraction of a window shared with its neighbour, in [0, 1).

    Returns:
        The hop (stride) between two consecutive window starts.
    """
    overlap = min(max(float(overlap), 0.0), _MAX_OVERLAP)
    return window * (1.0 - overlap)


def window_starts(length, window, hop):
    """Return the start offsets of the windows that tile one clip.

    A clip shorter than one window yields a single window at 0 (the collator
    pads it). Otherwise windows are spaced by `hop`, and the last one is snapped
    to the clip end so the tail is always fully covered by a full-length window.
    """
    if length <= window or hop <= 0:
        return [0.0]
    last = length - window
    starts = []
    offset = 0.0
    while offset < last - 1e-6:
        starts.append(offset)
        offset += hop
    starts.append(last)
    return starts


def plan_windows(lengths, window, hop):
    """Expand per-clip lengths into a flat window plan.

    Args:
        lengths: an iterable of clip lengths (seconds or samples).
        window: the window length in the same units.
        hop: the stride between consecutive windows in the same units.

    Returns:
        A list of `(clip_index, start)` pairs, one per window.
    """
    plan = []
    for index, length in enumerate(lengths):
        for start in window_starts(length, window, hop):
            plan.append((index, start))
    return plan
