"""Exact interval arithmetic — the numerical backbone of every gate metric.

Every question this project asks is an interval question: *how much of this
annotation box landed inside a recorded region?* The obvious implementation —
rasterise both onto a grid of 10 ms bins and count — is wrong in a way that
matters here, because our shortest annotation boxes are ~0.1 s and the grid
quantisation error is then a double-digit percentage of the thing being measured.

So we do it exactly, on sorted disjoint intervals, in closed form. `covered_upto`
is the trick: precompute the cumulative length of the region list, and the total
covered time before any instant `t` is a binary search plus one clamp. Overlap of
[a, b] is then `covered_upto(b) - covered_upto(a)`, with no grid anywhere.

This also makes the metrics resolution-independent: changing the gate's decision
window from 1 s to 0.5 s changes the regions, not the measurement of them.
"""
from __future__ import annotations

import numpy as np


def merge(intervals: np.ndarray) -> np.ndarray:
    """Sort and union a set of possibly-overlapping [start, end) intervals.

    Needed because annotation boxes overlap constantly — birds sing over each
    other. Summing per-box durations double-counts that overlap, so any
    "how many bird-seconds are there" question must be asked of the *union*.
    """
    iv = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    iv = iv[iv[:, 1] > iv[:, 0]]
    if len(iv) == 0:
        return np.zeros((0, 2))
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    out = [iv[0].copy()]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append(np.array([s, e]))
    return np.array(out)


def total(intervals: np.ndarray) -> float:
    """Total length. Assumes already merged (call `merge` first if unsure)."""
    iv = np.asarray(intervals).reshape(-1, 2)
    return float((iv[:, 1] - iv[:, 0]).sum()) if len(iv) else 0.0


def covered_upto(t: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Total covered time in (-inf, t], vectorised over `t`.

    `regions` must be sorted and disjoint (i.e. the output of `merge`).
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if len(regions) == 0:
        return np.zeros_like(t)
    rs, re = regions[:, 0], regions[:, 1]
    cum = np.concatenate([[0.0], np.cumsum(re - rs)])
    # j = index of the last region that starts at or before t
    j = np.searchsorted(rs, t, side="right") - 1
    inside = np.where(j >= 0, np.minimum(t, re[np.maximum(j, 0)]) - rs[np.maximum(j, 0)], 0.0)
    return cum[np.maximum(j, 0)] * (j >= 0) + np.maximum(inside, 0.0)


def overlap(starts, ends, regions: np.ndarray) -> np.ndarray:
    """Seconds of each [start, end) box that fall inside `regions`.

    `regions` must be merged. Returns an array aligned with `starts`/`ends`.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if len(regions) == 0:
        return np.zeros_like(starts)
    return np.maximum(covered_upto(ends, regions) - covered_upto(starts, regions), 0.0)


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of two merged interval sets. Used for 'bird-seconds kept'."""
    a, b = np.asarray(a).reshape(-1, 2), np.asarray(b).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 2))
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            out.append([lo, hi])
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.array(out) if out else np.zeros((0, 2))
