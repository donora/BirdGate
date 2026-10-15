"""1-s windows with **binary** presence labels, and the oracle-gate ceiling.

Differs from `birdsong/windows.py` (the sibling project's 5-s multi-label
windower) in three ways that matter:

- **Binary, not multi-hot.** `y[i] = 1` iff *any* species box overlaps window i.
  The gate does not care which bird. Species identity is kept in an optional
  side-matrix purely so `metrics.py` can report *per-species* retention, which is
  the failure mode that actually threatens a deployment.
- **1 s, not 5 s.** The AudioMoth ring buffer hands us one second at a time, so
  1 s is the natural decision unit and costs nothing to adopt.
- **A lower overlap floor** (0.15 s vs 0.5 s). A gate should open on a glimpse; a
  classifier should not train on one. Being trigger-happy is the correct bias for
  a recall-first detector.

`oracle_report()` runs before any model exists. It feeds the ground truth through
the same state machine a model would drive, which fixes the denominator for every
later claim — and, more usefully, it separates *state-machine* wins from *model*
wins for the rest of the project. If a model change and a hangover change both
move retention by 2%, only this table tells you which was which.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import data as D
from . import intervals as I
from .config import WindowParams


def build_windows(split_df=None, ann=None, p: WindowParams = WindowParams(),
                  return_species: bool = False):
    """Cut every recording into windows and label each 1 / 0 for any-bird-present.

    Returns `(windows, y)`, or `(windows, y, S)` with `return_species=True` where
    `S` is uint8 (n_windows, 56) indicating which species are in each window.

    `windows` columns: filename, split, win_idx, start_s, end_s, n_species.
    """
    if split_df is None:
        split_df = D.make_split()
    if ann is None:
        ann = D.load_annotations()

    codes = D.species_codes()
    col = {c: j for j, c in enumerate(codes)}
    by_file = {f: g for f, g in ann.groupby("filename")}

    frames, ys, Ss = [], [], []
    for rec in split_df.sort_values("filename").itertuples():
        hop = p.hop_for(rec.split)
        n_win = int(math.floor((rec.duration_s - p.win_s) / hop)) + 1
        if n_win <= 0:
            continue
        starts = np.arange(n_win) * hop
        S = np.zeros((n_win, len(codes)), dtype=np.uint8)

        for a in by_file.get(rec.filename, pd.DataFrame()).itertuples():
            # A box shorter than the floor must land entirely inside a window to
            # count — same `min` trick as the sibling project, for the same reason:
            # a fixed floor would make sub-floor boxes unlabellable.
            threshold = min(p.min_overlap_s, a.duration_s)
            k_lo = max(0, math.ceil((a.start_s - p.win_s) / hop))
            k_hi = min(n_win - 1, math.ceil(a.end_s / hop) - 1)
            if k_hi < k_lo:
                continue
            k = np.arange(k_lo, k_hi + 1)
            w0 = k * hop
            ov = np.minimum(a.end_s, w0 + p.win_s) - np.maximum(a.start_s, w0)
            hit = (ov > 0) & (ov >= threshold - 1e-9)
            S[k[hit], col[a.code]] = 1

        frames.append(pd.DataFrame({
            "filename": rec.filename, "split": rec.split,
            "win_idx": np.arange(n_win, dtype=np.int32),
            "start_s": starts, "end_s": starts + p.win_s,
        }))
        ys.append(S.max(axis=1).astype(np.uint8))
        Ss.append(S)

    windows = pd.concat(frames, ignore_index=True)
    y = np.concatenate(ys)
    S = np.concatenate(Ss, axis=0)
    windows["n_species"] = S.sum(axis=1).astype(np.int16)
    return (windows, y, S) if return_species else (windows, y)


def summarise(windows: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Per-split window counts and occupancy — the class balance of the gate task."""
    rows = []
    for split, g in windows.groupby("split"):
        yy = y[g.index.to_numpy()]
        win_s = (g["end_s"] - g["start_s"]).iloc[0]
        rows.append({
            "recordings": g["filename"].nunique(),
            "windows": len(g),
            "hours_of_windows": len(g) * win_s / 3600,
            "bird_windows": int(yy.sum()),
            "occupancy_pct": 100 * yy.mean(),
        })
    return pd.DataFrame(rows, index=[s for s, _ in windows.groupby("split")]).reindex(
        ["train", "val", "test"]).round(2)


def occupancy(ann=None, recordings=None) -> pd.DataFrame:
    """Per-recording bird occupancy, from the annotation boxes alone.

    This is the number that will differ most between Sierra Nevada (dawn chorus,
    selected for density) and a real 24-hour Highland deployment. Reading it
    per-recording rather than in aggregate shows the *range* the gate must cope
    with, which is the transferable quantity.
    """
    if ann is None:
        ann = D.load_annotations()
    if recordings is None:
        recordings = D.load_recordings()
    rows = []
    for rec in recordings.itertuples():
        g = ann[ann["filename"] == rec.filename]
        u = I.merge(g[["start_s", "end_s"]].to_numpy()) if len(g) else np.zeros((0, 2))
        rows.append({
            "filename": rec.filename, "date": rec.date, "hour": rec.hour,
            "block": rec.block, "duration_s": rec.duration_s,
            "n_boxes": len(g), "n_species": g["code"].nunique(),
            "bird_s": I.total(u),
            "occupancy_pct": 100 * I.total(u) / rec.duration_s,
        })
    return pd.DataFrame(rows)


def gap_stats(ann=None, recordings=None) -> np.ndarray:
    """Lengths of every silent (unannotated) gap, pooled across recordings.

    The gap *distribution*, not the total silence, is what decides whether a gate
    can save anything: a hangover of 2 s reclaims nothing from gaps shorter than
    2 s, no matter how many hours of them there are.
    """
    if ann is None:
        ann = D.load_annotations()
    if recordings is None:
        recordings = D.load_recordings()
    dur = dict(zip(recordings["filename"], recordings["duration_s"]))
    gaps = []
    for f, g in ann.groupby("filename"):
        u = I.merge(g[["start_s", "end_s"]].to_numpy())
        prev = 0.0
        for s, e in u:
            if s > prev:
                gaps.append(s - prev)
            prev = e
        if prev < dur.get(f, 3600.0):
            gaps.append(dur.get(f, 3600.0) - prev)
    return np.array(gaps)
