"""Adaptive thresholding — make the score mean the same thing everywhere.

**The problem this solves.** The arm comparison (docs/lab-notebook.md §8) found that one absolute threshold does not transfer
across recordings: at a fixed theta, birdsong retention swung from 54.9% to 99.3%
between test recordings. Gain, microphone, weather and background level move the
whole score distribution bodily, so "score > -0.037" means something different in
each hour. In a months-long unattended deployment that is the failure mode nobody
is there to notice.

**The fix is CFAR** (constant false-alarm rate, from radar): don't threshold the
raw score, threshold it *relative to the local background*. Track the background
in a trailing window and require the score to stand a margin above it.

    z_t = (s_t - b_t) / d_t          b_t = background level, d_t = background spread

then apply one global threshold to `z`. If it starts raining, `b_t` rises and the
gate demands a correspondingly louder bird — which is exactly the behaviour asked
for, and it needs no retraining.

⚠️ **Anchor on a LOW quantile, not the median, and never a high one.** A threshold
set at the q-th percentile of recent scores is a *duty-cycle controller*, not a
detector: it records q% of the time no matter what, so five minutes of continuous
song would still be q% recorded, and five minutes of silence would still trigger
q% of the time. Anchoring at ~0.3 keeps `b_t` inside the background population
even when occupancy is high (SNE has hours at 76% occupancy, and 1 - 0.76 = 0.24,
which is why 0.3 is about as high as this can safely go here).

**Everything here is causal** — window `t` uses only scores strictly before `t`, as
the device must. Two estimators, both cheap enough for a Cortex-M4:

⚠️⚠️ **Subtract the local background; do NOT divide by a local spread.** The first
version of this module did both, and it measurably destroyed within-recording
discrimination. On SNE_026 (a *training* recording, 0.1% bird for its first 45
minutes) the raw model separated cleanly — AUC 0.925, silence at -0.27 +/- 0.15,
birds at +1.13 — and normalising by a trailing MAD dropped it to 0.843 (`ema`) and
0.739 (`quantile`), with silence z-scores reaching +12.6 and +337 respectively.

The mechanism: in a long quiet stretch the local spread collapses (0.084 here),
so dividing by it amplifies meaningless fluctuation into large z, and the tail of
silence overtakes real birds. A minimum-on time then turns each spurious blip into
ten seconds of recording. The symptom is a gate that chatters through exactly the
silence it was supposed to skip.

What actually varies between recordings is the **offset** (gain, mic, background
level), not the spread of the score. So `scale='global'` uses one constant fitted
on the training split, which removes the offset without amplifying quiet. The
local-spread modes are kept only so this finding stays reproducible.

- `quantile`: exact order statistics over a trailing ring buffer. At one score per
  second a 120 s window is 120 floats (480 B). Accurate, trivially auditable.
- `ema`: two running floats, updated preferentially while the gate is **closed**,
  so singing birds do not inflate the background estimate. ~8 bytes and 4 ops per
  second. Degrades if the gate stays open for a long time, so it also creeps
  slowly while open rather than freezing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AdaptParams:
    """How the background is tracked. `mode='fixed'` reproduces the fixed-threshold results (docs/lab-notebook.md §8) exactly."""
    mode: Literal["fixed", "quantile", "ema"] = "quantile"
    window_s: float = 300.0
    # 0.2, not 0.3 and emphatically not 0.5. Measured on SNE_026's 45-minute quiet
    # stretch, the fraction of true silence recorded runs 5.2% at 0.2, 17.1% at
    # 0.3 and 75.0% at 0.5 — the last being the duty-cycle-controller failure
    # described above, arriving exactly as predicted.
    bg_quantile: float = 0.20
    scale: Literal["global", "mad", "iqr", "none"] = "global"
    global_scale: float = 1.0     # set from TRAIN scores; see the warning below
    ema_alpha: float = 0.002          # ~500 s time constant when closed
    ema_alpha_open: float = 0.0002    # 10x slower while recording
    min_scale: float = 1e-3           # floor, so a flat window can't divide by ~0

    # Score aggregation over the last `agg_n` decisions, applied AFTER background
    # normalisation. `max` is the measured winner: it cut recorded silence from
    # 12.6% to 4.5% at unchanged saving and better whole-box capture.
    #
    # Why max and not mean/median, which look like the more principled "require
    # sustained evidence" choice: max makes the score more sensitive, so the same
    # recall is reachable at a HIGHER theta — and it is the higher theta that
    # rejects silence. mean/median do the opposite. They suppress short calls, so
    # theta must come DOWN to keep recall, and recorded silence rises to 27-53%.
    # Measured, not assumed; the intuition points the wrong way.
    #
    # This is NOT redundant with `hangover_s`. Hangover only extends an already
    # open gate; max-of-n also changes where the gate opens, and therefore what
    # theta can be.
    agg: Literal["none", "max", "mean", "median"] = "max"
    agg_n: int = 3


def normalise(scores: np.ndarray, p: AdaptParams, win_s: float = 1.0) -> np.ndarray:
    """Causally convert raw scores to background-relative z-scores.

    Returns `scores` unchanged when `mode='fixed'`, so callers need no branch.
    """
    s = np.asarray(scores, dtype=np.float64)
    if p.mode == "fixed" or len(s) == 0:
        return s
    if p.mode == "ema":
        return _ema_normalise(s, p)

    w = max(2, int(round(p.window_s / win_s)))
    # Pad with the leading value so early windows have a defined background rather
    # than a partial one. The alternative (skip until the buffer fills) leaves the
    # first two minutes of every deployment ungated, which is worse.
    pad = np.concatenate([np.full(w, s[0]), s])
    view = np.lib.stride_tricks.sliding_window_view(pad, w)[:-1]   # strictly past
    bg = np.quantile(view, p.bg_quantile, axis=1)
    if p.scale == "global":
        sc = np.full_like(bg, p.global_scale)
    elif p.scale == "mad":
        sc = np.median(np.abs(view - np.median(view, axis=1, keepdims=True)), axis=1)
    elif p.scale == "iqr":
        sc = np.subtract(*np.quantile(view, [0.75, 0.25], axis=1))
    else:
        sc = np.ones_like(bg)
    return (s - bg) / np.maximum(sc, p.min_scale)


def _ema_normalise(s: np.ndarray, p: AdaptParams) -> np.ndarray:
    """Two-float background tracker. The device-cheap variant.

    Uses the *previous* z to decide whether the gate was plausibly open, so the
    background estimate is not dragged upward by the birds it is meant to detect.
    """
    b = s[0]
    use_global = p.scale == "global"
    d = p.global_scale if use_global else max(
        np.abs(s[:50] - s[0]).mean() if len(s) > 1 else 1.0, p.min_scale)
    out = np.empty_like(s)
    for i, v in enumerate(s):
        out[i] = (v - b) / max(d, p.min_scale)
        a = p.ema_alpha_open if out[i] > 2.0 else p.ema_alpha
        b += a * (v - b)
        if not use_global:
            d += a * (abs(v - b) - d)
    return out


def aggregate(z: np.ndarray, p: AdaptParams) -> np.ndarray:
    """Causal pooling over the last `agg_n` scores. Costs `agg_n` floats on device."""
    if p.agg == "none" or p.agg_n <= 1 or len(z) == 0:
        return z
    pad = np.concatenate([np.full(p.agg_n - 1, z[0]), z])
    sw = np.lib.stride_tricks.sliding_window_view(pad, p.agg_n)
    if p.agg == "max":
        return sw.max(axis=1)
    if p.agg == "mean":
        return sw.mean(axis=1)
    return np.median(sw, axis=1)


def normalise_by_recording(windows, scores: np.ndarray, p: AdaptParams,
                           win_s: float = 1.0) -> np.ndarray:
    """Apply `normalise` per recording, never across a file boundary.

    A device reboots between deployments and its background tracker restarts; more
    to the point, letting one hour's background leak into the next would be the
    exact cross-recording contamination this module exists to remove.
    """
    out = np.empty(len(scores), dtype=np.float64)
    for _, g in windows.groupby("filename", sort=False):
        idx = g.index.to_numpy()
        order = np.argsort(g["start_s"].to_numpy())
        out[idx[order]] = aggregate(normalise(scores[idx[order]], p, win_s), p)
    return out
