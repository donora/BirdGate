"""The gate state machine — per-window scores to recorded regions.

**This is the piece that decides what lands on the SD card**, and it is mirrored
exactly in `firmware/common/gate.c`. If the two drift, the simulator lies and every
number in every report is wrong. The C parity harness pins them together (ROADMAP F6).

It is deliberately implemented before any model exists, because the state machine
alone moves retention by more than most modelling choices will. Measuring it first
means we can never later mistake a hangover change for a model win.

Three mechanisms, each buying recall:

**Hysteresis.** Open at `theta_on`, close only below `theta_off < theta_on`. A
single threshold makes the gate chatter through a call whose score wobbles across
it, chopping the recording into fragments — the worst possible output, since a
fragmented call is neither usable nor cleanly absent.

**Pre-roll.** On opening, also keep the previous `pre_roll_s`. You cannot detect a
call before it starts, so with no pre-roll every recorded event is missing its
onset — the most diagnostic part of most vocalisations, and the part an automated
downstream classifier most needs. On the AudioMoth this is nearly free: the ring
buffer is *already captured* audio, so opening the gate rewinds `read_pointer`
rather than allocating anything. But the budget is smaller than it looks: the
buffer is 8 x 32 KB = 256 KB, which at 16 kHz is 8 s and at our 32 kHz is only
**4.1 s**, leaving ~3 s of usable pre-roll. Doubling the sample rate halved it.

**Hangover.** Stay open `hangover_s` after the last positive, so a phrase comes
out whole rather than as syllables. It is also the single largest consumer of the
saving budget; that trade is the point of `oracle_curve()`.

Everything here is **causal** — no window's decision uses a future score — exactly
as the device must be. Pre-roll is the sole exception and is not a violation: it
back-fills audio already sitting in the buffer, which is a memory fact, not a
prophecy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import intervals as I
from .config import GateParams


def resolve_thresholds(p: GateParams) -> tuple[float, float]:
    """(theta_on, theta_off). `theta_off=None` means derive it from the margin."""
    off = p.theta_on - p.hysteresis_margin if p.theta_off is None else p.theta_off
    return float(p.theta_on), float(min(off, p.theta_on))


def run(scores: np.ndarray, p: GateParams, win_s: float) -> np.ndarray:
    """Scores for ONE recording -> boolean 'this window was written to the card'.

    Call per recording. Running it over concatenated recordings would let a
    hangover leak across a file boundary, which cannot happen on the device.

    The loop is the device, step for step:

    - **IDLE**: score every window. `score > theta_on` opens the gate and
      back-fills `pre_roll_s` from the ring buffer.
    - **RECORDING**: audio goes to the card every window regardless. The
      classifier only re-runs every `recheck_period_s`, and **the gate can only
      close at a recheck instant** — between checks it is committed. That is a
      faithful model of duty-cycling the classifier while the SD write path is
      busy, and it means `recheck_period_s` acts as a coarse floor on the closing
      latency: a 5 s recheck period is already worth ~2.5 s of hangover on
      average, so paying for a long `hangover_s` on top is buying the same thing
      twice.
    - **`min_on_s`** holds the gate open regardless of score. Its justification is
      not acoustic — it is filesystem and listener ergonomics (see config.py).
      It is the most expensive parameter here, because it spends duty cycle on
      windows already known to be empty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    theta_on, theta_off = resolve_thresholds(p)
    pre = int(np.ceil(p.pre_roll_s / win_s))
    recheck = max(1, int(round(p.recheck_period_s / win_s)))
    min_on = p.min_on_s

    rec = np.zeros(len(scores), dtype=bool)
    is_open = False
    opened_at = last_pos = -1
    for i, s in enumerate(scores):
        if is_open:
            elapsed = i - opened_at
            # The classifier is only awake on recheck instants.
            if elapsed > 0 and elapsed % recheck == 0:
                if s > theta_off:
                    last_pos = i
                quiet_for = (i - last_pos) * win_s
                if quiet_for > p.hangover_s and elapsed * win_s >= min_on:
                    is_open = False
        if not is_open and s > theta_on:
            is_open = True
            opened_at = last_pos = i
            rec[max(0, i - pre):i] = True   # the ring-buffer rewind
        rec[i] = rec[i] or is_open
    return rec


def run_all(windows: pd.DataFrame, scores: np.ndarray, p: GateParams,
            win_s: float | None = None) -> np.ndarray:
    """`run` over every recording in a windows table, respecting file boundaries."""
    if not windows.index.equals(pd.RangeIndex(len(windows))):
        raise ValueError(
            "run_all needs a 0..n-1 RangeIndex — `scores` is positional and a "
            "filtered frame would silently misalign it. Call .reset_index(drop=True)."
        )
    if win_s is None:
        win_s = float((windows["end_s"] - windows["start_s"]).iloc[0])
    out = np.zeros(len(windows), dtype=bool)
    for _, g in windows.groupby("filename", sort=False):
        idx = g.index.to_numpy()
        order = np.argsort(g["start_s"].to_numpy())
        out[idx[order]] = run(scores[idx[order]], p, win_s)
    return out


def regions(windows: pd.DataFrame, recorded: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean per-window mask -> merged (start_s, end_s) regions per recording.

    Merging matters: adjacent recorded windows overlap when the hop is smaller
    than the window, and a raw list of window spans would double-count the
    overlap in every duty-cycle number downstream.
    """
    out = {}
    for f, g in windows.groupby("filename", sort=False):
        m = recorded[g.index.to_numpy()]
        iv = g.loc[g.index[m], ["start_s", "end_s"]].to_numpy()
        out[f] = I.merge(iv) if len(iv) else np.zeros((0, 2))
    return out


def oracle_scores(y: np.ndarray) -> np.ndarray:
    """Ground truth as a score stream: what a perfect detector would emit.

    Feeding this through `run` isolates the state machine's own cost. Use
    `GateParams(theta_on=0.5, hysteresis_margin=0.0)` so hysteresis is inert and
    only pre-roll and hangover are being measured.
    """
    return np.asarray(y, dtype=np.float64)
