"""Train/val/test split — **by recording**, never by window.

Why by recording: two windows from the same 1-hour file share the same individual
birds, the same microphone, the same background, the same weather. A model can
recognise *the recording* instead of the birds and still score well. Splitting at
the recording level is the only way the number means "generalises to a new
recording".

Why stratified rather than random: 33 recordings across 4 dates and 2 daily
blocks (dawn chorus / evening). A random draw of 6 test recordings can land
all-dawn or miss a date; stratifying over date x block spreads val and test across
both axes.

This is the same algorithm and seed as an earlier, unpublished species classifier
on this dataset, so the two are graded on the same held-out recordings. `tests/test_split.py` pins the result.
"""
from __future__ import annotations

import random

import pandas as pd

from . import data as D

# ~64/18/18. With 33 recordings that lands on 21 train / 6 val / 6 test.
DEFAULT_FRACTIONS = {"train": 0.64, "val": 0.18, "test": 0.18}
SPLITS = ("train", "val", "test")


def _apportion(fractions: dict[str, float], n: int) -> dict[str, int]:
    """Split `n` items into whole numbers matching `fractions` as closely as possible.

    Largest-remainder (Hamilton) apportionment: floor everything, then hand the
    leftovers to whoever was rounded down hardest. Guarantees the parts sum to
    exactly `n`, which plain rounding does not.
    """
    exact = {s: fractions[s] * n for s in SPLITS}
    base = {s: int(exact[s]) for s in SPLITS}
    leftover = n - sum(base.values())
    order = sorted(SPLITS, key=lambda s: (-(exact[s] - base[s]), SPLITS.index(s)))
    for s in order[:leftover]:
        base[s] += 1
    return base


def make_split(recordings: pd.DataFrame | None = None,
               fractions: dict[str, float] = DEFAULT_FRACTIONS,
               seed: int = 0) -> pd.DataFrame:
    """Assign each recording to train / val / test. Deterministic given `seed`.

    Group recordings into (date, block) strata, walk the strata in chronological
    order, and after each one re-apportion the *running total* of recordings seen
    so far. Each stratum then receives the difference between its new target and
    what has already been handed out, so every stratum contributes proportionally
    to all three splits and rounding residue is carried forward instead of dropped.

    The subtlety this avoids: filling each split greedily toward a global quota
    puts the whole first date in train, and val/test silently become a *temporal*
    holdout. Apportioning on running totals keeps every date in every split.

    Returns `recordings` with a `split` column added.
    """
    if recordings is None:
        recordings = D.load_recordings()
    recordings = recordings.copy()

    taken = dict.fromkeys(SPLITS, 0)
    assignment: dict[str, str] = {}
    processed = 0

    rng = random.Random(seed)
    for _, group in recordings.groupby(["date", "block"], sort=True):
        members = list(group["filename"])
        rng.shuffle(members)        # seeded, so reproducible

        processed += len(members)
        target = _apportion(fractions, processed)
        need = {s: target[s] - taken[s] for s in SPLITS}

        for filename in members:
            # Hand each recording to the split still owed the most from this
            # stratum. Ties break on the fixed SPLITS order, so it is deterministic.
            split = max(SPLITS, key=lambda s: (need[s], -SPLITS.index(s)))
            assignment[filename] = split
            need[split] -= 1
            taken[split] += 1

    recordings["split"] = recordings["filename"].map(assignment)
    return recordings
