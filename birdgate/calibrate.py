"""Turn the score into a knob: sweep theta, name the operating points.

The threshold IS the product. Everything a deployer wants to trade — "record more,
burn card" vs "record less, risk missing a call" — is one float in
`model_params.h`. Ship a table, not a number.

Proposed named points (final values set by measurement, not by taste):

| name       | intent                                                          |
|------------|-----------------------------------------------------------------|
| `paranoid` | >= 99.5% retention, per-species floor >= 98%. Saving is whatever |
|            | is left. For short deployments where the card is not the         |
|            | constraint and analysis time is.                                 |
| `balanced` | >= 98% retention at the knee of the curve. The default.          |
| `frugal`   | maximum saving subject to per-species floor >= 90%. For           |
|            | multi-month deployments where the card fills before the battery. |

Open question worth settling first (docs/lab-notebook.md section 9): **what is actually
scarce?** A 128 GB card holds ~23 days of ungated 32 kHz mono, and 3xAA runs ~48
days. If card > deployment length, the gate is buying analysis time, not storage,
and theta should sit far more generously than a storage-driven sweep would put it.

Calibration split: use `val` recordings only. Choosing theta on test and then
reporting test retention is the oldest way to publish a number that does not
replicate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import intervals as I
from . import metrics as M
from .config import GateParams

# Each named point maximises saving subject to its own floors, so the table below
# is one rule applied three times rather than three separate rules. Values are the
# acceptance criteria from this module's docstring; the thetas that meet them are
# measured, never chosen by taste.
POINTS = {
    "paranoid": {"retention": 0.995, "species_floor": 0.98},
    "balanced": {"retention": 0.98},
    "frugal": {"species_floor": 0.90},
}


def sweep(windows, z, gate, thetas=None, ann=None, recordings=None,
          min_boxes: int = 20, max_ci_width: float = 0.10, n_boot: int = 500,
          n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Sweep theta and record what the named points are actually defined by.

    `train.pick_theta` already sweeps theta, but its curve carries only retention,
    duty and event recall — and two of the three named points below are defined by
    the metric it omits, the per-SPECIES floor. That floor is the deployment risk
    (`metrics.per_species_retention`): a theta can look fine in aggregate while
    quietly dropping one quiet species, and because the discarded audio no longer
    exists, nothing downstream can detect it. `worst_rec` is here for the same
    reason at recording granularity — it is the 44-point spread that motivated the
    adaptive threshold in the first place.

    `windows` and `z` must be the **val** split already subset and row-aligned
    (`windows.reset_index(drop=True)`, `z[val_idx]`), the same shape `pick_theta`
    works in. Selecting theta on test and then reporting test retention is the
    oldest way to publish a number that does not replicate.

    `gate` is the state-machine dict (`train.GATE`), and must be the SAME one the
    test split will later be scored through — see `train.pick_theta`.

    `max_ci_width` is the guard that makes `species_floor` mean anything. Without
    it the floor is set by whichever species the split happens to under-sample: on
    val that is Brown-headed Cowbird, 41 s of song, 61% retention, 95% CI 32-95.
    The default 0.10 keeps only species whose retention is pinned to within ten
    points. `floor_species` and `n_admissible` travel with every row so the number
    is never anonymous — see `metrics.admissible` for why the admissible set is
    theta-dependent and where that is safe to read.
    """
    if ann is None:
        ann = D.load_annotations()
    if recordings is None:
        recordings = D.load_recordings()

    files = set(windows["filename"].unique())
    ann = ann[ann["filename"].isin(files)]

    if thetas is None:
        thetas = np.quantile(z, np.linspace(0.02, 0.995, n))

    # Union per file once: retention is measured against the union of boxes, not
    # their sum, and birds sing over each other constantly.
    boxes = {f: I.merge(g[["start_s", "end_s"]].to_numpy())
             for f, g in ann.groupby("filename", sort=False)}

    rows = []
    for th in thetas:
        gp = GateParams(theta_on=float(th), theta_off=None, **gate)
        m = M.evaluate(windows, z, gp, ann=ann, recordings=recordings)
        regions = m["_regions"]

        # Worst single recording, from the regions `evaluate` already returned
        # rather than by re-running the state machine once per file.
        worst = 1.0
        for f, u in boxes.items():
            bird = I.total(u)
            if bird <= 0:
                continue
            kept = I.total(I.intersect(u, regions.get(f, np.zeros((0, 2)))))
            worst = min(worst, kept / bird)

        ps = M.per_species_retention(windows, regions, ann=ann, n_boot=n_boot, seed=seed)
        ok = M.admissible(ps, min_boxes, max_ci_width)
        rows.append({
            "theta": float(th),
            "retention": m["birdsong_retention"],
            "duty": m["duty_cycle"],
            "saving": m["saving"],
            "whole_boxes": m["event_recall@1"],
            "event90": m["event_recall@0.9"],
            "worst_rec": worst,
            "species_floor": float(ok["retention"].min()) if len(ok) else float("nan"),
            "floor_species": ok.loc[ok["retention"].idxmin(), "common"] if len(ok) else "-",
            "n_admissible": len(ok),
            "n_scored": int((ps["n_boxes"] >= min_boxes).sum()),
        })
    return pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)


def named_operating_points(curve: pd.DataFrame, points: dict | None = None) -> pd.DataFrame:
    """The table that ships in `model_params.h`: one theta per named point.

    Feasible means every floor for that point is met somewhere on the curve; among
    the thetas that qualify, take the one that saves the most card. A point no
    theta can satisfy is returned with `met=False` rather than dropped, carrying
    the theta that comes closest on its binding constraint — a `paranoid` that
    cannot be reached is a fact about the model worth shipping, not an error.
    """
    points = POINTS if points is None else points
    out = []
    for name, floors in points.items():
        ok = curve
        for col, lo in floors.items():
            ok = ok[ok[col] >= lo]
        if len(ok):
            row, met = ok.loc[ok["saving"].idxmax()], True
        else:
            # Nothing qualifies: report the theta with the best worst-constraint,
            # so the table shows how far short the point falls and on which floor.
            slack = np.min([curve[c] / lo for c, lo in floors.items()], axis=0)
            row, met = curve.loc[int(np.argmax(slack))], False
        out.append({"point": name, "met": met, **{k: row[k] for k in curve.columns},
                    "binds": ",".join(sorted(floors))})
    return pd.DataFrame(out)
