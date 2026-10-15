"""Sweep theta on the val split -> operating curve + named operating points.

    ../.venv/bin/python -m scripts.sweep_threshold                 # mfcc, the shipping arm
    ../.venv/bin/python -m scripts.sweep_threshold --arm cnn-int8  # trains the network first
    ../.venv/bin/python -m scripts.sweep_threshold --n 30          # coarser grid, faster

The deliverable is the table at the bottom: three named thetas, each with the
retention, saving, worst recording and per-species floor it actually buys. That
table is what `export.py` writes into `model_params.h`, so this script is the
only place those numbers should ever come from.

Everything is selected on **val**. Test is not read here at all — `scripts.train`
and `scripts.compare_arms` own the test-split reporting, and the whole point of
calibrating separately is that the number shipped was not chosen on the split it
is reported on.
"""
from __future__ import annotations

import argparse

import pandas as pd

from birdgate import calibrate as C, frontends as FE, provenance as P, svm as S
from birdgate.config import FrontEndParams, GateConfig, GateParams, WindowParams
from scripts import train as T

LINEAR = ("carihc", "bands", "mfcc")


def val_scores(arm: str, seed: int = 0):
    """Normalised scores for every window, plus the frame, for one arm.

    Returns the FULL window frame and score vector; the caller subsets to val.
    The CNN comes back through `scripts.train_cnn` because it changes the
    classifier as well as the front-end, but its `_z` is aligned to the same rows.
    """
    wp = WindowParams()
    windows, y, X = T.build_xy(arm if arm in LINEAR else "mfcc", wp)

    if arm in LINEAR:
        names = FE.build(FrontEndParams(kind=arm), 32_000).feature_names
        tr = windows.index[windows["split"] == "train"].to_numpy()
        model = S.fit(X[tr], y[tr], arm, names, seed=seed)
        return windows, T.scores(model, windows, X, T.ADAPT, train_idx=tr), model

    from scripts.train_cnn import run as run_cnn
    r = run_cnn(target=0.98, verbose=True)[arm]
    return windows, r["_z"], r.get("_model")


def main(arm: str = "mfcc", n: int = 60, seed: int = 0, min_boxes: int = 20,
         max_ci_width: float = 0.10):
    pd.set_option("display.width", 200)
    windows, z, model = val_scores(arm, seed)

    va = windows.index[windows["split"] == "val"].to_numpy()
    val = windows[windows["split"] == "val"].reset_index(drop=True)
    print(f"  {arm}: sweeping {n} thetas over {len(va):,} val windows "
          f"({val['filename'].nunique()} recordings)", flush=True)

    curve = C.sweep(val, z[va], T.GATE, n=n, min_boxes=min_boxes,
                    max_ci_width=max_ci_width, seed=seed)
    table = C.named_operating_points(curve)

    pct = ["retention", "duty", "saving", "whole_boxes", "event90",
           "worst_rec", "species_floor"]
    show = curve.copy()
    show[pct] = (100 * show[pct]).round(2)
    print("\n" + "=" * 110)
    print(f"Operating curve, arm={arm}, VAL split (every column a percentage)")
    print("=" * 110)
    print(show.round(4).to_string(index=False))

    tshow = table.copy()
    tshow[pct] = (100 * tshow[pct]).round(2)
    print("\n" + "=" * 110)
    print(f"Named operating points, arm={arm}  (met=False: no theta satisfies it)")
    print("=" * 110)
    print(tshow.round(4).to_string(index=False))

    # The adaptive-threshold parameters appear in no earlier manifest (SUMMARY §12),
    # and the anchor quantile is the most consequential number in the system. The
    # table below is meaningless without them, so they ship in the same record.
    cfg = GateConfig(name=f"g4-calibrate-{arm}",
                     frontend=FrontEndParams(kind=arm if arm in LINEAR else "mfcc"),
                     gate=GateParams(**T.GATE), adapt=T.ADAPT, seed=seed)
    P.write(cfg, metrics={"curve": curve.to_dict("records"),
                          "points": table.to_dict("records"),
                          "gate": dict(T.GATE),
                          "adapt": vars(T.ADAPT),
                          "min_boxes": min_boxes,
                          "max_ci_width": max_ci_width},
            notes=f"Threshold calibration on val, arm={arm}, {n} thetas")
    return curve, table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mfcc", choices=[*LINEAR, "cnn", "cnn-int8"])
    ap.add_argument("--n", type=int, default=60, help="thetas on the grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-boxes", type=int, default=20,
                    help="species with fewer boxes are too noisy to set the floor")
    ap.add_argument("--max-ci-width", type=float, default=0.10,
                    help="drop species whose bootstrap CI is wider than this")
    a = ap.parse_args()
    main(a.arm, a.n, a.seed, a.min_boxes, a.max_ci_width)
