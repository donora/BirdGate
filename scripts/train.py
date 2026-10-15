"""Train one arm and score it through the real gate.

    ../.venv/bin/python -m scripts.train --arm carihc

Everything except the arm is held fixed — same windows, same recording split, same
gate state machine, same metrics — so a difference between arms is attributable to
the arm. Selection happens on **val**; test is read once, at the end.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from birdgate import adapt as A, cache as C, data as D, frontends as FE, gate as G
from birdgate import metrics as M, provenance as P, svm as S, windows as W
from birdgate.config import FrontEndParams, GateConfig, GateParams, WindowParams

# Fixed by the oracle knee (docs/lab-notebook.md §6) and re-confirmed driving the real model: the
# pre-roll/hangover optimum is the same either way. Held constant across every arm
# so the state machine can never be confused with the model.
#
# `min_on_s` is NOT an acoustic parameter — it is filesystem and listener
# ergonomics (see config.py). 20 s is the default: against 10 s it costs 0.16
# points of saving and lifts the worst recording 93.2% -> 94.1% with 7% fewer
# files, and at *matched* retention it costs nothing at all. Sweep it in the app.
GATE = dict(pre_roll_s=1.0, hangover_s=1.0, hysteresis_margin=0.25, min_on_s=20.0)

# Default: track the background so one theta means the same thing in every
# recording. `ema` costs 8 bytes on the device and lifted the worst recording from
# 67.5% to 90.4% retention.
# Trailing 300 s background at the 0.2 quantile, scaled by a GLOBAL constant fitted
# on train. Dividing by a *local* spread was measurably harmful — see adapt.py.
ADAPT = A.AdaptParams(mode="quantile", window_s=300.0, bg_quantile=0.20, scale="global")


def build_xy(arm: str, wp: WindowParams):
    windows, y = W.build_windows(p=wp)
    windows = windows.reset_index(drop=True)
    raw = C.assemble(windows, arm, wp)
    X = FE.normalise(raw, FrontEndParams(kind=arm))
    return windows, y, X


def scores(model, windows, X, adapt=ADAPT, train_idx=None):
    """Raw margin, then background-normalised per recording (docs/lab-notebook.md §9).

    `global_scale` is fitted on the TRAIN scores and travels with the model — it is
    one more constant in `model_params.h`, not something the device estimates.
    """
    raw = model.score(X)
    if adapt.scale == "global" and train_idx is not None:
        adapt = replace(adapt, global_scale=float(raw[train_idx].std()) or 1.0)
    return A.normalise_by_recording(windows, raw, adapt)


def pick_theta(model, windows, y, X, target_retention=0.98, z=None, gate=None):
    """Choose theta on VAL only: the lowest duty cycle meeting the retention target.

    Selecting theta on test and then reporting test retention is the oldest way to
    publish a number that does not replicate.

    ⚠️ `gate` must be the **same** state machine the test split will be scored
    through. It defaults to `GATE`, and passing a different one to `run()` while
    leaving this on the default silently selects theta under one policy and reports
    under another — which is how a `min_on_s` sweep ends up measuring the wrong
    thing entirely.
    """
    if z is None:
        z = scores(model, windows, X)
    gate = GATE if gate is None else gate
    val = windows[windows["split"] == "val"].reset_index(drop=True)
    sv = z[windows.index[windows["split"] == "val"].to_numpy()]
    rows = []
    for th in np.quantile(sv, np.linspace(0.02, 0.995, 60)):
        gp = GateParams(theta_on=float(th), theta_off=None, **gate)
        m = M.evaluate(val, sv, gp)
        rows.append((float(th), m["birdsong_retention"], m["duty_cycle"],
                     m["event_recall@0.9"]))
    curve = pd.DataFrame(rows, columns=["theta", "retention", "duty", "event90"])
    ok = curve[curve["retention"] >= target_retention]
    best = (ok.loc[ok["duty"].idxmin()] if len(ok)
            else curve.loc[curve["retention"].idxmax()])
    return float(best["theta"]), curve


def run(arm: str, target: float = 0.98, seed: int = 0, gate: dict | None = None,
        adapt: A.AdaptParams | None = None) -> dict:
    wp = WindowParams()
    windows, y, X = build_xy(arm, wp)
    fe = FE.build(FrontEndParams(kind=arm), 32_000)
    names = fe.feature_names

    gate = GATE if gate is None else gate
    tr = windows.index[windows["split"] == "train"].to_numpy()
    model = S.fit(X[tr], y[tr], arm, names, seed=seed)
    z = scores(model, windows, X, ADAPT if adapt is None else adapt, train_idx=tr)
    theta, curve = pick_theta(model, windows, y, X, target, z=z, gate=gate)

    cost = FE.cost_model(fe, n_weights=len(model.w))
    return {**report(windows, z, theta, gate, cost),
            "arm": arm, "n_features": len(model.w), "model_bytes": model.n_bytes,
            "_model": model, "_curve": curve}


def report(windows, z, theta: float, gate: dict, cost: dict, min_boxes: int = 20,
           max_ci_width: float | None = 0.10, n_boot: int = 500, seed: int = 0) -> dict:
    """Score the test split through the gate. **Every arm goes through here.**

    Shared rather than reimplemented per arm on purpose: the whole arm comparison is
    only worth anything if the metric code is byte-identical across arms, so the
    CNN arm (`scripts/train_cnn.py`) calls this same function with its own scores
    and its own cost model.

    `max_ci_width` defaults ON. Before it, `species_floor` was a minimum over
    every species with >= 20 boxes, which let a species with a handful of seconds of
    song set the safety metric for the whole comparison — and because the binding
    species was never named, the column could swing between arms for reasons that
    had nothing to do with the arm. It did not visibly break on test only by luck:
    the worst test species, Pacific-slope Flycatcher, has 37 boxes and 49 s and
    merely happened to score well. `floor_species` and `n_admissible` are returned
    alongside so the number can always be traced to the species that set it.
    """
    te = windows.index[windows["split"] == "test"].to_numpy()
    test = windows[windows["split"] == "test"].reset_index(drop=True)
    st = z[te]
    gp = GateParams(theta_on=theta, theta_off=None, **gate)
    m = M.evaluate(test, st, gp)
    ps = M.per_species_retention(test, m["_regions"], n_boot=n_boot, seed=seed)
    ok = M.admissible(ps, min_boxes, max_ci_width)

    worst = min(100 * M.evaluate(g.reset_index(drop=True), z[g.index.to_numpy()], gp)
                ["birdsong_retention"]
                for _, g in windows[windows["split"] == "test"].groupby("filename"))
    n_reg = sum(len(r) for r in m["_regions"].values())
    return {
        "theta": theta,
        "retention": 100 * m["birdsong_retention"],
        "whole_boxes": 100 * m["event_recall@1"],
        "event90": 100 * m["event_recall@0.9"],
        "duty": 100 * m["duty_cycle"], "saving": 100 * m["saving"],
        "species_floor": 100 * M.retention_floor(ps, min_boxes, max_ci_width),
        "floor_species": ok.loc[ok["retention"].idxmin(), "common"] if len(ok) else "-",
        "n_admissible": len(ok),
        "n_scored": int((ps["n_boxes"] >= min_boxes).sum()),
        "worst_rec": worst,
        "files_per_hr": n_reg / test["filename"].nunique(),
        "mmac_s": cost["macs_per_s"] / 1e6, "pct_m4f": cost["pct_of_48mhz_m4f"],
        "_per_species": ps, "_z": z,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="carihc")
    ap.add_argument("--target", type=float, default=0.98)
    a = ap.parse_args()
    r = run(a.arm, a.target)
    print({k: (round(v, 2) if isinstance(v, float) else v)
           for k, v in r.items() if not k.startswith("_")})
    print("top weights:", S.top_weights(r["_model"]))
    cfg = GateConfig(name=f"g3-{a.arm}", frontend=FrontEndParams(kind=a.arm),
                     gate=GateParams(theta_on=r["theta"], **GATE), adapt=ADAPT)
    P.write(cfg, metrics={k: v for k, v in r.items() if not k.startswith("_")})
