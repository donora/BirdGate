"""Run every arm under identical conditions and print the comparison table.

    ../.venv/bin/python -m scripts.compare_arms

    ../.venv/bin/python -m scripts.compare_arms --no-cnn   # skip the network

Held identical across arms: windows, recording split, gate state machine (the oracle
knee: 1 s pre-roll, 1 s hangover, 20 s minimum-on), threshold-selection rule
(lowest duty cycle on **val** meeting the retention target), and metrics.

For the three filterbank arms only the front-end changes, so a difference is
attributable to the front-end. **`cnn` changes the classifier as well**, so a win
there is a win for the pair, not for either half — which is the honest form of the
question it exists to answer (docs/lab-notebook.md §9b: does the data-scarcity argument
for an SVM survive when the data is not scarce?). `cnn-int8` is the deployable
one; the float row is there to show what quantisation cost.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from birdgate import provenance as P, svm as S, windows as W
from birdgate.config import GateConfig, GateParams, WindowParams
from scripts.train import ADAPT, GATE, run


def _auc_ap(y, s):
    """ROC-AUC and average precision, computed here so sklearn stays out of the loop."""
    y = np.asarray(y, dtype=np.float64)
    o = np.argsort(-np.asarray(s, dtype=np.float64))
    yo = y[o]
    tp, fp = np.cumsum(yo), np.cumsum(1 - yo)
    auc = float(np.trapezoid(tp / max(tp[-1], 1), fp / max(fp[-1], 1)))
    prec = tp / np.arange(1, len(yo) + 1)
    return round(auc, 4), round(float((prec * yo).sum() / max(tp[-1], 1)), 4)

ARMS = ("carihc", "bands", "mfcc")
CNN_ARMS = ("cnn", "cnn-int8")
COLS = ["arm", "retention", "worst_rec", "whole_boxes", "duty", "saving",
        "species_floor", "floor_species", "n_admissible", "n_scored",
        "files_per_hr", "mean_len_s", "model_bytes", "mmac_s", "pct_m4f"]


def _run_arms(target: float, with_cnn: bool):
    """Every arm's result dict, keyed by arm name.

    The CNN lives in `scripts.train_cnn` because it changes the classifier as well
    as the front-end, but it comes back through the same `report()` and is scored
    on the same rows, so it belongs in the same table. `mmac_s` for the CNN counts
    the log-mel FFT *and* the convolutions — see `cnn.cost`.
    """
    out = {arm: run(arm, target) for arm in ARMS}
    if with_cnn:
        from scripts.train_cnn import run as run_cnn
        out.update(run_cnn(target=target, verbose=True))
    return out


def main(target: float = 0.98, with_cnn: bool = True):
    pd.set_option("display.width", 220)
    rows, models = [], _run_arms(target, with_cnn)
    for arm, r in models.items():
        r["mean_len_s"] = 3600 * r["duty"] / 100 / max(r["files_per_hr"], 1e-9)
        rows.append({k: r[k] for k in COLS})
        print(f"  {arm:9s} done — retention {r['retention']:.1f}%  "
              f"saving {r['saving']:.1f}%", flush=True)

    windows, y = W.build_windows(p=WindowParams())
    windows = windows.reset_index(drop=True)
    te = windows.index[windows["split"] == "test"].to_numpy()
    for r, arm in zip(rows, models):
        # Threshold-free discrimination, on the same normalised scores the gate
        # sees. It cannot be read as a ranking on its own — docs/lab-notebook.md §10 exists
        # because a better AUC can still gate worse — but a large gap here says
        # the difference is in the model rather than in the calibration.
        r["auc"], r["ap"] = _auc_ap(y[te], models[arm]["_z"][te])

    table = pd.DataFrame(rows).round(2)
    print("\n" + "=" * 110)
    print(f"All arms, test split, theta chosen on val for >={100*target:.0f}% retention")
    print("=" * 110)
    print(table.to_string(index=False))

    print("\n" + "-" * 110)
    print("Diagnostic — saving at MATCHED retention, theta swept on test (not a headline)")
    print("-" * 110)
    print(frontier(models, windows).to_string(index=False))

    print("\nTop weights per arm (the model is small enough to just read):")
    for arm in ARMS:
        print(f"  {arm:7s} " + ", ".join(f"{n}={w:+.0f}"
              for n, w in S.top_weights(models[arm]["_model"], 5)))

    P.write(GateConfig(name="g3-compare", gate=GateParams(**GATE), adapt=ADAPT),
            metrics={"table": table.to_dict("records"),
                     "gate": dict(GATE), "adapt": vars(ADAPT),
                     "min_boxes": 20, "max_ci_width": 0.10},
            notes=f"Arm comparison, target retention {target}, "
                  f"species floor guarded on bootstrap CI width")
    return table, models


def frontier(models, windows, gate=None, targets=(0.975, 0.98, 0.985)):
    """Saving at *matched* retention, theta swept on **test**. A diagnostic only.

    ⚠️ This selects theta on the split it reports, which is exactly the thing
    `train.pick_theta` exists to avoid, and it is therefore **not** a headline
    number. It answers a different and narrower question: the val-selected table
    compares arms at whatever retention each one happens to land on, and when two
    arms land 1.2 points apart on retention their savings are not comparable at
    all — the steepness of the trade near 98% is such that a point of retention is
    worth more than the entire gap between arms. This isolates the model from that
    calibration noise. Read the two together: a win here and a loss in the
    val-selected table means the arm discriminates well and calibrates badly, which
    for an unattended deployment is not a win.
    """
    from birdgate import metrics as M
    from birdgate.config import GateParams
    from scripts.train import GATE

    gate = GATE if gate is None else gate
    test = windows[windows["split"] == "test"].reset_index(drop=True)
    te = windows.index[windows["split"] == "test"].to_numpy()
    groups = list(windows[windows["split"] == "test"].groupby("filename"))
    rows = []
    for arm, r in models.items():
        z = r["_z"]
        st = z[te]
        pts = []
        for th in np.quantile(st, np.linspace(0.02, 0.995, 80)):
            gp = GateParams(theta_on=float(th), theta_off=None, **gate)
            m = M.evaluate(test, st, gp)
            worst = min(100 * M.evaluate(g.reset_index(drop=True), z[g.index.to_numpy()],
                                         gp)["birdsong_retention"] for _, g in groups)
            pts.append((m["birdsong_retention"], 100 * m["saving"],
                        100 * m["event_recall@1"], worst))
        pts = np.array(sorted(pts))
        for t in targets:
            if pts[:, 0].max() < t:
                continue
            i = int(np.argmax(pts[:, 0] >= t))
            rows.append({"arm": arm, "retention": 100 * t, "saving": pts[i, 1],
                         "whole_boxes": pts[i, 2], "worst_rec": pts[i, 3]})
    return pd.DataFrame(rows).round(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--no-cnn", action="store_true",
                    help="skip the CNN arm (it trains a network; the rest is seconds)")
    a = ap.parse_args()
    main(a.target, with_cnn=not a.no_cnn)
