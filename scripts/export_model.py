"""Train an arm, calibrate theta on val, and emit `model_params.h`.

    ../.venv/bin/python -m scripts.export_model --arm mfcc-dev

The header is the ONLY bridge to the C. It is generated rather than written so no
human ever retypes a weight, and `firmware/test/test_parity.c` closes the loop by
running the same audio through both sides.

Float, not fixed-point, deliberately (PLAN §13 G6b): the M4F has a hardware FPU and
mfcc uses 3.2% of it, so float runs comfortably. Quantising at the same time as
integrating would confound a port bug with fixed-point drift — do the integration
first, measure it working, then quantise as an isolated change with a before/after.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from birdgate import adapt as A, data as D, export as E, frontends as FE, metrics as M
from birdgate import provenance as P, svm as S
from birdgate.config import (SAMPLE_RATE, FrontEndParams, GateConfig, GateParams,
                             WindowParams)
from scripts import train as T


def main(arm: str = "mfcc-dev", target: float = 0.98, seed: int = 0, out=None):
    wp = WindowParams()
    windows, y, X = T.build_xy(arm, wp)
    names = FE.build(FrontEndParams(kind=arm), SAMPLE_RATE).feature_names
    tr = windows.index[windows["split"] == "train"].to_numpy()

    model = S.fit(X[tr], y[tr], arm, names, seed=seed)

    # Refit global_scale here rather than reading it back out of `T.scores`, so the
    # constant written into the header is demonstrably the one used to pick theta.
    raw = model.score(X)
    adapt = replace(T.ADAPT, global_scale=float(raw[tr].std()) or 1.0)
    z = A.normalise_by_recording(windows, raw, adapt)

    theta, _ = T.pick_theta(model, windows, y, X, target, z=z, gate=T.GATE)

    va = windows.index[windows["split"] == "val"].to_numpy()
    val = windows[windows["split"] == "val"].reset_index(drop=True)
    gp = GateParams(theta_on=theta, theta_off=None, **T.GATE)
    m = M.evaluate(val, z[va], gp)
    ps = M.per_species_retention(val, m["_regions"], n_boot=500, seed=seed)
    floor = M.retention_floor(ps, 20, 0.10)

    cfg = GateConfig(name=f"g5-export-{arm}", frontend=FrontEndParams(kind=arm),
                     gate=gp, adapt=adapt, seed=seed)
    prov = {
        "arm": arm, "config_hash": P.manifest(cfg)["config_hash"], "split_seed": seed,
        "n_features": len(model.w), "theta_on": f"{theta:.6f}",
        "val_retention": f"{100 * m['birdsong_retention']:.2f}%",
        "val_saving": f"{100 * m['saving']:.2f}%",
        "val_species_floor": f"{100 * floor:.2f}%",
        "selected_on": f"val, lowest duty meeting {100 * target:.0f}% retention",
    }
    out = out or D.PROJECT / "firmware" / "arms" / arm / "model_params.h"
    E.export_header(model, theta, adapt, T.GATE, wp.win_s, out_path=out, provenance=prov)

    print(f"wrote {out}")
    for k, v in prov.items():
        print(f"  {k:18s} {v}")
    P.write(cfg, metrics={k: v for k, v in prov.items()},
            notes=f"Float export for {arm}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mfcc-dev")
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.arm, a.target, a.seed)
