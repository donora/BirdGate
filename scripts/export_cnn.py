"""Emit `firmware/arms/cnn/model_params.h` from the persisted CNN arm.

    ../.venv/bin/python -m scripts.export_cnn [--seed 0] [--target 0.98]

Deliberately does not train: `scripts.train_cnn --save-model` persists the deployable
int8 arm, and this reads it. The linear arm's exporter retrains because a convex fit
is seconds; this one is 45 minutes, and an export that silently refits is an export
whose weights you cannot be sure match the numbers you read.

What ships is the **quantised** arm. `cnn.quantise` rounds the folded weights onto the
int8 grid and fits one scale per activation tensor; the firmware stores activations as
one byte each and reads them back as `(code - offset) * scale`, so the device runs that
grid exactly rather than an approximation of it. See ROADMAP §4.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from birdgate import adapt as A, cnn as CN, data as D, export as E, metrics as M
from birdgate import provenance as P, windows as W
from birdgate.config import FrontEndParams, GateConfig, GateParams, WindowParams
from scripts import train as T
from scripts.train_cnn import load_model


def main(frontend: str = "logmel-dev", target: float = 0.98, seed: int = 0, out=None):
    wp = WindowParams()
    cfg = CN.CnnParams(seed=seed, frontend=frontend)
    gm = load_model(cfg)

    windows, y = W.build_windows(p=wp)
    windows = windows.reset_index(drop=True)
    tr = windows.index[windows["split"] == "train"].to_numpy()

    raw = gm.score_all(windows, wp)
    # Refit global_scale here rather than trusting a cached one, so the constant in
    # the header is demonstrably the one theta was chosen against.
    adapt = replace(T.ADAPT, global_scale=float(raw[tr].std()) or 1.0)
    z = A.normalise_by_recording(windows, raw, adapt)
    theta, _ = T.pick_theta(None, windows, y, None, target, z=z, gate=T.GATE)

    va = windows.index[windows["split"] == "val"].to_numpy()
    val = windows[windows["split"] == "val"].reset_index(drop=True)
    gp = GateParams(theta_on=theta, theta_off=None, **T.GATE)
    m = M.evaluate(val, z[va], gp)
    ps = M.per_species_retention(val, m["_regions"], n_boot=500, seed=seed)
    floor = M.retention_floor(ps, 20, 0.10)

    cfg_g = GateConfig(name=f"g5-export-cnn-{frontend}",
                       frontend=FrontEndParams(kind=frontend), gate=gp,
                       adapt=adapt, seed=seed)
    prov = {
        "arm": "cnn-int8", "frontend": frontend,
        "config_hash": P.manifest(cfg_g)["config_hash"], "split_seed": seed,
        "n_features": int(CN.CnnParams().channels[-1]),
        "theta_on": f"{theta:.6f}",
        "val_retention": f"{100 * m['birdsong_retention']:.2f}%",
        "val_saving": f"{100 * m['saving']:.2f}%",
        "val_species_floor": f"{100 * floor:.2f}%",
        "selected_on": f"val, lowest duty meeting {100 * target:.0f}% retention",
    }
    out = out or D.PROJECT / "firmware" / "arms" / "cnn" / "model_params.h"
    E.export_cnn_header(gm, theta, adapt, T.GATE, wp.win_s, out_path=out, provenance=prov)

    print(f"wrote {out}")
    for k, v in prov.items():
        print(f"  {k:18s} {v}")
    P.write(cfg_g, metrics=dict(prov), notes=f"CNN export for {frontend}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="logmel-dev")
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    main(frontend=ap.parse_args().frontend, target=ap.parse_args().target,
         seed=ap.parse_args().seed)
