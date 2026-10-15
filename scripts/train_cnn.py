"""Train the tiny-CNN arm and score it through the same gate as every other arm.

    ../.venv/bin/python -m scripts.train_cnn
    ../.venv/bin/python -m scripts.train_cnn --epochs 20 --no-augment

This is the arm that tests the data-scarcity premise for a tiny model (see
`birdgate/cnn.py`). Everything outside the model is imported from `scripts.train`
— the same windows, the same recording split, the same adaptive threshold, the
same theta-selection rule on val, and the same `report()` — so a difference in the
final table is attributable to the classifier and nothing else.

Two numbers are reported, and the **int8 one is the result**: a float CNN is not a
comparison arm, because it cannot be deployed on the target.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
from dataclasses import asdict, replace

import numpy as np

from birdgate import adapt as A, cnn as CN, data as D, provenance as P, windows as W
from birdgate.config import FrontEndParams, GateConfig, GateParams, WindowParams
from scripts.train import ADAPT, GATE, pick_theta, report


def _z(windows, raw, train_idx, adapt):
    """Raw logit -> background-normalised z, exactly as `scripts.train.scores` does.

    `global_scale` is fitted on TRAIN scores and ships with the model; the CNN's
    logits have a different natural scale from the SVM's margin, which is precisely
    why that constant is fitted per model rather than shared.
    """
    if adapt.scale == "global":
        adapt = replace(adapt, global_scale=float(raw[train_idx].std()) or 1.0)
    return A.normalise_by_recording(windows, raw, adapt)


SCORES = "scores.npz"
"""Where `--save-scores` writes the raw logits, under `runs/g3-cnn/`.

The app cannot train a network between slider moves, but it does not need to: what
it needs is the **raw** score for every window, because everything the compare tab
varies — background window, anchor quantile, pooling, hangover, minimum-on — is
applied *after* the model in `adapt.py` and `gate.py`. Cache the logits and the CNN
becomes just another arm in that tab, responding to every slider exactly as the
SVM arms do. Only the weights are frozen, and `params` records what produced them.
"""


def save_scores(windows, raw: dict, cfg: CN.CnnParams, path=None):
    """Write raw logits plus the keys needed to *verify* alignment, not assume it.

    `filename` and `start_s` travel with the scores so a consumer can assert the
    rows still line up with its own `build_windows()`. A silently misaligned score
    vector would show up as a plausible-looking but wrong lane in the app, which is
    the kind of bug that survives review.
    """
    import json
    from dataclasses import asdict
    path = path or (D.RUNS / "g3-cnn" / SCORES)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        filename=windows["filename"].to_numpy().astype("U"),
        start_s=windows["start_s"].to_numpy().astype("float32"),
        split=windows["split"].to_numpy().astype("U"),
        params=json.dumps({"cnn_params": asdict(cfg), "git_rev": P.git_rev(),
                           "arms": sorted(raw)}),
        **{f"raw_{k}": v.astype("float32") for k, v in raw.items()},
    )
    return path


def persist_model(gm, cfg: CN.CnnParams, path=None):
    """Save the deployable arm, so exporting the firmware header does not mean
    retraining. The quantised gate carries everything the exporter needs: the folded
    and grid-rounded weights, the activation scales, and the input normalisation."""
    import torch
    path = path or (D.RUNS / f"g3-cnn-{cfg.frontend}" / "cnn-int8.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gm, path)
    return path


def load_model(cfg: CN.CnnParams, path=None):
    import torch
    path = path or (D.RUNS / f"g3-cnn-{cfg.frontend}" / "cnn-int8.pt")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} — train it first:\n"
            f"    ../.venv/bin/python -m scripts.train_cnn "
            f"--frontend {cfg.frontend} --seed {cfg.seed} --save-model")
    return torch.load(path, weights_only=False)


def run(arm: str = "cnn", target: float = 0.98, seed: int = 0, gate: dict | None = None,
        adapt: A.AdaptParams | None = None, cfg: CN.CnnParams | None = None,
        verbose: bool = True, save: bool = False, save_model: bool = False) -> dict:
    wp = WindowParams()
    gate = GATE if gate is None else gate
    adapt = ADAPT if adapt is None else adapt
    cfg = CN.CnnParams(seed=seed) if cfg is None else cfg

    windows, y = W.build_windows(p=wp)
    windows = windows.reset_index(drop=True)
    tr = windows.index[windows["split"] == "train"].to_numpy()

    model = CN.fit(windows, y, cfg, wp, verbose=verbose)
    out, raws = {}, {}
    for tag, gm in _arms(model, windows, wp, tr, cfg):
        raw = raws[tag] = gm.score_all(windows, wp)
        z = _z(windows, raw, tr, adapt)
        theta, curve = pick_theta(None, windows, y, None, target, z=z, gate=gate)
        cost = CN.cost(cfg, model.model)
        out[tag] = {**report(windows, z, theta, gate, cost), "arm": tag,
                    "n_features": CN.N_MELS * CN.frames_per_window(wp, cfg.frontend),
                    "model_bytes": gm.n_bytes, "_curve": curve, "_gate": gm}
        if verbose:
            r = out[tag]
            print(f"  {tag:9s} retention {r['retention']:.1f}%  saving {r['saving']:.1f}%"
                  f"  worst {r['worst_rec']:.1f}%  {r['model_bytes']} B", flush=True)
    out["cnn"]["_model"] = model
    if save_model:
        path = persist_model(out["cnn-int8"]["_gate"], cfg)
        if verbose:
            print(f"  deployable arm -> {path}", flush=True)
    if save:
        path = save_scores(windows, raws, cfg)
        if verbose:
            print(f"  raw logits -> {path}", flush=True)
    return out


def _arms(model, windows, wp, tr, cfg):
    """The float arm, then the int8 arm calibrated on a TRAIN patch sample."""
    yield "cnn", model
    arrays, file_idx, f0, n = CN.patch_source(windows, wp, cfg.frontend)
    rng = np.random.default_rng(cfg.seed)
    rows = rng.choice(tr, size=min(2048, len(tr)), replace=False)
    # Centred exactly as inference will present them, or the activation scales are
    # calibrated against a distribution the device never sees.
    calib = CN.gather(arrays, file_idx, f0, n, rows,
                      CN.centre_source(arrays, cfg), model.norm.mean)
    yield "cnn-int8", CN.quantise(model, calib)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--epochs", type=int, default=CN.CnnParams.epochs)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--device", default=CN.CnnParams.device,
                    help="auto|cpu|mps — 'cpu' for a run whose number replicates")
    ap.add_argument("--frontend", default=CN.CnnParams.frontend,
                    help="logmel-dev (causal, what the firmware runs) or logmel")
    ap.add_argument("--save-model", action="store_true",
                    help="persist the deployable int8 arm for scripts.export_cnn")
    ap.add_argument("--save-scores", action="store_true",
                    help=f"write raw logits to runs/g3-cnn/{SCORES} so the app can "
                         "show this arm without retraining it")
    a = ap.parse_args()
    cfg = CN.CnnParams(epochs=a.epochs, seed=a.seed, spec_augment=not a.no_augment,
                       device=a.device, frontend=a.frontend)
    res = run(target=a.target, seed=a.seed, cfg=cfg, save=a.save_scores,
              save_model=a.save_model)
    for tag, r in res.items():
        print(tag, json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                               for k, v in r.items() if not k.startswith("_")}))
    P.write(GateConfig(name=f"g3-cnn-{a.frontend}", frontend=FrontEndParams(kind=a.frontend),
                       gate=GateParams(**GATE), adapt=ADAPT, seed=a.seed),
            metrics={tag: {k: v for k, v in r.items() if not k.startswith("_")}
                     for tag, r in res.items()},
            notes=json.dumps({"cnn_params": asdict(cfg),
                              "history": res["cnn"]["_model"].history}))
