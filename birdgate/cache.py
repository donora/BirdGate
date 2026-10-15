"""Precompute per-recording features to `cache/`, streaming off the USB drive.

Raw FLAC is cold and can live on an external drive; derived features are hot and
live on the SSD. One array per **recording**, not per window, so the decision window
and hop stay free parameters.

Cached at the **finest** hop we will ever want (`train_hop_s`), so evaluation at a
coarser hop is a stride into the same array rather than a recompute.

⚠️ `soundfile` cannot decode these FLACs — libsndfile 1.2.2 fails with "flac
decoder lost sync" at frame 0, though it reads the header fine. librosa falls back
to `audioread`, which turns out to be fast here (~1300x realtime), so this is a
non-issue in practice. Noted because the failure is silent-ish (a warning) and
would otherwise look like a corrupt drive.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import data as D
from . import frontends as FE
from .config import FrontEndParams, WindowParams

ARMS = ("carihc", "bands", "mfcc", "mfcc-dev", "logmel", "logmel-dev")


def arm_dir(kind: str) -> Path:
    return D.CACHE / kind


def feat_path(kind: str, filename: str) -> Path:
    return arm_dir(kind) / (Path(filename).stem + ".npy")


def build(arms=ARMS, wp: WindowParams = WindowParams(), force: bool = False,
          sr: int = 32_000, verbose: bool = True) -> dict:
    """Compute and write features for every recording, for every arm.

    Audio is decoded **once per recording** and shared across arms — decoding is
    ~3 s and the filterbanks are ~20 s each, so re-reading per arm would waste
    little, but holding one decode also keeps the USB read pattern sequential.
    """
    import librosa

    D.assert_audio_available()
    recs = D.load_recordings()
    win = int(round(wp.win_s * sr))
    hop = int(round(wp.train_hop_s * sr))

    fes = {k: FE.build(FrontEndParams(kind=k), sr) for k in arms}
    for k in arms:
        arm_dir(k).mkdir(parents=True, exist_ok=True)

    todo = [r for r in recs.itertuples()
            if force or any(not feat_path(k, r.filename).exists() for k in arms)]
    if verbose:
        print(f"{len(todo)}/{len(recs)} recordings to process, arms={list(arms)}")

    t_start = time.time()
    for i, rec in enumerate(todo, 1):
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # the soundfile->audioread fallback
            y, _ = librosa.load(D.audio_path(rec.filename), sr=sr, mono=True)
        parts = [f"load {time.time()-t0:.0f}s"]
        for k in arms:
            out = feat_path(k, rec.filename)
            if out.exists() and not force:
                parts.append(f"{k} cached")
                continue
            t1 = time.time()
            arr = (FE.logmel(y, sr, dev=k.endswith("-dev")) if k.startswith("logmel")
                   else FE.raw_templates(y, fes[k], win, hop))
            tmp = out.with_suffix(".npy.tmp")         # atomic: no half-written cache
            with open(tmp, "wb") as fh:
                np.save(fh, arr.astype(np.float32))
            tmp.replace(out)
            parts.append(f"{k} {time.time()-t1:.0f}s{arr.shape}")
        if verbose:
            el = time.time() - t_start
            print(f"[{i:2d}/{len(todo)}] {rec.filename}  " + " | ".join(parts)
                  + f"  (eta {el/i*(len(todo)-i)/60:.0f} min)", flush=True)

    meta = {
        "arms": list(arms),
        "window": asdict(wp),
        "sr": sr,
        "win_samples": win,
        "hop_samples": hop,
        "frontend_params": {k: asdict(FrontEndParams(kind=k)) for k in arms},
        "n_features": {k: fes[k].n_features for k in arms},
    }
    (D.CACHE / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def load_meta() -> dict:
    p = D.CACHE / "meta.json"
    if not p.exists():
        raise FileNotFoundError(
            "No feature cache. Build it with:\n"
            "    ../.venv/bin/python -m scripts.build_cache"
        )
    return json.loads(p.read_text())


def assemble(windows, kind: str, wp: WindowParams = WindowParams()):
    """Features aligned row-for-row with a `windows` table, for one arm.

    The cache is stored at `train_hop_s`; a window starting at `t` seconds is row
    `t / train_hop_s`. Eval windows use a coarser hop, so they simply stride. This
    is the one place an off-by-one would poison every number downstream without
    raising, so the index is computed from `start_s` and asserted integral.
    """
    meta = load_meta()
    hop_s = meta["window"]["train_hop_s"]
    out = None
    for f, g in windows.groupby("filename", sort=False):
        arr = np.load(feat_path(kind, f), mmap_mode="r")
        idx_f = g["start_s"].to_numpy() / hop_s
        idx = np.rint(idx_f).astype(int)
        assert np.abs(idx_f - idx).max() < 1e-6, f"{f}: window start not on the cache grid"
        assert idx.max() < len(arr), (
            f"{f}: window {idx.max()} beyond cached {len(arr)} rows — cache is stale")
        if out is None:
            out = np.zeros((len(windows), arr.shape[1]), dtype=np.float32)
        out[g.index.to_numpy()] = arr[idx]
    return out
