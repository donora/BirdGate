"""Build `demo/`: the few MB that let a fresh clone run the app without the audio.

The full pipeline needs the 3.2 GB of Sierra Nevada FLACs and a 3 GB feature cache.
Everything the app *shows* after the model, though — thresholds, the gate, every
metric — acts on one raw score per window. So this freezes those scores, exactly
as `scripts.train_cnn --save-scores` already does for the CNN:

    demo/
    ├── scores.npz     the window table + raw scores for every arm (+ linear weights)
    ├── maps/          linker-map snapshots of the device builds, for the silicon tab
    └── clips/         eight 30 s excerpts of the test recordings, for the players

Run after any retrain or rebuild, with the cache and the firmware builds present:

    python -m scripts.build_demo           # scores and maps
    python -m scripts.build_demo --clips   # and re-cut the clips (needs the audio)

The clips are chosen from the data, not by hand, each to show one thing: the
densest dawn chorus, the busiest evening, true silence, the silence that scores
highest (what fools the gate), a lone call the model nearly misses, and then
whatever adds the most common species the rest do not cover.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from birdgate import budget as BG
from birdgate import data as D
from birdgate import frontends as FE
from birdgate import svm as S
from birdgate.config import FrontEndParams, WindowParams
from scripts.train import build_xy

LINEAR = ("carihc", "bands", "mfcc", "mfcc-dev")
SEED = 0
CLIP_S = 30
N_CLIPS = 8


def _segments(scores: dict) -> pd.DataFrame:
    """Every 30 s segment of the test recordings, on a 5 s grid, with what is in it."""
    ann = D.load_annotations()
    top = set(ann["label"].value_counts().head(25).index)
    w = pd.DataFrame({k: scores[k] for k in ("filename", "start_s", "split")})
    w["score"] = scores["raw_mfcc"]
    w = w[w["split"] == "test"]
    rows = []
    for f, g in w.groupby("filename"):
        a = ann[ann["filename"] == f]
        dur = float(g["start_s"].max()) + 1
        block = D._row_from_filename(f, 32_000, dur)["block"]
        for t0 in np.arange(0, dur - CLIP_S + 1, 5.0):
            t1 = t0 + CLIP_S
            inside = a[(a["start_s"] >= t0) & (a["end_s"] <= t1)]
            touching = a[(a["end_s"] > t0) & (a["start_s"] < t1)]
            # Annotators sometimes leave a stretch unlabelled (SNE_016 has clear calls
            # before its first box at 43 s), so "no birds" means no box within 30 s
            # either side and not in the first minute — not merely an empty segment.
            near = a[(a["end_s"] > t0 - CLIP_S) & (a["start_s"] < t1 + CLIP_S)]
            sc = g.loc[(g["start_s"] >= t0) & (g["start_s"] < t1), "score"]
            rows.append({
                "filename": f, "start_s": t0, "block": block,
                "n_touching": len(touching), "n_species": touching["code"].nunique(),
                "species": frozenset(inside["label"]) & top,
                "quiet": len(near) == 0 and t0 >= 60,
                "short_call": len(touching) == 1 and len(inside) == 1
                              and float(inside["duration_s"].iloc[0]) < 3.0,
                "score_mean": float(sc.mean()), "score_max": float(sc.max()),
            })
    return pd.DataFrame(rows)


def pick_clips(scores: dict) -> pd.DataFrame:
    """The eight segments, by rule — see the module docstring."""
    seg = _segments(scores)
    picks = []

    def take(df, by, ascending, why):
        df = df.sort_values(by, ascending=ascending)
        for _, r in df.iterrows():
            # never overlap a segment already taken
            if all(r["filename"] != p["filename"] or abs(r["start_s"] - p["start_s"]) >= CLIP_S
                   for p in picks):
                picks.append({**r.to_dict(), "why": why})
                return

    empty = seg[seg["quiet"]]
    take(seg[seg["block"] == "dawn"], "n_species", False, "dawn chorus, the most species at once")
    take(seg[seg["block"] == "evening"], "n_touching", False, "the busiest evening stretch")
    take(empty, "score_max", True, "true silence: no birds, and the lowest scores")
    take(empty, "score_max", False, "no birds, but the highest score: what fools the gate")
    take(seg[seg["short_call"]], "score_max", True, "one short call the model nearly misses")
    while len(picks) < N_CLIPS:
        have = frozenset().union(*(p["species"] for p in picks))
        gain = seg["species"].map(lambda sp: len(sp - have))
        take(seg.assign(gain=gain), "gain", False, "covers common species the others miss")
    return pd.DataFrame(picks)


def cut_clips(scores: dict) -> None:
    import librosa
    import soundfile as sf
    D.assert_audio_available()
    out = D.DEMO / "clips"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.flac"):
        f.unlink()
    picks = pick_clips(scores)
    rows = []
    for _, r in picks.iterrows():
        name = f"{Path(r['filename']).stem}_{int(r['start_s']):04d}.flac"
        y, _ = librosa.load(str(D.audio_path(r["filename"])), sr=32_000,
                            offset=float(r["start_s"]), duration=CLIP_S, mono=True)
        sf.write(out / name, y, 32_000, subtype="PCM_16")
        rows.append({"clip": name, "filename": r["filename"], "start_s": r["start_s"],
                     "end_s": r["start_s"] + CLIP_S, "why": r["why"]})
        print(f"clip      {name}  {r['why']}")
    pd.DataFrame(rows).to_csv(out / "clips.csv", index=False)


def main() -> None:
    D.DEMO.mkdir(exist_ok=True)
    wp = WindowParams()
    cnn = D.RUNS / "g3-cnn" / "scores.npz"
    with np.load(cnn, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}

    for arm in LINEAR:
        windows, y, X = build_xy(arm, wp)
        assert np.array_equal(windows["filename"].to_numpy().astype("U"), out["filename"]), \
            f"{arm}: window table differs from the CNN's — rebuild the CNN scores first"
        fe = FE.build(FrontEndParams(kind=arm), 32_000)
        tr = windows.index[windows["split"] == "train"].to_numpy()
        model = S.fit(X[tr], y[tr], arm, fe.feature_names, seed=SEED)
        out[f"raw_{arm}"] = model.score(X).astype(np.float32)
        out[f"w_{arm}"] = np.asarray(model.w, dtype=np.float32)
        out[f"b_{arm}"] = np.float32(model.b)
        print(f"{arm:9s} scored {len(windows):,} windows")
    out["seed"] = np.int64(SEED)
    np.savez_compressed(D.DEMO / "scores.npz", **out)

    maps = D.DEMO / "maps"
    maps.mkdir(exist_ok=True)
    for arm, path in BG.arm_builds().items():
        # The parser keeps only the object name after the last "/", so the build
        # paths carry nothing it needs — and they name the builder's home directory.
        text = path.read_text().replace(str(Path.home()), "~")
        (maps / f"{arm}.map").write_text(text)
        print(f"map       {arm}")

    if "--clips" in sys.argv:
        cut_clips(out)

    size = sum(f.stat().st_size for f in D.DEMO.rglob("*") if f.is_file())
    print(f"demo/ is {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
