"""The C chain and the Python reference must agree: features to tolerance, the gate
mask exactly. If they drift, every number measured in Python is a claim about a
machine that does not exist.

Builds `firmware/test/test_parity.c` from the SAME sources the device build uses,
runs audio through it and through `frontends` -> `svm` -> `adapt` -> `gate`, and
compares. Synthetic audio always; the busiest real recording too when the drive is
mounted. Skips when there is no C compiler.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from birdgate import adapt as A, data as D, frontends as FE, gate as G
from birdgate.config import FrontEndParams, GateParams

FW = D.PROJECT / "firmware"
COMMON = ["test/test_parity.c", "common/fft.c", "common/adapt.c", "common/gate.c",
          "common/birdgate.c", "common/bg_stream.c"]
# Each arm contributes its front-end and its scorer; everything else is shared. An arm
# with no generated header has not been exported yet and its cases skip.
ARM_SOURCES = {
    "mfcc-dev": ["arms/mfcc-dev/mfcc.c", "arms/mfcc-dev/scorer.c"],
    "cnn": ["arms/cnn/logmel.c", "arms/cnn/cnn.c"],
}
SR, WIN = 32_000, 32_000


def header_for(arm: str) -> Path:
    return FW / "arms" / arm / "model_params.h"


HEADER = header_for("mfcc-dev")


def parse_header(text: str) -> dict:
    h = {k: float(v) for k, v in re.findall(r"#define (BG_\w+)\s+(-?[\d.]+(?:e[+-]?\d+)?)f?\b", text)}
    for name in ("BG_W", "BG_ACT_SCALE", "BG_ACT_OFFSET", "BG_LIN_W"):
        m = re.search(rf"{name}\[\d+\] = \{{(.*?)\}};", text, re.S)
        if m:
            h[name] = np.array([float(v.rstrip("f"))
                                for v in re.findall(r"-?[\d.]+(?:e[+-]?\d+)?f", m.group(1))])
    h["W"] = h.get("BG_W", h.get("BG_LIN_W"))     # the linear arm's weights, or the CNN's head
    return h


def build_harness(arm: str, dest: Path) -> Path:
    """Compile the harness from the SAME sources the device build uses, for one arm."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    if not header_for(arm).exists():
        pytest.skip(f"{arm} has no generated model_params.h yet")
    out = dest / f"parity-{arm}"
    cmd = [cc, "-O2", "-Wall", "-Wextra", "-std=c99", f"-I{FW/'common'}", f"-I{FW/'arms'/arm}",
           *[str(FW / s) for s in COMMON + ARM_SOURCES[arm]], "-lm", "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "warning" not in r.stderr, r.stderr
    return out


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    return build_harness("mfcc-dev", tmp_path_factory.mktemp("parity"))


@pytest.fixture(scope="module")
def cnn_harness(tmp_path_factory):
    return build_harness("cnn", tmp_path_factory.mktemp("parity-cnn"))


def run_python_cnn(x: np.ndarray, h: dict) -> dict:
    """The CNN arm's reference chain: causal log-mel, the streamed quantised network,
    then the same adaptive threshold and gate every arm shares."""
    import torch
    from birdgate import cnn as CN, frontends as FE
    ckpt = D.RUNS / "g3-cnn-logmel-dev" / "cnn-int8.pt"
    if not ckpt.exists():
        pytest.skip(f"{ckpt} missing — train with --save-model")
    qg = torch.load(ckpt, weights_only=False)

    lm = FE.logmel(x, SR, n_mels=int(h["BG_N_MELS"]), dev=True)
    fpw = int(h["BG_FRAMES_PER_WINDOW"])
    per_win = WIN // int(h["BG_HOP_SAMPLES"])            # frames advanced per window
    n_win = 1 + (lm.shape[1] - fpw) // per_win if lm.shape[1] >= fpw else 0
    n_win = min(n_win, len(x) // WIN)
    # The arm tracks the acquisition level, so the centring is part of what has to
    # agree: a drifting state would show up here as a slow divergence rather than a
    # constant offset, which is the failure that would be easiest to miss.
    centres = CN.level_centres(lm, float(h["BG_IN_LEVEL_TAU_S"]), fpw)
    patches = np.stack([lm[:, w * per_win:w * per_win + fpw] - centres[w * per_win]
                        + np.float32(qg.norm.mean) for w in range(n_win)])[:, None]
    score = CN.stream_forward_quant(qg, patches.astype(np.float32))

    ap = A.AdaptParams(mode="quantile", window_s=h["BG_ADAPT_WINDOW_S"],
                       bg_quantile=h["BG_ADAPT_QUANTILE"], scale="global",
                       global_scale=h["BG_GLOBAL_SCALE"], min_scale=h["BG_MIN_SCALE"],
                       agg="max", agg_n=int(h["BG_AGG_N"]))
    z_inst = A.normalise(score, ap, h["BG_WIN_S"])
    z = A.aggregate(z_inst, ap)
    gp = GateParams(theta_on=h["BG_THETA_ON"], theta_off=h["BG_THETA_OFF"],
                    pre_roll_s=h["BG_PRE_ROLL_S"], hangover_s=h["BG_HANGOVER_S"],
                    min_on_s=h["BG_MIN_ON_S"], recheck_period_s=h["BG_RECHECK_S"])
    return {"score": score, "z_inst": z_inst, "z": z,
            "mask": G.run(z, gp, h["BG_WIN_S"]),
            "detect": z_inst > h["BG_THETA_ON"]}


def compare_cnn(c: np.ndarray, py: dict, h: dict, label: str):
    """The CNN arm cannot meet the linear arm's bar, and the reason is structural.

    Quantisation turns arbitrarily small differences into whole-step jumps: the C and
    PyTorch sum a convolution in different orders, differ in the seventh digit, and a
    value sitting near a rounding boundary then lands on a different code. The linear
    arm has no such amplifier — it agrees to 3e-6 with 79x headroom over its closest
    window to theta. Here the closest window is *inside* the numerical error.

    So the bar is: the gate masks — the thing that decides what reaches the card — must
    be identical, and **every** per-window disagreement must be explained by the window
    sitting within numerical error of theta. That second clause is what separates
    boundary noise from an implementation bug: a real bug would move windows that are
    nowhere near the threshold.
    """
    nf = int(h["BG_N_FEATURES"])
    n = len(py["score"])
    c = c[:n]
    mask_c = c[:, nf + 3] > 0.5
    for i in np.flatnonzero(c[:, nf + 4] > 0.5):
        mask_c[i - int(c[i, nf + 4]):i] = True
    detect_c = c[:, nf + 1] > h["BG_THETA_ON"]

    zd = np.abs(c[:, nf + 1] - py["z_inst"])
    margin = np.abs(py["z_inst"] - h["BG_THETA_ON"])
    flips = detect_c != py["detect"]
    print(f"\n[cnn {label}] windows={n} recorded(py)={int(py['mask'].sum())} "
          f"recorded(C)={int(mask_c.sum())} detected={int(py['detect'].sum())} "
          f"z max|diff|={zd.max():.2e} median={np.median(zd):.2e} "
          f"closest-to-theta={margin.min():.2e} detection flips={int(flips.sum())}")

    assert np.allclose(c[:, nf], py["score"], atol=0.25), np.abs(c[:, nf] - py["score"]).max()
    assert zd.max() < 0.05, zd.max()
    # Every flipped decision must be a window that was on the threshold anyway.
    assert (margin[flips] <= zd.max()).all(), "a detection flipped away from the boundary"
    assert flips.mean() < 0.02, f"{100 * flips.mean():.1f}% of detections flipped"
    assert np.array_equal(mask_c, py["mask"]), "gate masks differ"


def test_cnn_synthetic_parity(cnn_harness, tmp_path):
    h = parse_header(header_for("cnn").read_text())
    x = synthetic_audio()
    c = run_c(cnn_harness, x, tmp_path, arm="cnn")
    compare_cnn(c, run_python_cnn(x, h), h, "synthetic")
    assert np.array_equal(run_c(cnn_harness, x, tmp_path, i16=True, arm="cnn"), c), "int16 ring path differs"


def test_cnn_real_audio_parity(cnn_harness, tmp_path):
    if not D.AUDIO_DIR.resolve().exists():
        pytest.skip("audio drive not mounted")
    import librosa
    ann = D.load_annotations()
    busiest = ann.groupby("filename")["duration_s"].sum().idxmax()
    x, _ = librosa.load(D.audio_path(busiest), sr=SR, mono=True, duration=900)
    x = (np.round(x * 32768) / 32768).astype(np.float32)
    h = parse_header(header_for("cnn").read_text())
    py = run_python_cnn(x, h)
    compare_cnn(run_c(cnn_harness, x, tmp_path, arm="cnn"), py, h, busiest)
    assert py["mask"].any(), "parity on audio where the gate never fires proves nothing"


def run_c(harness: Path, x: np.ndarray, tmp: Path, i16: bool = False,
          freeze: int = 0, arm: str = "mfcc-dev") -> np.ndarray:
    """Rows of [features, score, z_inst, z, open, backfill]. The feature count is the
    arm's own, so the reader cannot silently mis-shape another arm's output."""
    fin = tmp / "in.f32"
    fout = tmp / f"out{'_i16' if i16 else ''}{f'_f{freeze}' if freeze else ''}.f32"
    x.astype(np.float32).tofile(fin)
    args = [str(harness), str(fin), str(fout)] + (["--i16"] if i16 else [])
    if freeze:
        args += ["--freeze", str(freeze)]
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    nf = int(parse_header(header_for(arm).read_text())["BG_N_FEATURES"])
    return np.fromfile(fout, dtype=np.float32).reshape(-1, nf + 5)


def run_python(x: np.ndarray, h: dict) -> dict:
    fe = FE.build(FrontEndParams(kind="mfcc-dev"), SR)
    X = FE.normalise(FE.raw_templates(x, fe, WIN, WIN), FrontEndParams(kind="mfcc-dev"))
    score = X.astype(np.float64) @ h["W"] + h["BG_BIAS"]
    ap = A.AdaptParams(mode="quantile", window_s=h["BG_ADAPT_WINDOW_S"], bg_quantile=h["BG_ADAPT_QUANTILE"],
                       scale="global", global_scale=h["BG_GLOBAL_SCALE"], min_scale=h["BG_MIN_SCALE"],
                       agg="max", agg_n=int(h["BG_AGG_N"]))
    z_inst = A.normalise(score, ap, h["BG_WIN_S"])        # this window alone
    z = A.aggregate(z_inst, ap)                           # max over the last agg_n
    gp = GateParams(theta_on=h["BG_THETA_ON"], theta_off=h["BG_THETA_OFF"], pre_roll_s=h["BG_PRE_ROLL_S"],
                    hangover_s=h["BG_HANGOVER_S"], min_on_s=h["BG_MIN_ON_S"],
                    recheck_period_s=h["BG_RECHECK_S"])
    return {"feat": X, "score": score, "z_inst": z_inst, "z": z,
            "mask": G.run(z, gp, h["BG_WIN_S"]),
            # What the BG_LED_RAW=1 firmware lights the green LED on.
            "detect": z_inst > h["BG_THETA_ON"]}


def synthetic_audio(seconds: int = 240, seed: int = 0) -> np.ndarray:
    """Coloured noise with two stretches of modulated chirps, on the int16 grid so the
    float and int16 paths see identical samples."""
    rng = np.random.default_rng(seed)
    n = seconds * SR
    t = np.arange(n) / SR
    noise = np.cumsum(rng.standard_normal(n)) ; noise -= np.convolve(noise, np.ones(64) / 64, "same")
    x = 0.01 * noise / (np.abs(noise).max() + 1e-9)
    for (a, b) in [(60, 95), (150, 175)]:
        seg = (t >= a) & (t < b)
        f = 2500 + 4000 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t[seg]))
        am = 0.5 + 0.5 * np.sin(2 * np.pi * 9.0 * t[seg]) ** 8
        x[seg] += 0.3 * am * np.sin(2 * np.pi * np.cumsum(f) / SR)
    x = np.clip(x, -1, 1)
    return (np.round(x * 32768) / 32768).astype(np.float32)


def compare(c: np.ndarray, py: dict, h: dict, label: str):
    n = len(py["mask"])
    assert c.shape[0] == n, (c.shape, n)
    nf = len(h["W"])
    feat_diff = np.abs(c[:, :nf] - py["feat"]).max()
    z_diff = np.abs(c[:, nf + 2] - py["z"]).max()
    # The C gate reports its live state and, on opening, how many earlier windows to
    # recover from the ring buffer. The recorded mask is the union of the two — the
    # device rewinds its read cursor; gate.py writes rec[max(0, i - pre):i] = True.
    mask_c = c[:, nf + 3] > 0.5
    for i in np.flatnonzero(c[:, nf + 4] > 0.5):
        mask_c[i - int(c[i, nf + 4]):i] = True
    detect_c = c[:, nf + 1] > h["BG_THETA_ON"]
    margin = np.abs(py["z"] - h["BG_THETA_ON"]).min()
    print(f"\n[{label}] windows={n} recorded(py)={int(py['mask'].sum())} recorded(C)={int(mask_c.sum())} "
          f"detected={int(py['detect'].sum())} feat max|diff|={feat_diff:.2e} "
          f"z max|diff|={z_diff:.2e} closest-to-theta={margin:.2e}")
    assert np.allclose(c[:, :nf], py["feat"], rtol=1e-4, atol=1e-2), feat_diff
    assert np.allclose(c[:, nf], py["score"], atol=1e-3)
    assert np.allclose(c[:, nf + 1], py["z_inst"], atol=1e-3)
    assert np.allclose(c[:, nf + 2], py["z"], atol=1e-3), z_diff
    assert np.array_equal(mask_c, py["mask"]), "gate masks differ"
    # The detector build's LED signal, pinned in its own right: it is what a person
    # will read the model's behaviour off, so it cannot be merely nearly right.
    assert np.array_equal(detect_c, py["detect"]), "per-window detections differ"
    return margin, z_diff


def test_synthetic_parity(harness, tmp_path):
    h = parse_header(HEADER.read_text())
    x = synthetic_audio()
    c = run_c(harness, x, tmp_path)
    py = run_python(x, h)
    compare(c, py, h, "synthetic")
    # The device path: int16 ring, slot-sized arrivals, wrap handling. Same samples,
    # so it must reproduce the float path exactly.
    c16 = run_c(harness, x, tmp_path, i16=True)
    assert np.array_equal(c16, c), "int16 ring path differs from the float path"


def test_real_audio_parity(harness, tmp_path):
    if not D.AUDIO_DIR.resolve().exists():
        pytest.skip("audio drive not mounted")
    import librosa
    ann = D.load_annotations()
    busiest = ann.groupby("filename")["duration_s"].sum().idxmax()
    x, _ = librosa.load(D.audio_path(busiest), sr=SR, mono=True, duration=900)
    x = (np.round(x * 32768) / 32768).astype(np.float32)
    h = parse_header(HEADER.read_text())
    c = run_c(harness, x, tmp_path)
    py = run_python(x, h)
    margin, z_diff = compare(c, py, h, busiest)
    assert py["mask"].any(), "parity on audio where the gate never fires proves nothing"
    assert margin > 10 * z_diff, "a window sits within numerical error of theta"


def test_frozen_background_stops_moving(harness, tmp_path):
    """`bg_adapt_set_freeze` is a bench aid: after N windows the background is held, so
    a sound can be tried repeatedly against a threshold that does not drift.

    Note the freeze point has to sit past the ring's warm-up. The ring is pre-filled
    with BG_ADAPT_N copies of the first score, and the anchor is the 0.2 quantile, so
    it cannot move at all until more than a fifth of those copies have been overwritten
    — 60 windows here, and it is not fully clear of the pre-fill until 300.
    """
    h = parse_header(HEADER.read_text())
    nf = len(h["W"])
    n_adapt = int(h["BG_ADAPT_N"])
    freeze = n_adapt - 20                      # past the point the anchor starts moving
    scale = max(h["BG_GLOBAL_SCALE"], h["BG_MIN_SCALE"])
    x = synthetic_audio(seconds=freeze + 120)

    def anchors(c):
        return c[:, nf] - c[:, nf + 1] * scale      # score - z*scale == the anchor used

    frozen = anchors(run_c(harness, x, tmp_path, freeze=freeze))
    free = anchors(run_c(harness, x, tmp_path))

    assert np.ptp(frozen[freeze:]) < 1e-5, f"still moving after freeze: {np.ptp(frozen[freeze:]):.2e}"
    assert np.allclose(frozen[:freeze], free[:freeze], atol=1e-6), "freeze changed the warm-up"
    # Guard against a freeze that passes only because nothing was moving anyway.
    assert np.ptp(free[freeze:]) > 1e-4, "the unfrozen anchor never moved; test proves nothing"
