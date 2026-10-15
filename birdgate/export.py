"""Emit `firmware/include/model_params.h` — the one bridge to the C side.

Everything the firmware needs, generated from the trained model so the two can
never drift: sample rate, window length, channel count, quantised CAR coefficients,
SVM weights and bias, the named threshold table, and the gate parameters.

Also emits a provenance block (config hash, split seed, val retention, duty cycle
at each named point) as a C comment, so a `.bin` recovered from a device six months
later can be traced back to the run that produced it. Firmware in the field with
unknown provenance is unusable data.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE
from .frontends import MEL_HOP, MEL_NFFT

N_MELS, N_MFCC, FMIN, FMAX = 128, 20, 150.0, 14_000.0
AMIN = 1e-10                      # power_to_db's floor; the C must use the same one


def mel_filterbank(sr: int = SAMPLE_RATE, n_mels: int = N_MELS):
    """librosa's slaney-normalised mel basis — the exact matrix the model was fit on."""
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=MEL_NFFT, n_mels=n_mels, fmin=FMIN, fmax=FMAX)


def dct_matrix(n_mels: int = N_MELS, n_mfcc: int = N_MFCC):
    """DCT-II orthonormal, as `scipy.fft.dct(type=2, norm="ortho")` applies it.

    Built by transforming the identity rather than writing the cosine formula out:
    the normalisation of DCT-II differs between references for k=0, and getting it
    wrong is a silent constant-offset error in the first coefficient — which is the
    loudest feature in the model.
    """
    import scipy.fft
    return scipy.fft.dct(np.eye(n_mels), type=2, norm="ortho", axis=0)[:n_mfcc]


def hann_window(n: int = MEL_NFFT):
    """Periodic Hann (`fftbins=True`), the STFT convention. NOT `np.hanning(n)`."""
    return np.hanning(n + 1)[:-1]


def _cf(v, fmt: str = ".9g") -> str:
    """A C float literal.

    `%g` of 1.0 is `"1"`, and `1f` is **not** a float constant in C — it is an integer
    with an invalid suffix, which the compiler rejects outright. Any formatted value
    with no `.`, `e` or `E` needs a decimal point putting back.
    """
    s = f"{v:{fmt}}"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def _floats(name: str, a, per_line: int = 6, indent: str = "    ") -> str:
    """C float array. `%.9g` round-trips float32 exactly; fewer digits silently drift."""
    a = np.asarray(a, dtype=np.float64).ravel()
    body, line = [], indent
    for i, v in enumerate(a):
        tok = _cf(v) + ","
        if len(line) + len(tok) > 96:
            body.append(line.rstrip())
            line = indent
        line += tok + " "
        if (i + 1) % per_line == 0:
            body.append(line.rstrip())
            line = indent
    if line.strip():
        body.append(line.rstrip())
    return f"static const float {name}[{a.size}] = {{\n" + "\n".join(body) + "\n};\n"


def _ints(name: str, a, ctype: str = "uint16_t") -> str:
    a = np.asarray(a).ravel()
    vals = ", ".join(str(int(v)) for v in a)
    wrapped, line = [], "    "
    for tok in vals.split(", "):
        if len(line) + len(tok) > 96:
            wrapped.append(line.rstrip())
            line = "    "
        line += tok + ", "
    if line.strip():
        wrapped.append(line.rstrip())
    return f"static const {ctype} {name}[{a.size}] = {{\n" + "\n".join(wrapped) + "\n};\n"


def export_header(model, theta: float, adapt, gate: dict, win_s: float = 1.0,
                  out_path=None, provenance: dict | None = None) -> str:
    """Emit `model_params.h`: every constant the device needs, from the trained model.

    The point of generating it is that the C and the Python cannot drift — there is no
    step where a human retypes a weight. `test_parity.c` closes the loop by running the
    same audio through both.

    The mel basis is stored **sparse**. Each band is a triangle touching ~7 of the 513
    FFT bins, so the dense 128x513 matrix is 263 KB of which 99.7% is zero; as
    (start, len, weights) it is 4.5 KB. The Hann window is stored as its first half plus
    the midpoint and mirrored in C, since a periodic Hann satisfies w[N-n] == w[n].
    """
    mel, dct, win = mel_filterbank(), dct_matrix(), hann_window()
    nz = mel > 0
    start = np.array([np.argmax(r) if r.any() else 0 for r in nz], dtype=np.int64)
    length = nz.sum(axis=1).astype(np.int64)
    weights = np.concatenate([mel[i, start[i]:start[i] + length[i]] for i in range(len(mel))])
    offset = np.concatenate([[0], np.cumsum(length)[:-1]])

    # A band whose weights are not contiguous would be silently truncated by the
    # (start, len) encoding. Mel triangles always are, but assert rather than assume.
    for i in range(len(mel)):
        assert nz[i, start[i]:start[i] + length[i]].all(), f"mel band {i} is not contiguous"

    win_samples = int(round(win_s * SAMPLE_RATE))
    fpw = 1 + (win_samples - MEL_NFFT) // MEL_HOP
    half = MEL_NFFT // 2
    theta_off = theta - gate["hysteresis_margin"]

    # The C port implements exactly one adaptive-threshold configuration — quantile
    # background, one global scale, max-of-N pooling — and every gate parameter the
    # Python state machine reads. Refuse to export anything else rather than let the
    # device silently run a different statistic from the one theta was calibrated on.
    from .config import GateParams
    gp = GateParams(theta_on=theta, **gate)
    if (adapt.mode, adapt.scale, adapt.agg) != ("quantile", "global", "max"):
        raise ValueError(f"firmware implements mode=quantile/scale=global/agg=max, "
                         f"got {adapt.mode}/{adapt.scale}/{adapt.agg}")
    adapt_n = int(round(adapt.window_s / win_s))
    if adapt_n < 2:
        raise ValueError("adaptive window must hold at least two scores")

    p = provenance or {}
    prov = "\n".join(f" *   {k:<22s} {v}" for k, v in p.items())
    out = f'''/* model_params.h — GENERATED by birdgate/export.py. DO NOT HAND-EDIT.
 *
 * Regenerate with:  ../.venv/bin/python -m scripts.export_model --arm mfcc-dev
 *
 * Provenance — so a card recovered from a device months later resolves to the run
 * that produced the firmware that wrote it:
 *
{prov}
 *   generated              {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
 */
#ifndef BIRDGATE_MODEL_PARAMS_H
#define BIRDGATE_MODEL_PARAMS_H
#include <stdint.h>

#define BG_SAMPLE_RATE        {SAMPLE_RATE}
#define BG_WIN_SAMPLES        {win_samples}
#define BG_FFT_SIZE           {MEL_NFFT}
#define BG_HOP_SAMPLES        {MEL_HOP}
#define BG_N_BINS             {MEL_NFFT // 2 + 1}
#define BG_N_MELS             {N_MELS}
#define BG_N_MFCC             {N_MFCC}

/* Frames per decision window. With center=False the last frame must END inside the
 * window: 1 + (32000 - 1024)/320 = {fpw}, NOT 32000/320 = {win_samples // MEL_HOP}. Using the
 * latter reads {win_samples // MEL_HOP - fpw} frames of the NEXT window's audio into this one. */
#define BG_FRAMES_PER_WINDOW  {fpw}
#define BG_N_FEATURES         {len(model.w)}   /* {N_MFCC} means then {N_MFCC} stds */
#define BG_MEL_NNZ            {len(weights)}
#define BG_LOG_FLOOR          {_cf(AMIN)}      /* power_to_db amin; clamp BEFORE log10 */

/* Periodic Hann, first half + midpoint. Mirror in C: w[n] = BG_HANN[n <= {half} ? n : {MEL_NFFT} - n] */
{_floats("BG_HANN", win[:half + 1])}
/* FFT twiddles for an in-place radix-2 complex DFT of size {MEL_NFFT}: for stage span s,
 * step = {MEL_NFFT}/(2s) and the k-th factor is (BG_TW_COS[k*step], BG_TW_SIN[k*step]).
 * Kept in FLASH rather than built at init with cosf/sinf, because this port is
 * SRAM-limited and flash-rich — see docs/lab-notebook.md §13 G6b. */
{_floats("BG_TW_COS", np.cos(-2.0 * np.pi * np.arange(MEL_NFFT // 2) / MEL_NFFT))}
{_floats("BG_TW_SIN", np.sin(-2.0 * np.pi * np.arange(MEL_NFFT // 2) / MEL_NFFT))}
/* Mel filterbank, sparse. Band m covers FFT bins [BG_MEL_START[m], +BG_MEL_LEN[m])
 * with weights at BG_MEL_W[BG_MEL_OFF[m] ...]. slaney-normalised, fmin={FMIN:.0f}, fmax={FMAX:.0f}. */
{_ints("BG_MEL_START", start)}{_ints("BG_MEL_LEN", length)}{_ints("BG_MEL_OFF", offset)}{_floats("BG_MEL_W", weights)}
/* DCT-II orthonormal, row-major [{N_MFCC}][{N_MELS}]: mfcc[k] = sum_m BG_DCT[k][m] * db[m] */
{_floats("BG_DCT", dct)}
/* Linear SVM over [{N_MFCC} means, {N_MFCC} stds]. The StandardScaler is already folded in
 * (svm.fold), so the device applies these directly to raw features. */
{_floats("BG_W", model.w)}
#define BG_BIAS               {_cf(model.b)}

/* Adaptive threshold (CFAR). Track the background at the {adapt.bg_quantile:g} quantile over a
 * trailing {adapt.window_s:g} s and subtract it, so one theta means the same thing in every
 * recording. Divide by a LOCAL spread and discrimination collapses — see adapt.py;
 * BG_GLOBAL_SCALE is fitted on TRAIN and travels with the model. */
#define BG_WIN_S              {_cf(win_s, "g")}
#define BG_ADAPT_WINDOW_S     {_cf(adapt.window_s, "g")}
#define BG_ADAPT_N            {adapt_n}     /* scores in the trailing ring: window_s / win_s */
#define BG_ADAPT_QUANTILE     {_cf(adapt.bg_quantile, "g")}
#define BG_GLOBAL_SCALE       {_cf(adapt.global_scale)}
#define BG_MIN_SCALE          {_cf(adapt.min_scale, "g")}   /* floor on the divisor */
#define BG_AGG_N              {adapt.agg_n}      /* z = max over the last N z-scores */

/* Gate state machine. Mirrors birdgate/gate.py — must be bit-identical in behaviour. */
#define BG_THETA_ON           {_cf(theta)}
#define BG_THETA_OFF          {_cf(theta_off)}   /* theta_on - hysteresis_margin */
#define BG_PRE_ROLL_S         {_cf(gp.pre_roll_s, "g")}
#define BG_HANGOVER_S         {_cf(gp.hangover_s, "g")}
#define BG_MIN_ON_S           {_cf(gp.min_on_s, "g")}
#define BG_RECHECK_S          {_cf(gp.recheck_period_s, "g")}   /* the gate can only close at a recheck */

#endif /* BIRDGATE_MODEL_PARAMS_H */
'''
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(out)
    return out


# --------------------------------------------------------------------------- #
# The CNN arm — a different front-end (32 mel bands, no DCT) and a different
# scorer (three convolutions), but the same threshold and gate constants.
# --------------------------------------------------------------------------- #
CNN_N_MELS = 32


def _conv_block(tag: str, w, b, stride_t: int, pad: int) -> str:
    """One BN-folded convolution: OIHW weights flattened, plus its bias.

    Row-major [out][in][kh][kw] is torch's own layout, so the C indexes it the same
    way the reference does and there is no transpose to get wrong.
    """
    o, i, kh, kw = w.shape
    return (f"#define BG_{tag}_OUT      {o}\n"
            f"#define BG_{tag}_IN       {i}\n"
            f"#define BG_{tag}_K        {kh}    /* square: {kh}x{kw} */\n"
            f"#define BG_{tag}_STRIDE_T {stride_t}\n"
            f"#define BG_{tag}_PAD      {pad}\n"
            f"{_floats(f'BG_{tag}_W', w.reshape(-1))}"
            f"{_floats(f'BG_{tag}_B', b)}")


def export_cnn_header(qgate, theta: float, adapt, gate: dict,
                      win_s: float = 1.0, out_path=None,
                      provenance: dict | None = None) -> str:
    """Emit `arms/cnn/model_params.h` from a **quantised** model (`cnn.quantise`).

    Float, not int8, and deliberately — see ROADMAP §4. `cnn.quantise` simulates the
    int8 grid rather than reproducing CMSIS-NN's requantisation, so quantising while
    integrating would confound a port bug with arithmetic drift. Flash is not the
    constraint here: the 10.2 KB DCT matrix the linear arm needs does not exist in
    this arm, which pays for the float weights on its own.
    """
    import torch.nn as nn

    norm, cfg, acts = qgate.norm, qgate.cfg, qgate.acts
    convs = [m for m in qgate.mods if isinstance(m, nn.Conv2d)]
    lin = qgate.mods[-1]
    if len(acts) != len(qgate.mods) + 1:
        raise ValueError(f"expected one quantiser per module plus the input, "
                         f"got {len(acts)} for {len(qgate.mods)} modules")
    if not isinstance(lin, nn.Linear):
        raise ValueError(f"expected a Linear last, got {type(lin).__name__}")
    for c in convs:
        if c.bias is None:
            raise ValueError("convolutions must be BN-folded before export (cnn.fold_bn)")

    mel = mel_filterbank(n_mels=CNN_N_MELS)
    win = hann_window()
    nz = mel > 0
    start = np.array([np.argmax(r) if r.any() else 0 for r in nz], dtype=np.int64)
    length = nz.sum(axis=1).astype(np.int64)
    weights = np.concatenate([mel[i, start[i]:start[i] + length[i]] for i in range(len(mel))])
    offset = np.concatenate([[0], np.cumsum(length)[:-1]])
    for i in range(len(mel)):
        assert nz[i, start[i]:start[i] + length[i]].all(), f"mel band {i} is not contiguous"

    win_samples = int(round(win_s * SAMPLE_RATE))
    fpw = 1 + (win_samples - MEL_NFFT) // MEL_HOP
    half = MEL_NFFT // 2
    theta_off = theta - gate["hysteresis_margin"]

    from .config import GateParams
    gp = GateParams(theta_on=theta, **gate)
    if (adapt.mode, adapt.scale, adapt.agg) != ("quantile", "global", "max"):
        raise ValueError(f"firmware implements mode=quantile/scale=global/agg=max, "
                         f"got {adapt.mode}/{adapt.scale}/{adapt.agg}")
    adapt_n = int(round(adapt.window_s / win_s))

    if cfg.input_centre != "ema":
        raise ValueError("the firmware tracks the input level; export an arm trained "
                         "with CnnParams.input_centre='ema'")

    blocks = "\n".join(
        _conv_block(f"CONV{i}", c.weight.detach().numpy(), c.bias.detach().numpy(),
                    stride_t=(cfg.first_stride_t if i == 0 else 1), pad=cfg.kernel // 2)
        for i, c in enumerate(convs))

    p = provenance or {}
    prov = "\n".join(f" *   {k:<22s} {v}" for k, v in p.items())
    out = f'''/* model_params.h — GENERATED by birdgate/export.py. DO NOT HAND-EDIT.
 *
 * Regenerate with:  ../.venv/bin/python -m scripts.export_cnn
 *
 * The CNN arm: 32-band log-mel -> three 3x3 convolutions -> global average pool ->
 * one linear unit. BatchNorm is folded into the convolutions, so what is here is
 * exactly what runs. Float, not int8 — ROADMAP §4 says why.
 *
 * Provenance — so a card recovered from a device months later resolves to the run
 * that produced the firmware that wrote it:
 *
{prov}
 *   generated              {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
 */
#ifndef BIRDGATE_MODEL_PARAMS_H
#define BIRDGATE_MODEL_PARAMS_H
#include <stdint.h>

#define BG_SAMPLE_RATE        {SAMPLE_RATE}
#define BG_WIN_SAMPLES        {win_samples}
#define BG_FFT_SIZE           {MEL_NFFT}
#define BG_HOP_SAMPLES        {MEL_HOP}
#define BG_N_BINS             {MEL_NFFT // 2 + 1}
#define BG_N_MELS             {CNN_N_MELS}

/* Frames per decision window. With center=False the last frame must END inside the
 * window: 1 + ({win_samples} - {MEL_NFFT})/{MEL_HOP} = {fpw}, NOT {win_samples // MEL_HOP}. Global average
 * pooling makes the network indifferent to this width, but the WEIGHTS are not:
 * they were fitted on {fpw}-frame patches from the logmel-dev cache. */
#define BG_FRAMES_PER_WINDOW  {fpw}

/* What the scorer consumes: the globally-average-pooled channels of the last conv. */
#define BG_N_FEATURES         {lin.in_features}
#define BG_MEL_NNZ            {len(weights)}
#define BG_LOG_FLOOR          {AMIN:g}f      /* power_to_db amin; clamp BEFORE log10 */

/* Input standardisation. The network sees (log-mel - tracked level) / BG_IN_STD.
 *
 * BG_IN_LEVEL_ALPHA is the per-window weight of a causal estimate of the acquisition
 * level: seeded from the first frame, updated once per window from that window's own
 * mean, and applied to the window AFTER the one that produced it. It replaces a fixed
 * global mean, which made the arm depend on absolute recording level -- measured,
 * detection fell from 91.5% to 19.0% at -20 dB and to nothing at -30 dB, because the
 * input quantiser spans only ~46 dB and everything outside it clamps.
 *
 * BG_IN_MEAN is recorded for provenance only: the reference adds it before the
 * normaliser subtracts it, so it cancels and the firmware never needs it. */
#define BG_IN_MEAN            {_cf(norm.mean)}
#define BG_IN_STD             {_cf(norm.std)}
#define BG_IN_LEVEL_ALPHA     {_cf(1.0 / max(cfg.centre_tau_s, 1e-9))}
#define BG_IN_LEVEL_TAU_S     {_cf(cfg.centre_tau_s, "g")}

/* Periodic Hann, first half + midpoint. Mirror in C: w[n] = BG_HANN[n <= {half} ? n : {MEL_NFFT} - n] */
{_floats("BG_HANN", win[:half + 1])}
/* FFT twiddles for an in-place radix-2 complex DFT of size {MEL_NFFT}. */
{_floats("BG_TW_COS", np.cos(-2.0 * np.pi * np.arange(MEL_NFFT // 2) / MEL_NFFT))}
{_floats("BG_TW_SIN", np.sin(-2.0 * np.pi * np.arange(MEL_NFFT // 2) / MEL_NFFT))}
/* Mel filterbank, sparse. slaney-normalised, fmin={FMIN:.0f}, fmax={FMAX:.0f}. */
{_ints("BG_MEL_START", start)}{_ints("BG_MEL_LEN", length)}{_ints("BG_MEL_OFF", offset)}{_floats("BG_MEL_W", weights)}
/* The network. Convolutions are OIHW, padding is SAME on both axes, and every conv
 * is followed by ReLU then 2x2 max-pool. Frequency: {CNN_N_MELS} -> {CNN_N_MELS // 2} -> {CNN_N_MELS // 4} -> {CNN_N_MELS // 8}. */
#define BG_N_CONV             {len(convs)}
{blocks}
/* Activation grid. One quantiser for the input, then one after EVERY module, which
 * is the order cnn.quantise builds them in: (conv, ReLU, pool) x BG_N_CONV, then the
 * average pool, the flatten and the linear unit. A stored code is read back as
 * (code - offset) * scale — exactly the value cnn.ActQuant produces — so the device
 * runs the int8 grid with float arithmetic and needs no integer requantisation. */
#define BG_N_ACT              {len(acts)}
{_floats("BG_ACT_SCALE", np.array([a.scale for a in acts]))}{_floats("BG_ACT_OFFSET", np.array([128.0 if a.signed else 0.0 for a in acts]))}
/* Final linear unit over the pooled channels. */
{_floats("BG_LIN_W", lin.weight.detach().numpy().reshape(-1))}
#define BG_LIN_B              {_cf(float(lin.bias.detach().numpy()[0]))}

/* Adaptive threshold (CFAR) — identical to every other arm. */
#define BG_WIN_S              {_cf(win_s, "g")}
#define BG_ADAPT_WINDOW_S     {_cf(adapt.window_s, "g")}
#define BG_ADAPT_N            {adapt_n}     /* scores in the trailing ring */
#define BG_ADAPT_QUANTILE     {_cf(adapt.bg_quantile, "g")}
#define BG_GLOBAL_SCALE       {_cf(adapt.global_scale)}
#define BG_MIN_SCALE          {_cf(adapt.min_scale, "g")}
#define BG_AGG_N              {adapt.agg_n}

/* Gate state machine. Mirrors birdgate/gate.py — must be bit-identical in behaviour. */
#define BG_THETA_ON           {_cf(theta)}
#define BG_THETA_OFF          {_cf(theta_off)}   /* theta_on - hysteresis_margin */
#define BG_PRE_ROLL_S         {_cf(gp.pre_roll_s, "g")}
#define BG_HANGOVER_S         {_cf(gp.hangover_s, "g")}
#define BG_MIN_ON_S           {_cf(gp.min_on_s, "g")}
#define BG_RECHECK_S          {_cf(gp.recheck_period_s, "g")}

#endif /* BIRDGATE_MODEL_PARAMS_H */
'''
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(out)
    return out

