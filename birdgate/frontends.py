"""One interface over the front-end arms, so `train.py` is arm-agnostic.

| arm      | what it tests                                                         |
|----------|-----------------------------------------------------------------------|
| `carihc` | the neuromorphic cochlea that inspired the project                    |
| `bands`  | **the control that matters** — an ERB-spaced bank of *independent*     |
|          | biquad bandpass filters at the same P, same rectify-and-sum. If it     |
|          | ties, the cascade and the asymmetric zeros are decoration and we ship  |
|          | the simpler thing, which is easier to get right in fixed point.        |
| `mfcc`   | the standard front-end, usually rejected on cost — and the one that ships |
| `logmel` | input to the tiny-CNN arm; `logmel-dev` is its causal, device form    |

Every arm returns `(n_windows, n_features)` float32 and reports its cost, so the
budget table in docs/lab-notebook.md §8 is computed rather than guessed.

`bands` is deliberately the *closest possible* control: same channel count, same
ERB spacing, same half-wave rectification, same window summation, same
normalisation. The only difference is cascade-with-zeros versus independent
bandpass. That isolates the cascade's actual claim instead of comparing two
pipelines that differ in five ways at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

from . import carfac
from .config import FrontEndParams

EPS = 1e-12


@dataclass(frozen=True)
class FrontEnd:
    """A designed front-end: how to compute features, and what it costs."""
    kind: str
    n_channels: int
    centre_hz: np.ndarray
    feature_names: list[str]
    _bank: object = None
    _sos: np.ndarray = None
    sr: int = 32_000

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def build(p: FrontEndParams, sr: int) -> FrontEnd:
    """Design the front-end named by `p.kind`."""
    if p.kind == "carihc":
        bank = carfac.design(p.carihc, sr)
        centre = bank.pole_hz
        names = [f"car{i:02d}@{f:.0f}Hz" for i, f in enumerate(centre)]
        fe = FrontEnd("carihc", bank.n_channels, centre, names + _extra_names(p),
                      _bank=bank, sr=sr)
    elif p.kind == "bands":
        c = p.carihc
        rates = np.linspace(carfac.erb_rate(c.f_hi), carfac.erb_rate(c.f_lo), c.n_channels)
        centre = carfac.erb_rate_inv(rates)
        # One ERB wide, the auditory-filter bandwidth at that centre — so the bank
        # tiles the spectrum the same way the cascade does.
        bw = carfac.erb_hz(centre)
        sos = np.stack([
            butter(1, [max(f - bw_ / 2, 20.0) / (sr / 2), min(f + bw_ / 2, sr / 2 - 20) / (sr / 2)],
                   btype="bandpass", output="sos")[0]
            for f, bw_ in zip(centre, bw)
        ])
        names = [f"bnd{i:02d}@{f:.0f}Hz" for i, f in enumerate(centre)]
        fe = FrontEnd("bands", c.n_channels, centre, names + _extra_names(p), _sos=sos, sr=sr)
    elif p.kind in ("mfcc", "mfcc-dev", "logmel", "logmel-dev"):
        # 320-sample hop = exactly 100 frames/s at 32 kHz, so a 1 s window is 100
        # frames and a 0.5 s hop is 50 — no fractional-frame alignment anywhere.
        n = 20 if p.kind in ("mfcc", "mfcc-dev") else 32
        centre = np.zeros(n)
        names = ([f"mfcc{i:02d}_mean" for i in range(n)] + [f"mfcc{i:02d}_std" for i in range(n)]
                 if p.kind in ("mfcc", "mfcc-dev") else [f"mel{i:02d}" for i in range(n)])
        fe = FrontEnd(p.kind, n, centre, names, sr=sr)
    else:
        raise ValueError(f"unknown arm {p.kind!r}")
    return fe


def _extra_names(p: FrontEndParams) -> list[str]:
    extra = []
    if p.append_energy:
        extra.append("log_energy")
    if p.append_flatness:
        extra.append("log_flatness")
    return extra


MEL_HOP = 320       # 100 frames/s at 32 kHz
MEL_NFFT = 1024


def logmel(x: np.ndarray, sr: int, n_mels: int = 32, dev: bool = False) -> np.ndarray:
    """Log-mel for the CNN arm, (n_mels, n_frames) at 100 frames/s.

    Deliberately 32 bands, not the sibling project's 128: this has to fit a
    microcontroller, and the mel filterbank table itself is flash. Note the CNN
    arm pays for an FFT that neither `carihc` nor `bands` needs — which is the
    whole cost argument against it, so it must be counted in `cost_model`.

    `dev=True` is the DEVICE-FAITHFUL variant, and the two changes are exactly the
    ones `mfcc-dev` needed — see docs/lab-notebook.md §13 G6b:

    - `center=False`. librosa otherwise pads by n_fft//2 and centres frame i on
      sample i*hop, which a device has no future samples to do. It also drops a 1 s
      window from 100 whole frames to 97, since the last frame must *end* inside it.
    - `top_db=None`. The default clamps 80 dB below the maximum of the **whole
      array**, so audio arriving later silently rewrites features already computed.
      That is the non-causality that makes C/Python parity fail intermittently, on
      exactly the loud-transient recordings a deployment produces most.
    """
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=MEL_NFFT, hop_length=MEL_HOP,
                                       n_mels=n_mels, fmin=150, fmax=min(14_000, sr // 2),
                                       power=2.0, center=not dev)
    return librosa.power_to_db(m, ref=1.0, top_db=None if dev else 80.0).astype(np.float32)


def raw_templates(x: np.ndarray, fe: FrontEnd, win_samples: int,
                  hop_samples: int | None = None) -> np.ndarray:
    """Un-normalised Phi: rectified channel output summed over each window."""
    if fe.kind in ("mfcc", "mfcc-dev"):
        import librosa
        x = np.asarray(x, dtype=np.float32)
        hop_samples = hop_samples or win_samples
        if fe.kind == "mfcc-dev":
            # Device-faithful: every value here is one a streaming implementation
            # could compute from audio it has ALREADY seen. Two changes, both about
            # causality rather than accuracy — see docs/lab-notebook.md §13 G6b.
            #
            # `center=False`: librosa pads by n_fft//2 and centres frame i on sample
            # i*hop. A device has no future samples to centre on, so frame i covers
            # [i*hop, i*hop + n_fft).
            #
            # `top_db=None`: power_to_db's default clamps 80 dB below the maximum of
            # the WHOLE array, so audio arriving later silently rewrites features
            # already computed. Measured on this corpus it fires on 1 recording in 8
            # and 0.42% of that one's frames — so dropping it barely moves the model,
            # but it is precisely the non-causality that would make C/Python parity
            # fail intermittently, on the loud-transient recordings a field
            # deployment produces most.
            S = librosa.feature.melspectrogram(
                y=x, sr=fe.sr, n_fft=MEL_NFFT, hop_length=MEL_HOP, fmin=150,
                fmax=min(14_000, fe.sr // 2), power=2.0, center=False)
            mf = librosa.feature.mfcc(S=librosa.power_to_db(S, ref=1.0, top_db=None),
                                      n_mfcc=fe.n_channels)
            # With center=False a 1 s window holds 97 WHOLE frames, not 100 — the last
            # frame has to end inside the window. Using 100 would pull three frames of
            # the NEXT window's audio into this window's features.
            fpw = 1 + (win_samples - MEL_NFFT) // MEL_HOP
        else:
            mf = librosa.feature.mfcc(y=x, sr=fe.sr, n_mfcc=fe.n_channels,
                                      n_fft=MEL_NFFT, hop_length=MEL_HOP, fmin=150,
                                      fmax=min(14_000, fe.sr // 2))
            fpw = win_samples // MEL_HOP
        fph = hop_samples // MEL_HOP
        n_win = 1 + (mf.shape[1] - fpw) // fph if mf.shape[1] >= fpw else 0
        idx = np.arange(n_win)[:, None] * fph + np.arange(fpw)[None, :]
        frames = mf[:, idx]                                  # (C, W, fpw)
        return np.concatenate([frames.mean(axis=2).T, frames.std(axis=2).T],
                              axis=1).astype(np.float32)
    if fe.kind == "carihc":
        return carfac.templates(x, fe._bank, win_samples, hop_samples)
    if hop_samples is None:
        hop_samples = win_samples
    x = np.asarray(x, dtype=np.float64)
    n_win = 1 + (len(x) - win_samples) // hop_samples if len(x) >= win_samples else 0
    if n_win <= 0:
        return np.zeros((0, fe.n_channels), dtype=np.float32)
    # Chunked like the cascade, for the same memory reason: 40 channels x 1 h at
    # 32 kHz would be ~18 GB held at once.
    starts_all = np.arange(n_win) * hop_samples
    out = np.zeros((n_win, fe.n_channels), dtype=np.float32)
    hops_per_chunk = max(1, int(round(30.0 * fe.sr / hop_samples)))
    zi = [None] * fe.n_channels
    w0 = 0
    while w0 < n_win:
        w1 = min(w0 + hops_per_chunk, n_win)
        s0 = w0 * hop_samples
        s1 = min(len(x), (w1 - 1) * hop_samples + win_samples)
        rect = np.empty((fe.n_channels, s1 - s0), dtype=np.float32)
        for ch in range(fe.n_channels):
            if zi[ch] is None:
                zi[ch] = np.zeros((1, 2))
            y, zi[ch] = sosfilt(fe._sos[ch:ch + 1], x[s0:s1], zi=zi[ch])
            rect[ch] = np.maximum(y, 0.0)
        out[w0:w1] = carfac.window_sums(rect, starts_all[w0:w1] - s0, win_samples,
                                       hop_samples)
        w0 = w1
    return out


def normalise(phi: np.ndarray, p: FrontEndParams) -> np.ndarray:
    """Apply the normaliser and append the level features.

    ⚠️ **docs/lab-notebook.md §4 — this is not a cosmetic choice for a presence detector.**
    L1-across-channels makes the vector loudness-invariant, which is exactly right
    for species ID (a bird at 5 m and at 15 m should classify alike) and exactly
    wrong for "is anything there", because silence and birdsong both normalise to
    a unit-sum vector. The two appended features put the discarded information
    back explicitly, for two weights:

    - `log_energy` — the level the normaliser threw away.
    - `log_flatness` — spectral flatness (geometric / arithmetic mean across
      channels). Near 1 for noise-like signals, near 0 for tonal ones. Wind,
      running water and rain are *loud but flat*; birdsong is *structured*. On the
      Highland site those are the failure modes that will otherwise fill a card,
      so this feature is aimed squarely at the target deployment rather than at
      Sierra Nevada.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.size == 0:
        return phi.astype(np.float32)
    if p.kind in ("mfcc", "mfcc-dev"):
        # MFCCs are signed (they are a cosine transform of log energies), so L1
        # normalisation and a geometric mean are both meaningless here. The arm
        # gets its own natural treatment — raw coefficients, standardised in the
        # model pipeline — rather than being handicapped by a normaliser designed
        # for non-negative filterbank outputs.
        return phi.astype(np.float32)

    total = phi.sum(axis=1, keepdims=True)
    # Geometric mean via logs — safe at zero, unlike prod()**(1/n).
    logphi = np.log(phi + EPS)
    flatness = np.exp(logphi.mean(axis=1)) / (phi.mean(axis=1) + EPS)

    if p.normalise == "l1":
        shape = phi / (total + EPS)
    elif p.normalise == "l2":
        shape = phi / (np.linalg.norm(phi, axis=1, keepdims=True) + EPS)
    elif p.normalise == "none":
        shape = phi
    else:
        raise ValueError(f"unknown normaliser {p.normalise!r}")

    cols = [shape]
    if p.append_energy:
        cols.append(np.log(total + EPS))
    if p.append_flatness:
        cols.append(np.log(flatness + EPS)[:, None])
    return np.concatenate(cols, axis=1).astype(np.float32)


def extract(x: np.ndarray, fe: FrontEnd, p: FrontEndParams, win_samples: int,
            hop_samples: int | None = None) -> np.ndarray:
    """Waveform -> per-window feature matrix, normalised. The arm-agnostic call."""
    return normalise(raw_templates(x, fe, win_samples, hop_samples), p)


def cost_model(fe: FrontEnd, n_weights: int | None = None) -> dict:
    """MACs/s, SRAM and flash on a Cortex-M4F. Feeds docs/lab-notebook.md §8.

    Counted honestly per sample, coupled form for `carihc` (6 MACs/stage) and
    transposed direct-form II for `bands` (5 MACs/stage), plus ~1 op/channel for
    rectify-and-accumulate. State is 2 floats per channel either way.
    """
    if fe.kind in ("mfcc", "mfcc-dev", "logmel", "logmel-dev"):
        # An FFT the filterbank arms simply do not perform: N log2 N butterflies
        # per frame at 100 frames/s, plus the mel projection and (for mfcc) a DCT.
        n_fft, frames = MEL_NFFT, fe.sr / MEL_HOP
        macs = frames * (n_fft * np.log2(n_fft) * 0.5 + n_fft // 2 * fe.n_channels)
    else:
        per_stage = 6.0 if fe.kind == "carihc" else 5.0
        macs = fe.n_channels * fe.sr * (per_stage + 1.0)
    n_weights = fe.n_features if n_weights is None else n_weights
    return {
        "kind": fe.kind,
        "macs_per_s": macs,
        "pct_of_48mhz_m4f": 100 * macs / 48e6,
        # ⚠️ The three byte counts below describe a BIQUAD BANK — 2 state and 5
        # coefficients per channel. For the FFT arms they are meaningless: those
        # have no per-channel filter state, and their real memory is the window,
        # twiddle, mel and DCT tables, which are nowhere in this model. Measured
        # properly, mfcc is 20.6 KB flash and 16.3 KB SRAM — see docs/lab-notebook.md §13 G6b.
        # `macs_per_s` is the only field here valid for every arm.
        "state_bytes": fe.n_channels * 2 * 4,
        "coeff_bytes": fe.n_channels * 5 * 4,       # 5 coefficients per stage
        "weight_bytes": (n_weights + 1) * 4,
        "total_bytes": fe.n_channels * 7 * 4 + (n_weights + 1) * 4,
    }
