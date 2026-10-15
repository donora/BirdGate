"""CAR-IHC — the neuromorphic cochlea front-end (Lyon's CARFAC, linear CAR).

A **Cascade of Asymmetric Resonators**: `P` two-pole-two-zero stages in series,
pole frequencies ERB-spaced from high (cochlear base) to low (apex). Each stage's
basilar-membrane output is tapped as one channel. The Inner Hair Cell model here
is simply a half-wave rectifier. Channel p's template value is its
rectified output summed over the decision window.

Why a *cascade* rather than a bank of independent bandpass filters: the signal
reaching channel p has already passed through every higher-frequency stage, so
each channel inherits a sharp high-side roll-off for free. That is both what real
cochlear tuning curves look like (asymmetric — steep above the characteristic
frequency, gradual below) and why it is cheap: one biquad per channel buys a
filter whose effective order is `p`.

The case for this over MFCC: **no FFT and no logarithm**. The entire front-end is a
chain of biquads and a `max(0, x)`. Whether that case holds at 32 kHz is what the
arm comparison measures.

## Two implementations, deliberately

`cascade()` uses `scipy.signal.sosfilt` and is the fast path used for training.
`cascade_stepwise()` is the literal sample-by-sample **coupled form** that the C
firmware will mirror. `tests/test_carfac_reference.py` asserts they agree.

They are the same filter — the stage transfer function derives to a plain biquad:

    H(z) = g * (1 + (h·r·c0 - 2·r·a0)·z^-1 + r²·z^-2)
              / (1 -            2·r·a0 ·z^-1 + r²·z^-2)

with `g` chosen for unity DC gain, which reproduces Lyon's `CARFAC_Stage_g`
exactly. Keeping both matters for a fixed-point port: **direct-form biquads are a poor choice in
fixed point** when poles sit near the unit circle (r ≈ 0.99 at low channels here),
because the denominator coefficients become huge and ill-conditioned. The coupled
form stores a rotation (`a0`, `c0` are a sine/cosine pair, both ≤ 1) and is
numerically well-behaved at Q15. The firmware must use the coupled form; float
training can use either.

## Validation

⚠️ A cascade is an IIR chain: an error in the pole design does not announce
itself, it silently shifts every channel and degrades the model by an amount you
will attribute to the classifier. An earlier project on this dataset lost real
measured performance to exactly this (mel bin edges built from filter centres). So the
filterbank is validated against signals whose answer is known *before* anything is
built on it — see `tests/test_carfac_reference.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from .config import CarIhcParams

# ⚠️ There are TWO "ERB" definitions in play here and they are not the same.
#
#   erb_hz()   — the BANDWIDTH of the auditory filter at f, used to set damping.
#                These are Lyon's CARFAC constants: ERB = (f + 165.3)/9.2645.
#                That has the same slope as Glasberg & Moore 1990
#                (24.7*(4.37*f/1000 + 1)) but a different intercept — 17.84 vs
#                24.70 — so the two agree above ~1 kHz and differ by ~19% at
#                100 Hz. Kept as Lyon's because the damping constants below were
#                tuned against them; swapping one without the other would retune
#                the whole bank silently.
#
#   erb_rate() — POSITION on the ERB-number scale, used to space the channels.
#                This one *is* Glasberg & Moore: 21.4*log10(1 + 0.00437 f).
#
# Spacing channels uniformly in erb_rate means equal numbers of auditory filters
# apart, not equal Hz — which is why the low channels are packed close together.
ERB_BREAK_HZ = 165.3
ERB_Q = 1000.0 / (24.7 * 4.37)   # 9.2645


def erb_hz(f_hz):
    """Auditory filter bandwidth (Hz) at `f_hz`, in Lyon's CARFAC form.

    NOT identical to Glasberg & Moore — same slope, lower intercept. See the note
    above the constants. Used only for damping, never for channel spacing.
    """
    return (np.asarray(f_hz, dtype=float) + ERB_BREAK_HZ) / ERB_Q


def erb_rate(f_hz):
    """Position on the Glasberg & Moore ERB-number scale — how many auditory
    filters up `f_hz` sits. This is what channel spacing is uniform in."""
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(f_hz, dtype=float))


def erb_rate_inv(rate):
    """Inverse of `erb_rate`."""
    return (10.0 ** (np.asarray(rate, dtype=float) / 21.4) - 1.0) / 0.00437


@dataclass(frozen=True)
class CarBank:
    """A designed filterbank. Everything the cascade and the exporter need."""
    sr: int
    pole_hz: np.ndarray      # (P,) descending — base to apex, cascade order
    r: np.ndarray            # pole radius
    a0: np.ndarray           # cos(pole angle)
    c0: np.ndarray           # sin(pole angle)
    h: np.ndarray            # zero placement
    g: np.ndarray            # DC-normalising gain
    sos: np.ndarray          # (P, 6) second-order sections, one per stage

    @property
    def n_channels(self) -> int:
        return len(self.pole_hz)

    def erb_per_step(self) -> float:
        rates = erb_rate(self.pole_hz)
        return float(np.abs(np.diff(rates)).mean())


def design(p: CarIhcParams, sr: int) -> CarBank:
    """Design the cascade. Poles ERB-spaced, descending from `f_hi` to `f_lo`.

    Spacing is uniform on the ERB-*number* scale between the two endpoints, so
    `n_channels` is exact rather than a consequence of a step size. Lyon's CARFAC
    instead fixes `ERB_per_step` and takes whatever channel count falls out; we
    have a hard flash budget, so channel count is the constraint and spacing is
    derived. `CarBank.erb_per_step()` reports what that worked out to.

    Damping follows CARFAC: `zeta` is the damping ratio, raised at high channels
    where the ERB is a larger fraction of the centre frequency. `r = 1 - zr*zeta`
    with `zr < pi`, so `r < 1` and every stage is stable by construction — but the
    tests assert it anyway, because "by construction" is how bugs hide.
    """
    n = p.n_channels
    rates = np.linspace(erb_rate(p.f_hi), erb_rate(p.f_lo), n)
    pole_hz = erb_rate_inv(rates)                       # descending: base -> apex

    # `pole_theta` is the pole ANGLE in radians per sample, Lyon's notation.
    # Named in full to avoid collision with `theta_on`/`theta_off`, which are the
    # gate's decision thresholds and have nothing to do with filter design.
    pole_theta = pole_hz * (2 * np.pi / sr)
    a0, c0 = np.cos(pole_theta), np.sin(pole_theta)

    # Zero placement. `zero_ratio` is the zero frequency as a multiple of the
    # pole frequency; h = c0*(ratio^2 - 1). This is the "asymmetric" in CAR:
    # zeros above the poles make the high side of each response steep.
    f = p.zero_ratio ** 2 - 1.0
    h = c0 * f

    # Damping. `zr` compresses toward zero as pole_theta -> pi so the highest channels
    # do not end up over-damped; `min_zetas` lifts damping where channels are
    # widely spaced. Both are Lyon's; the constants are exposed in CarIhcParams.
    x = pole_theta / np.pi
    zr = np.pi * (x - p.high_f_damping_compression * x ** 3)
    min_zetas = p.min_zeta + 0.25 * (erb_hz(pole_hz) / pole_hz - p.min_zeta)
    r = 1.0 - zr * min_zetas                            # least-damped (open loop)

    # Unity gain at DC, per stage, so a P-stage cascade does not accumulate gain.
    g = (1 - 2 * r * a0 + r ** 2) / (1 - 2 * r * a0 + h * r * c0 + r ** 2)

    sos = np.stack([
        g, g * (h * r * c0 - 2 * r * a0), g * r ** 2,     # b0, b1, b2
        np.ones_like(r), -2 * r * a0, r ** 2,             # a0, a1, a2
    ], axis=1)

    return CarBank(sr=sr, pole_hz=pole_hz, r=r, a0=a0, c0=c0, h=h, g=g, sos=sos)


def cascade(x: np.ndarray, bank: CarBank, zi: np.ndarray | None = None,
            return_state: bool = False):
    """Fast path: BM output per channel, shape (P, len(x)).

    Channel p's output is the input filtered through stages 0..p, so the cascade
    is computed incrementally — each stage filters the previous stage's output.
    `zi` carries filter state across chunks so a long recording can be processed
    in pieces without a seam.
    """
    x = np.asarray(x, dtype=np.float64)
    n = bank.n_channels
    out = np.empty((n, len(x)), dtype=np.float32)
    state = np.zeros((n, 2)) if zi is None else np.asarray(zi, dtype=float).copy()

    sig = x
    for ch in range(n):
        sig, state[ch] = sosfilt(bank.sos[ch:ch + 1], sig, zi=state[ch:ch + 1].reshape(1, 2))
        state[ch] = state[ch].reshape(2)
        out[ch] = sig
    return (out, state) if return_state else out


def cascade_stepwise(x: np.ndarray, bank: CarBank) -> np.ndarray:
    """The literal coupled-form loop the firmware mirrors. Slow; for tests only.

    This is `CARFAC_CAR_Step` with the AGC and outer-hair-cell nonlinearity
    removed — this arm uses the linear CAR. Note the ripple: within one sample,
    stage `ch`'s output feeds stage `ch+1` *with no delay*, which is why the
    channel loop is inside the sample loop and cannot be vectorised over channels.
    """
    x = np.asarray(x, dtype=np.float64)
    n = bank.n_channels
    ra0, rc0 = bank.r * bank.a0, bank.r * bank.c0
    h, g = bank.h, bank.g
    z1 = np.zeros(n)
    z2 = np.zeros(n)
    out = np.empty((n, len(x)), dtype=np.float32)

    for i, u in enumerate(x):
        z1n = ra0 * z1 - rc0 * z2       # rotate, from the OLD state
        z2n = rc0 * z1 + ra0 * z2
        zY = h * z2n
        for ch in range(n):             # the zero-delay ripple
            z1n[ch] += u
            u = g[ch] * (u + zY[ch])
            out[ch, i] = u
        z1, z2 = z1n, z2n
    return out


def ihc(bm: np.ndarray) -> np.ndarray:
    """Inner hair cell: half-wave rectification. That is the whole model."""
    return np.maximum(bm, 0.0)


def cochleagram(x: np.ndarray, bank: CarBank) -> np.ndarray:
    """Rectified BM output over time, (P, len(x)). The debugging picture."""
    return ihc(cascade(x, bank))


def templates(x: np.ndarray, bank: CarBank, win_samples: int, hop_samples: int | None = None,
              chunk_s: float = 30.0) -> np.ndarray:
    """Audio -> per-window template vectors Phi, shape (n_windows, P).

    `Phi[t, p] = sum over window t of max(0, BM_p)` — the template vector, before
    normalisation. Normalisation is `frontends.normalise`'s job, not this
    function's: for a *presence* detector the normaliser choice is load-bearing
    and must stay visible (see docs/lab-notebook.md §4).

    Processed in chunks with carried filter state, because the full cochleagram of
    a 1-hour recording at 32 kHz x 40 channels would be ~18 GB. Chunks are snapped
    to whole hops so no window straddles a seam.
    """
    if hop_samples is None:
        hop_samples = win_samples
    x = np.asarray(x, dtype=np.float64)
    n_win = 1 + (len(x) - win_samples) // hop_samples if len(x) >= win_samples else 0
    if n_win <= 0:
        return np.zeros((0, bank.n_channels), dtype=np.float32)

    # Non-overlapping hops let us accumulate per-window sums streaming. Overlapping
    # hops need the window's samples in memory at once, so fall back to summing
    # from a chunk large enough to contain any single window.
    hops_per_chunk = max(1, int(round(chunk_s * bank.sr / hop_samples)))
    out = np.zeros((n_win, bank.n_channels), dtype=np.float32)

    state = None
    w0 = 0
    while w0 < n_win:
        w1 = min(w0 + hops_per_chunk, n_win)
        s0 = w0 * hop_samples
        s1 = min(len(x), (w1 - 1) * hop_samples + win_samples)
        bm, state = cascade(x[s0:s1], bank, zi=state, return_state=True)
        out[w0:w1] = window_sums(ihc(bm), np.arange(w0, w1) * hop_samples - s0,
                                 win_samples, hop_samples)
        w0 = w1
    return out


def window_sums(rect: np.ndarray, starts: np.ndarray, win_samples: int,
                hop_samples: int | None = None) -> np.ndarray:
    """Sum each channel of `rect` over windows starting at `starts`, shape (W, P).

    Two strategies, because the obvious one is a memory hog:

    **Block sums (used whenever `win_samples` is a multiple of `hop_samples`,
    which is every configuration we actually run).** Sum non-overlapping blocks of
    `hop_samples` once, then a window is the sum of `win/hop` consecutive blocks.
    The expensive reduction happens once per sample instead of once per
    (window, channel) pair, and overlapping windows cost nothing.

    **Prefix sum (fallback).** Correct for any alignment, but it needs a float64
    array the size of the input — ~300 MB per chunk at 40 channels — and float32
    would not do, because differencing two large prefix sums is exactly where
    accumulated rounding surfaces.
    """
    starts = np.asarray(starts)
    if hop_samples and win_samples % hop_samples == 0:
        k = win_samples // hop_samples
        n_blk = rect.shape[1] // hop_samples
        blk = rect[:, :n_blk * hop_samples].reshape(rect.shape[0], n_blk, hop_samples)
        blk = blk.sum(axis=2, dtype=np.float64)                 # (P, n_blk)
        cs = np.concatenate([np.zeros((rect.shape[0], 1)), np.cumsum(blk, axis=1)], axis=1)
        b0 = starts // hop_samples
        return (cs[:, b0 + k] - cs[:, b0]).T.astype(np.float32)

    csum = np.zeros((rect.shape[0], rect.shape[1] + 1), dtype=np.float64)
    np.cumsum(rect, axis=1, out=csum[:, 1:])
    return (csum[:, starts + win_samples] - csum[:, starts]).T.astype(np.float32)
