"""Validate the filterbank against signals whose answer is known.

⚠️ **Why this file exists.** A cascade is an IIR chain. An error in the pole
design does not raise — it silently shifts every channel, and the resulting drop
in model performance gets attributed to the classifier, where it is unfixable.
An earlier project on this dataset lost real measured performance to exactly this
failure mode (mel bin edges built from filter *centres*). The lesson
taken from it is that a front-end must be tested in isolation against ground
truth you can derive by hand, before anything is built on top of it.

Run:  ../.venv/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pytest

from birdgate.carfac import (
    CarBank, cascade, cascade_stepwise, design, erb_hz, erb_rate, erb_rate_inv, ihc, templates,
)
from birdgate.config import CarIhcParams

SR = 32_000


@pytest.fixture(scope="module")
def bank() -> CarBank:
    return design(CarIhcParams(), SR)


def tone_rms(bank: CarBank, f_hz: float, dur_s: float = 0.3) -> np.ndarray:
    """Per-channel RMS response to a steady tone, transient discarded."""
    t = np.arange(int(dur_s * SR)) / SR
    bm = cascade(np.sin(2 * np.pi * f_hz * t) * 0.5, bank)
    return np.sqrt((bm[:, int(0.15 * SR):] ** 2).mean(axis=1))


# --------------------------------------------------------------------------- #
# Design                                                                       #
# --------------------------------------------------------------------------- #
def test_erb_scale_roundtrips():
    f = np.geomspace(50, 15_000, 50)
    assert np.allclose(erb_rate_inv(erb_rate(f)), f)


def test_erb_bandwidth_tracks_glasberg_moore_slope():
    """`erb_hz` is Lyon's CARFAC variant, not literally Glasberg & Moore.

    Same slope (0.10794 Hz per Hz), different intercept (17.84 vs 24.70), so they
    agree above ~1 kHz and diverge ~19% at 100 Hz. Pinning this stops someone
    "fixing" one constant to match a textbook and silently retuning the damping,
    which is derived from these same numbers.
    """
    f = np.array([1000.0, 8000.0])
    gm = 24.7 * (4.37 * f / 1000 + 1)
    assert np.allclose(erb_hz(f) / gm, [0.948, 0.992], atol=0.005)
    slope = np.diff(erb_hz(f)) / np.diff(f)
    assert slope[0] == pytest.approx(24.7 * 4.37 / 1000, rel=1e-6)


def test_poles_descend_and_span_the_requested_band(bank):
    assert bank.pole_hz[0] == pytest.approx(12_000.0)
    assert bank.pole_hz[-1] == pytest.approx(150.0)
    assert np.all(np.diff(bank.pole_hz) < 0), "cascade order must be base -> apex"


def test_channels_are_equally_spaced_on_the_erb_scale(bank):
    steps = np.abs(np.diff(erb_rate(bank.pole_hz)))
    assert steps.std() < 1e-9, "spacing must be uniform in ERB-number, not in Hz"


def test_every_stage_is_stable(bank):
    """r < 1 is the stability condition; assert it rather than trusting design()."""
    assert np.all(bank.r < 1.0)
    assert np.all(bank.r > 0.0)
    for ch in range(bank.n_channels):
        poles = np.roots([1.0, -2 * bank.r[ch] * bank.a0[ch], bank.r[ch] ** 2])
        assert np.abs(poles).max() < 1.0


def test_each_stage_has_unity_dc_gain(bank):
    """The whole point of `g`. Without it a 40-stage cascade would blow up or die."""
    b, a = bank.sos[:, :3], bank.sos[:, 3:]
    dc = b.sum(axis=1) / a.sum(axis=1)
    assert np.allclose(dc, 1.0, atol=1e-12)


def test_dc_survives_the_full_cascade(bank):
    """40 stages at unity DC gain must still pass DC at unity, not 0.9^40."""
    x = np.ones(4000)
    out = cascade(x, bank)
    assert out[-1, -1] == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Frequency behaviour — the tests the mel-bin bug would have failed             #
# --------------------------------------------------------------------------- #
def test_tone_excites_its_own_channel(bank):
    """A tone at channel p's pole must peak at channel p (or its neighbour).

    The +1 tolerance is not slop, it is the cascade: channel p's *effective*
    response is the product of stages 0..p, and the stages above p have already
    attenuated, which pushes the peak of the effective response slightly above the
    design pole. Measured, the empirical best frequency sits ~7.5% high, so a tone
    at pole_hz[p] most excites p or p+1. Anything further means the design is wrong.
    """
    offsets = [int(np.argmax(tone_rms(bank, f))) - ch
               for ch, f in enumerate(bank.pole_hz)]
    assert min(offsets) >= 0 and max(offsets) <= 1, f"offsets out of range: {offsets}"


def test_best_frequency_is_monotonic(bank):
    """Channel order must track frequency order. A scrambled bank still 'works'
    numerically and produces a plausible-looking model — this is what catches it."""
    freqs = np.geomspace(120, 14_000, 120)
    resp = np.array([tone_rms(bank, f) for f in freqs])          # (F, P)
    bf = freqs[np.argmax(resp, axis=0)]
    assert np.all(np.diff(bf) <= 0), "best frequency must fall as channel index rises"
    ratio = bf / bank.pole_hz
    assert 0.9 < np.median(ratio) < 1.2, f"BF/pole median {np.median(ratio):.3f}"


def test_response_is_asymmetric_steeper_above_than_below(bank):
    """The 'A' in CAR. If the zeros are misplaced this symmetry test fails while
    every other test still passes."""
    ch = bank.n_channels // 2
    f0 = bank.pole_hz[ch]
    peak = tone_rms(bank, f0)[ch]
    above = tone_rms(bank, f0 * 2.0)[ch]
    below = tone_rms(bank, f0 / 2.0)[ch]
    assert above < below, "high side must roll off faster than the low side"
    assert above < peak and below < peak


def test_chirp_sweeps_channels_monotonically(bank):
    """A rising chirp must walk the argmax channel down, smoothly."""
    dur = 2.0
    t = np.arange(int(dur * SR)) / SR
    f0, f1 = 200.0, 10_000.0
    phase = 2 * np.pi * f0 * dur / np.log(f1 / f0) * ((f1 / f0) ** (t / dur) - 1)
    rect = ihc(cascade(np.sin(phase) * 0.5, bank))
    # coarse frames so a single sample's noise can't flip the argmax
    n_fr = 40
    frames = rect[:, :n_fr * (rect.shape[1] // n_fr)].reshape(bank.n_channels, n_fr, -1)
    peak_ch = frames.sum(axis=2).argmax(axis=0)[2:-2]           # drop edge transients
    # Channel 0 is the HIGHEST frequency (cascade order is base -> apex), so a
    # *rising* chirp must walk the argmax index *downward*, without reversing.
    assert np.all(np.diff(peak_ch) <= 0), f"chirp did not sweep monotonically: {peak_ch}"
    assert peak_ch[0] - peak_ch[-1] > bank.n_channels // 2, "chirp barely moved"


# --------------------------------------------------------------------------- #
# Degenerate inputs                                                            #
# --------------------------------------------------------------------------- #
def test_silence_gives_exactly_zero_and_no_nan(bank):
    """Catches a normaliser that divides by zero, and any NaN seeded at init."""
    out = cascade(np.zeros(2000), bank)
    assert np.all(out == 0.0)
    phi = templates(np.zeros(SR * 2), bank, win_samples=SR)
    assert phi.shape == (2, bank.n_channels)
    assert np.all(np.isfinite(phi)) and np.all(phi == 0.0)


def test_impulse_response_decays(bank):
    """Every stage stable => the cascade must ring down, not sustain or grow."""
    x = np.zeros(SR); x[0] = 1.0
    out = np.abs(cascade(x, bank))
    early = out[:, :SR // 10].max(axis=1)
    late = out[:, -SR // 10:].max(axis=1)
    assert np.all(late < 0.01 * early + 1e-12), "impulse response failed to decay"


def test_ihc_is_half_wave_rectification():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.array_equal(ihc(x), np.array([0.0, 0.0, 0.0, 0.5, 2.0]))


# --------------------------------------------------------------------------- #
# The parity a fixed-point port depends on                                                #
# --------------------------------------------------------------------------- #
def test_sosfilt_path_matches_the_coupled_form(bank):
    """The training path and the firmware's reference must be the same filter.

    `cascade` uses scipy for speed; `cascade_stepwise` is the literal coupled-form
    loop `firmware/src/car_ihc.c` mirrors. If these ever diverge, every number
    measured in Python is a number the device will not reproduce.
    """
    x = np.random.default_rng(0).standard_normal(3000) * 0.01
    fast, literal = cascade(x, bank), cascade_stepwise(x, bank)
    assert np.abs(fast - literal).max() / np.abs(literal).max() < 1e-10


def test_chunked_templates_match_unchunked(bank):
    """Filter state must carry across chunk seams, or long recordings get a glitch
    every chunk_s seconds that no test on short signals would ever reveal."""
    x = np.random.default_rng(1).standard_normal(SR * 8) * 0.01
    a = templates(x, bank, win_samples=SR, chunk_s=1.0)
    b = templates(x, bank, win_samples=SR, chunk_s=100.0)
    assert np.allclose(a, b, rtol=1e-6, atol=1e-9)
