"""Every knob in one place, so an experiment is a config diff.

Two rules this file exists to enforce:

1. **The device rate and the training rate are the same number.** 32 kHz, native
   to the SNE FLACs and native to what we ask the AudioMoth to sample. Resampling
   between train and deploy is a silent-bug factory; there is no resample step.

2. **Anything that has to appear in `model_params.h` lives here**, so `export.py`
   has exactly one thing to read and the firmware can never drift from the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .adapt import AdaptParams

SAMPLE_RATE = 32_000
"""Device and training sample rate. See docs/lab-notebook.md §3 — 16 kHz would clip the upper
band of ~10% of annotations (95th pct high_hz = 9166 Hz)."""


@dataclass(frozen=True)
class CarIhcParams:
    """CAR-IHC cochlea front-end (Lyon's CARFAC, linear CAR + half-wave IHC).

    A cascade of `n_channels` asymmetric 2-pole-2-zero resonators; channel p's
    basilar-membrane output is half-wave rectified, summed over the window, and
    normalised. Pole frequencies are ERB-spaced from `f_lo` to `f_hi`.
    """
    n_channels: int = 40          # paper used 30 at 8 kHz; more band to cover here
    f_lo: float = 150.0           # below this is wind/rumble, not birds
    f_hi: float = 12_000.0        # covers the 99th pct of annotation high_hz

    # The "asymmetric" in CAR: zeros sit `zero_ratio` x above the poles, which
    # makes each stage's high side steep and its low side gradual — the shape of
    # a real cochlear tuning curve. sqrt(2) is Lyon's default.
    zero_ratio: float = 1.41421356

    # Damping. `r = 1 - zr*zeta` sets the pole radius, so smaller zeta means a
    # sharper, longer-ringing resonance. `min_zeta` is the least-damped (most
    # sensitive) setting; the linear CAR has no AGC, so this is simply where we
    # choose to sit. `high_f_damping_compression` keeps the top channels from
    # over-damping as the pole angle approaches pi.
    min_zeta: float = 0.10
    high_f_damping_compression: float = 0.5


@dataclass(frozen=True)
class FrontEndParams:
    """Which front-end, and how its output is normalised.

    ⚠️ `normalise` is *not* a cosmetic choice for a presence detector — see
    docs/lab-notebook.md §4. L1-across-channels makes the feature loudness-invariant, which is
    right for species ID and actively wrong for "is anything there", since silence
    and birdsong both normalise to a unit-sum vector. Hence `append_energy`.
    """
    kind: Literal["carihc", "bands", "mfcc", "mfcc-dev", "logmel", "logmel-dev"] = "carihc"
    normalise: Literal["none", "l1", "l2"] = "l1"
    append_energy: bool = True      # log(total energy) as an extra feature
    append_flatness: bool = True    # log(spectral flatness) — separates wind/stream
    carihc: CarIhcParams = field(default_factory=CarIhcParams)


@dataclass(frozen=True)
class WindowParams:
    """The decision unit.

    1 s matches the AudioMoth ring-buffer slot (8 x 32 KB @ 16-bit), so it is free.
    Our median annotation box is 2.4 s and the 10th percentile 1.2 s, so a 1 s
    decision rarely splits a call. Training uses overlap; evaluation must not.
    """
    win_s: float = 1.0
    train_hop_s: float = 0.5
    eval_hop_s: float = 1.0
    min_overlap_s: float = 0.15   # a window is positive if >= this much box lands in it

    def hop_for(self, split: str) -> float:
        return self.train_hop_s if split == "train" else self.eval_hop_s


@dataclass(frozen=True)
class GateParams:
    """The state machine that turns per-second scores into recorded regions.

    Hysteresis (`theta_on` > `theta_off`) stops the gate flickering mid-call.
    `pre_roll_s` is nearly free on device: the ring buffer already holds 8 seconds
    of past audio, so opening the gate can rewind `read_pointer` instead of
    allocating anything. It is the cheapest recall in the system.
    """
    # theta_on / theta_off are the GATE THRESHOLDS: the classifier emits one real
    # number per window, f(X) = sum_p Q_p*Phi_p(X) + b, and these decide what score
    # counts as "bird". This is THE tunable knob of the whole system — lower it to
    # record more and miss fewer calls, raise it to record less. Not learned during
    # training; calibrated afterwards on the val recordings (calibrate.py).
    # (Unrelated to `pole_theta` in carfac.py, which is a filter pole angle.)
    theta_on: float = 0.0        # set by calibrate.py; 0.0 = raw SVM decision boundary
    theta_off: float | None = None   # None -> theta_on - hysteresis_margin
    hysteresis_margin: float = 0.25

    # Pre-roll is bounded by the ring buffer, which is 8 x 32 KB = 256 KB of
    # external SRAM. At 16 kHz a 32 KB slot is 1 s, giving 8 s. At
    # our 32 kHz a slot is 0.512 s, so the WHOLE buffer is only 4.1 s and usable
    # pre-roll is ~3 s once live capture slots are reserved. Doubling the sample
    # rate halved the pre-roll budget; it is not a free parameter.
    pre_roll_s: float = 1.0
    hangover_s: float = 1.0      # keep recording this long after the last positive

    # Minimum recording length once the gate opens.
    # Non-zero buys two things that have nothing to do with birds: it stops the
    # filesystem thrashing on many tiny WAVs, and it gives a listener enough
    # context to identify what triggered.
    #
    # 20 s is the default, up from the 10 s of the first sweep. Measured on
    # `mfcc` at a >=98% retention target, 10 s -> 20 s costs 0.16 points of saving
    # (8.91% -> 8.75%) and buys a better worst recording (93.2% -> 94.1%), better
    # whole-box capture (95.7% -> 96.1%), a better species floor (94.6% -> 95.5%)
    # and 7% fewer files. Swept at *matched* retention it is free outright: 7.7%
    # saving at 98.0% either way, with the same +1.2 points on the worst
    # recording. See docs/lab-notebook.md §9.
    #
    # It is a preference, not an optimum — 30 s continues the same trend on
    # `mfcc` — and the trade is worth taking only because the binding metrics here
    # are the worst recording and the species floor, not the mean saving.
    min_on_s: float = 20.0

    # How often the classifier re-runs WHILE recording. 1.0 = every decision
    # window (continuous). Larger values duty-cycle the classifier during
    # recording, freeing CPU exactly when SD writes are competing for it — but
    # they also coarsen the close decision, since the gate can only shut at a
    # recheck instant. See gate.py: this interacts with hangover_s and you should
    # not pay for both.
    recheck_period_s: float = 1.0


@dataclass(frozen=True)
class GateConfig:
    """A whole experiment."""
    name: str = "carihc-baseline"
    sample_rate: int = SAMPLE_RATE
    frontend: FrontEndParams = field(default_factory=FrontEndParams)
    window: WindowParams = field(default_factory=WindowParams)
    gate: GateParams = field(default_factory=GateParams)
    # The adaptive threshold travels with the config so a manifest records the anchor
    # quantile, window and scale actually used — the most consequential numbers in the
    # system, and the ones an earlier audit found in no manifest at all (SUMMARY §12).
    adapt: AdaptParams = field(default_factory=AdaptParams)
    seed: int = 0
    # Negatives are ~44% of windows here but will be 90-98% in a real 24 h
    # deployment. Weight the loss toward the *deployment* prior, not this one.
    deployment_negative_prior: float = 0.95
