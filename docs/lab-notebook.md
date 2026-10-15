# BirdGate — an always-on birdsong gate for the AudioMoth

## 1. What this is for

**The goal is a deployment in the Scottish Highlands.** A rewilding project, AudioMoth
recorders left out for long unattended periods, and the problem that continuous recording
fills SD cards and buries the interesting audio in hours of wind and silence. BirdGate is
the classifier that sits in front of the SD write: *is a bird singing right now?* If yes,
record. If no, don't.

Everything else in this document exists to serve that. In particular:

> **The Sierra Nevada data is the test rig, not the target.**

SNE is here because it is 33 hours of continuous soundscape with 20,147 strong
time-frequency annotations — the only way to answer *can continuous field recordings train
a gate this small?* without waiting for labelled Highland audio. When the Scottish
recordings arrive, the front-end, the state machine, the metrics, the quantisation and the
export path carry over unchanged; only the **training set and the calibrated threshold** are
re-fit. That is the whole design constraint on how this code is written.

Consequence for every decision below: **prefer the choice that transfers.** Where a
parameter could be tuned to squeeze another 2% out of Sierra Nevada, and a different choice
would generalise better to an unseen forest with unseen species, take the second. The
absolute numbers on this dataset are not the deliverable; the *method that survives a change
of continent* is.

### The three things that must hold

1. **Don't miss birdsong.** Every other failure is recoverable — a false positive costs a
   few megabytes and can be filtered later. Audio the gate discards **does not exist**, and
   no amount of later cleverness recovers it. Recall is the constraint; saving is the
   objective subject to it.
2. **Repeatable.** Same config + same data → the same gate, byte for byte. Seeds pinned,
   splits deterministic, no un-recorded parameter anywhere in the path from audio to `.bin`.
3. **Traceable.** Any recording on any recovered card must resolve back to the exact model,
   threshold, training set and code that let it through. See §11 — this is a first-class
   requirement, not hygiene.

### Sibling project

An earlier, learning-first species-classification pipeline on the same dataset (not
published). BirdGate shares its data and its recording-level split — deliberately, so the
two are graded on the same six held-out recordings — and shares nothing else.

---

## 2. Glossary

Terms this document uses repeatedly. Defined here because they are jargon, not
because they are complicated.

| term | meaning |
|---|---|
| **arm** | One variant in a controlled comparison, from clinical-trial usage. G3 runs four arms; everything except the named difference is held identical — same windows, same recording split, same gate state machine, same metrics — so any difference in the result is attributable to the one thing that changed. |
| **control** | An arm that exists to test whether the headline method is actually responsible for the result. `bands` is the control that matters: if a bank of plain bandpass filters matches the cochlea, the cochlea is decoration. |
| **θ (theta)** | **The decision threshold on the classifier's output** — and the tuning knob of the whole system. The template-SVM emits one real number per window, `f(X) = Σ Qₚ·Φₚ(X) + b`, higher meaning more bird-like. `θ_on` is the score at which the gate opens and recording starts; `θ_off` is the lower value it must fall below to close again. **θ is not learned during training** — it is calibrated afterwards on the val recordings (G4). Lower θ → record more, miss fewer calls, more false positives. Raise it → record less. Lives in `GateParams`. |
| **hysteresis** | The gap between `θ_on` and `θ_off`. Without it, a score wobbling across a single threshold makes the gate chatter and chops a call into fragments. |
| **pre-roll** | Audio kept from *before* the gate opened, recovered from the ring buffer. You cannot detect a call before it starts, so without pre-roll every recording is missing its onset. |
| **hangover** | How long the gate stays open after the last positive score. |
| **duty cycle** | Fraction of wall-clock time actually written to the card. `saving = 1 − duty cycle`. |
| **retention** | Fraction of annotated bird-seconds that survived the gate. The headline recall metric. |
| **event recall @ k** | Fraction of annotation *boxes* with at least k% of their duration retained. More sensitive than retention, because retention is dominated by a few very long boxes. |
| **oracle** | The gate driven by ground truth instead of a model. Sets the ceiling, and separates state-machine cost from model cost. |
| **Φ (phi)** | The template vector — one number per filter channel, being that channel's rectified output summed over the decision window. This *is* the classifier's input. |
| **CAR-IHC** | Cascade of Asymmetric Resonators with Inner Hair Cells: the filterbank front-end. "IHC" is just a half-wave rectifier here. |
| **pole angle** | In `carfac.py`, a filter's resonant frequency expressed in radians per sample (`2πf/sr`). ⚠️ Written `pole_theta` in code and **unrelated to θ above**, which is a threshold. Lyon's papers call it θ; we do not, to avoid exactly this confusion. |
| **strong labels** | Annotations localised in time (and here frequency), as opposed to "this species is somewhere in this clip". The SNE dataset's 20,147 boxes are strong. |

---

## 3. Origin, and what was kept from it

The starting point was Sabbella et al., *An Always-On tinyML Acoustic Classifier for
Ecological Applications* (ISCAS 2022): a template-SVM over a CAR-IHC neuromorphic cochlea
running always-on inside an AudioMoth, classifying four species. What this project kept is
the **idea** — a classifier small enough to sit in front of the SD write, deciding per
ring-buffer slot whether that slot is written — and the device. Everything else was
re-derived here:

| | starting point | BirdGate |
|---|---|---|
| Task | 4-way species ID | binary: birdsong present / absent |
| Model | a few KB of weights over 30 cochlea channels | ~170 B — one score, `P`+2 weights and a bias; front-end chosen by a four-arm comparison (§8), which the cochlea lost |
| Training data | ~100 hand-curated clean clips per species | continuous soundscape with strong boxes |
| Negatives | a curated noise class | **every unannotated second of the same forest** — wind, stream, insects, rain, aircraft |
| Sample rate | 8 kHz train / 16 kS/s device | **32 kHz** (§4) |
| Threshold | fixed | adaptive, background-relative (§9) |
| Metric | accuracy | birdsong retention vs bytes written, per-species floor, battery-days (§10) |

The negatives are the real upgrade, and the part that most directly serves the Highland
deployment: a curated noise class is not the acoustic background a recorder sits in. Wind
in trees, running water and rain are what will actually fill a card in Scotland, and they
are only learnable from continuous field audio.

The cochlea front-end is kept as one arm (`carihc`) because it is the control that makes
the comparison mean something, not because anything downstream depends on it.

---

## 4. Decisions taken

**32 kHz on device (16 kHz Nyquist).** 8 kHz, the common choice for this class of device, would be wrong here.
Measured on our boxes: the 95th percentile of `high_hz` is 9,166 Hz and the 99th is
10,150 Hz. A 16 kHz device rate (8 kHz Nyquist) clips the upper band of roughly 10% of
annotations — disproportionately the high, thin, quiet calls a gate is already most likely
to miss. Scottish species (goldcrest, treecreeper, crested tit) sit at the *top* of that
range, so this margin matters more at the target site than at the test site. 32 kHz also
means **no resampling between training data and device**, removing a whole class of silent
bug. Cost is 2× filter compute and 2× SD write rate; §12 shows both fit.

**Python first, C scaffold alongside.** Full training/eval/calibration pipeline plus a
bit-honest device simulator in Python; `firmware/` holds a C reference and a generated
`model_params.h`, with a float-vs-fixed-point parity test. No AudioMoth firmware fork until
the method is shown to work.

**Four arms** (see §2 — an *arm* is one variant in a controlled comparison). `carihc` is the
headline; the other three are the controls that establish whether it earns its place. All four
share the same windows, the same recording split, the same gate state machine and the same
metrics, so any difference is attributable.

| arm | front-end | classifier | what it tests |
|---|---|---|---|
| `carihc` | CAR-IHC cascade, `P`=30-40 | linear SVM | the neuromorphic cochlea — the original headline, and what the controls test |
| `bands` | ERB-spaced bank of *independent* bandpass biquads, same `P` | linear SVM | **the control that matters.** Only the cascade-and-zeros differs. Does the *asymmetric resonator* do anything 40 plain bandpass filters wouldn't? |
| `mfcc` | MFCC, per-window mean and std of 20 coefficients | linear SVM | the standard front-end, usually rejected on cost for this class of device. Is that argument true at 32 kHz? |
| `cnn` | log-mel, int8 | tiny CNN, ~5-10 k params | **the usual premise for a tiny linear model is data scarcity** (rare species, ~50 clips). We have abundant labelled audio. Does the SVM's sparse-data advantage survive? |

⚠️ Arms 1-3 vary **only the front-end**, so a difference between them is attributable to the
front-end. `cnn` varies the front-end *and* the classifier, so if it wins it will not tell us
which change did it. That is deliberate — it asks "would a different approach be better?"
rather than "why?" — but it is weaker evidence and must be reported as such.

If `bands` ties `carihc`, ship the simpler thing — a plain biquad bank is easier to
implement correctly in fixed point, which matters more for a device left in a forest than
any accuracy tenth.

---

## 5. Design decisions that follow from the task

**The normaliser is not neutral for a presence detector.** A species-ID front-end normalises the summed
IHC outputs across channels — correct for species ID, where you want spectral *shape* and
not loudness, so a bird at 5 m and at 15 m classify alike. For "is anything there?", the
absolute level is the single most informative feature, and L1-normalising across channels
throws it away: silence and birdsong both become a unit-sum vector.

Plan: make the normaliser a swept axis (`none` / `l1` / `l2`), defaulting to `l1` **plus** an
appended `log(total energy)` and `log(spectral flatness)` — shape *and* level, for two extra
weights. Wind and running water are loud but spectrally flat and low-frequency, which is
exactly what a shape feature separates from a call. Both are Highland failure modes.

**The threshold is the product.** `f(X)` is a real-valued score. Everything a deployer tunes
— "record more, burn card" vs "record less, risk missing" — is one float in
`model_params.h`. Ship a table of calibrated operating points, not a number.

**Hysteresis, pre-roll, hangover.** A per-second hard threshold gives a flickering gate that
chops calls. The device needs a small state machine (`gate.py`, mirrored in `gate.c`): open
on `score > θ_on`, close only after the hangover expires below `θ_off < θ_on`, and flush the
pre-roll already sitting in the ring buffer on opening — a `read_pointer` rewind into audio
already captured, so it costs no extra memory. §6 shows it is also the most efficient recall
in the system.

⚠️ **Pre-roll is capped by the ring buffer, and 32 kHz halved that cap.** The buffer is
8 × 32 KB = 256 KB, the whole external SRAM. At 16 kHz a 32 KB slot is one
second, giving 8 s. At our 32 kHz a slot is 0.512 s, so the entire buffer is **4.1 s** and
usable pre-roll is **~3 s** once live-capture slots are reserved. Our 1 s recommendation is
safe, but this is not a free parameter and the §4 sample-rate decision paid for it.

---

## 6. G1 results ✅ — the oracle gate

Built and measured before any model exists: `windows.py` (1 s binary presence windows),
`gate.py` (the state machine), `intervals.py` (exact continuous-time overlap), `metrics.py`,
`provenance.py`, and `app.py` to see all of it. Ground truth is fed through the *real* state
machine, so this is simultaneously the ceiling and the control.

**The data.** 33 h wall clock; 16.26 h of birdsong (union of boxes) = 49.3% occupancy;
10,842 silent gaps, median **1.9 s**, 75th percentile 4.1 s. Per-recording occupancy ranges
**1.4% to 86.1%** — the 04:00 pre-dawn recordings are nearly empty, the dawn chorus nearly
saturated.

**Oracle gate, all 33 recordings, 1 s decision window:**

| pre-roll | hangover | retention | whole boxes kept | duty cycle | saving | per-species floor |
|---|---|---|---|---|---|---|
| 0 s | 0 s | 99.71% | 91.73% | 54.91% | 45.09% | 99.15% |
| 1 s | 0 s | 99.90% | 97.04% | 60.78% | 39.22% | 99.58% |
| **1 s** | **1 s** | **100.00%** | **100.00%** | **64.89%** | **35.11%** | **100.00%** |
| 1 s | 2 s | 100.00% | 100.00% | 67.91% | 32.09% | 100.00% |
| 1 s | 3 s | 100.00% | 100.00% | 70.12% | 29.88% | 100.00% |
| 3 s | 5 s | 100.00% | 100.00% | 74.87% | 25.13% | 100.00% |

**What this establishes:**

- **Set pre-roll = 1 s, hangover = 1 s and stop.** That is the knee: perfect retention,
  perfect whole-box capture, perfect per-species floor, at 35.1% saving. Every longer
  hangover in that table costs 3 percentage points of saving and buys *nothing* — the
  ground truth is already fully captured. **Any hangover beyond 1 s must be justified by
  model uncertainty, not by the audio.** Fixing this now means we can never later credit a
  state-machine change to the model.
- **Pre-roll is nearly free; hangover is what you pay.** 0→1 s of pre-roll costs 5.9 points
  of saving and lifts whole-box capture from 91.7% to 97.0%; it is the best-value parameter
  in the system, and on device it is a pointer rewind into audio already captured.
- **Whole-box capture is the sensitive metric, exactly as predicted.** At pre=0/hang=0,
  second-level retention reads a comfortable 99.71% while **8.3% of annotation boxes are
  clipped**. Second-level retention is dominated by the long tail (99th-pct box = 32 s);
  event recall weights a 0.3 s chip equally. Report both, always, and treat the box metric
  as the binding one.
- ⚠️ **Retention cannot reach 100% at pre=0/hang=0 — it caps at 99.71%.** That residue is
  the `min_overlap_s = 0.15 s` *labelling* floor, not the gate: a box clipping a window by
  under 0.15 s never labels it, so a sliver is dropped. It is a property of the labelling
  rule and every model number must be read against it, not against a naive 100%.
- **Occupancy varies 1.4%–86.1% across recordings, and that range is the transferable
  finding.** The mean is an artefact of which hours the SNE collectors chose to record. The
  near-empty 04:00 recordings are the closest analogue to a real 24-hour Highland
  deployment, and are where the threshold should be calibrated — see §14.

### The recording policy: minimum-on time and recheck cadence

A gate that decides afresh every second is the simplest design. The deployment we actually want is: wake on
birdsong, then **record a decent stretch**, rechecking periodically rather than continuously,
and stop when the birds do. Two parameters, both in `GateParams`:

- **`min_on_s`** — hold the gate open this long once triggered, regardless of score.
- **`recheck_period_s`** — how often the classifier re-runs *while recording*. The gate can
  only close at a recheck instant, so this also acts as a floor on closing latency.

Measured on the oracle. The right column pair is the one that isn't about birds at all:

**All 33 recordings (49% occupancy):**

| policy | retention | whole boxes | saving | files/hr | mean file |
|---|---|---|---|---|---|
| pre 1 s, hang 1 s (G1 knee) | 100% | 100% | **35.1%** | 109 | 21 s |
| pre 1 s, hang 1 s, recheck 5 s | 99.99% | 99.59% | 33.4% | 92 | 26 s |
| pre 1 s, min-on 20 s, recheck 5 s | 99.99% | 99.76% | 26.6% | 47 | 56 s |

**The four sparsest recordings (1.4–13% occupancy) — the closest analogue to a Highland
deployment, and the row that should drive the decision:**

| policy | retention | whole boxes | saving | species floor | files/hr | mean file |
|---|---|---|---|---|---|---|
| pre 1 s, hang 1 s (G1 knee) | 100% | 100% | **85.5%** | 100% | 49 | 11 s |
| pre 1 s, hang 1 s, recheck 5 s | 99.98% | 99.55% | 84.5% | 99.98% | 43 | 13 s |
| pre 1 s, min-on 10 s, recheck 5 s | 99.98% | 99.55% | 82.0% | 99.95% | 32 | 21 s |
| pre 1 s, min-on 20 s, recheck 5 s | 99.98% | 99.55% | 80.1% | 99.96% | 24 | 30 s |

**What this establishes:**

- **The policy is affordable.** The full scheme costs 5.4 points of saving on the sparse
  recordings (85.5% → 80.1%) and essentially nothing in recall: retention 99.98%, whole-box
  capture 99.55%, per-species floor 99.96%.
- **Rechecking at 5 s is nearly free — 1 point of saving — and it was the thing most at risk.**
  Sampling one second in five could in principle miss a bird singing only in the four
  unprocessed seconds. Measured, whole-box capture drops 100% → 99.55%. That is small enough
  to stop worrying about, and it is now measured rather than assumed.
- **`min_on_s` is the expensive knob; `recheck_period_s` is the cheap one.** They are not
  interchangeable, and `min_on_s` below `recheck_period_s` is a **no-op** — a 5 s minimum-on
  with a 5 s recheck is byte-identical to no minimum at all, because the recheck already
  holds the gate that long. Don't pay for both.
- **The real argument for `min_on_s` is not acoustic — it is files.** At the G1 knee the
  sparse recordings produce **49 separate recordings per hour averaging 11 seconds**. That
  thrashes FAT32 directory entries, multiplies file-open latency (the most expensive SD
  operation), and gives a human listener almost no context around each trigger. A 20 s
  minimum halves it to 24 files/hour at 30 s. **That is what the 5 points of saving buys**,
  and it is a real thing to buy.
- **20 s is not magic.** 10 s recovers nearly 2 points of saving for 32 files/hour at 21 s —
  probably the better default until the Highland site tells us how the files are consumed.

⚠️ **Open, and it decides the file question: how does the stock AudioMoth firmware write a
thresholded recording?** Later firmware already implements threshold-triggered recording
(an amplitude threshold, and in more recent versions a Goertzel frequency-domain trigger),
which is excellent news — the conditional-write path exists and we are replacing the
*trigger*, not restructuring the firmware. But we must confirm whether it emits **many small
timestamped WAVs** or **one WAV per period with the quiet parts excised**. The second breaks
traceability outright: offsets within the file no longer map to wall-clock time, so "when did
this bird sing" becomes unanswerable. If it behaves that way, we need the other
device: a **separate detection log with timestamps** written alongside the audio. Verify
against the exact firmware version before G6.

**Measurement note.** All of this is exact continuous-time interval arithmetic
(`intervals.py`), not a rasterised grid. Our shortest boxes are ~0.1 s, so grid quantisation
would have been a double-digit percentage of the quantity being measured. Retention is
computed against the **union** of boxes, never their sum — birds sing over each other, and
summing durations double-counts the overlap.

---

## 7. G2 results ✅ — the CAR-IHC front-end

### The stage is a biquad, and that buys two implementations

Deriving the CARFAC coupled-form stage symbolically gives

    H(z) = g · (1 + (h·r·c0 − 2·r·a0)·z⁻¹ + r²·z⁻²) / (1 − 2·r·a0·z⁻¹ + r²·z⁻²)

with `g = (1 − 2a0r + r²)/(1 − 2a0r + c0hr + r²)` for unity DC gain — which reproduces
Lyon's `CARFAC_Stage_g` exactly, derived independently. So:

- `cascade()` uses `scipy.signal.sosfilt` — **25× faster**, used for all training.
- `cascade_stepwise()` is the literal sample-by-sample coupled form the firmware mirrors.
- They agree to **5×10⁻¹⁵ relative** (machine precision), asserted in the test suite.

⚠️ **The firmware must use the coupled form, not the biquad.** At the low channels `r ≈ 0.996`,
so direct-form denominator coefficients are ill-conditioned and will not survive Q15. The
coupled form stores a rotation — `a0`, `c0` are a sine/cosine pair, both ≤ 1 — and is
well-behaved. This is a G5 decision made early, in G2, because it is a *design* constraint.

### Validation — 16 tests, all passing

The front-end is tested in isolation against signals whose answer is derivable by hand,
because an IIR cascade degrades silently and the loss gets blamed on the classifier:

| test | result |
|---|---|
| tone at channel p's pole peaks at channel p | ✅ offset 0 or +1 for all 40 |
| best frequency monotonic across channels | ✅ strictly, 12 kHz → 150 Hz |
| response asymmetric (steeper above than below) | ✅ — the "A" in CAR |
| rising chirp sweeps channels monotonically | ✅ ch 37 → 5, no reversal |
| every stage stable (r < 1) | ✅ r ∈ [0.827, 0.996] |
| unity DC gain per stage, and through 40 stages | ✅ to 1e-12 |
| silence → exact zero, no NaN | ✅ |
| impulse response decays | ✅ |
| sosfilt path == coupled form | ✅ 5e-15 |
| chunked == unchunked templates | ✅ (no seam glitch on long files) |

Measured best frequency sits **~7.5% above** the design pole. That is not an error: channel
p's *effective* response is the product of stages 0..p, and the stages above have already
attenuated, which shifts the peak up. Documented so nobody "fixes" it.

⚠️ **Two different "ERB" definitions are in play and they are not interchangeable.** Channel
*spacing* uses the Glasberg & Moore ERB-number scale; the damping constants use Lyon's
bandwidth formula `(f + 165.3)/9.2645`, which has G&M's slope but a lower intercept (17.84 vs
24.70) — identical above ~1 kHz, 19% apart at 100 Hz. A test pins this so that "correcting"
one to match a textbook cannot silently retune the whole bank.

### Cost — it fits, comfortably

| | channels | MMAC/s | % of a 48 MHz M4F | state+coeffs+weights |
|---|---|---|---|---|
| `carihc` | 40 | 9.0 | **19%** | 1,292 B |
| `bands` | 40 | 7.7 | 16% | 1,292 B |
| `carihc` | 30 | 6.7 | 14% | 972 B |

Against the 8 KB SRAM / 100 KB flash targets in §12, this is not close to the limit. Compute was
the parameter at risk after the 32 kHz decision; it is not a problem.

### Does Φ carry presence signal before any training?

Checked on `SNE_023` — a deliberately hard hour at **13.1% occupancy**, closest in character
to a Highland deployment. Single-feature AUC against the binary label, no model fitted:

| | best single channel | at | channels with AUC > 0.65 | log-energy | log-flatness |
|---|---|---|---|---|---|
| `carihc` | **0.847** | 3,284 Hz | 25 / 40 | 0.621 | **0.503** |
| `bands` | **0.857** | 3,612 Hz | 28 / 40 | 0.605 | 0.778 |

**Three findings:**

- **One normalised channel already reaches AUC 0.85.** The front-end plainly carries the
  signal; G3's job is combining 42 of these, which is a much easier problem than starting
  from nothing. This substantially de-risks the whole approach.
- ⚠️ **`bands` is not worse — it is marginally better here.** Early evidence that the cascade
  and its asymmetric zeros may be decoration for *this* task. Hold it loosely: single-feature
  AUC is not the model, and the SVM sees all 42 jointly. But it is exactly why the control
  exists, and if G3 confirms it we ship the simpler filterbank, which is also the easier one
  to get right in fixed point.
- ⚠️ **`log_flatness` works for `bands` (0.778) and is useless for `carihc` (0.503).** This is
  explainable rather than mysterious: every stage has unity DC gain, so low-frequency energy
  appears in *all* cascade channels and the across-channel profile is strongly correlated,
  which flattens the geometric/arithmetic ratio into near-constancy. The feature was added
  specifically to reject wind and rain at the Highland site, so **it needs redefining for the
  cascade** — most likely computed over adjacent-channel *differences* rather than levels.
  Carried into G3 as a concrete task.

Throughput: ~21 s per hour of audio per arm, so a full 33 h feature cache is ~12 min per arm.

---

## 8. G3 results ⚠️ — the arms, and a problem

Three of four arms trained. Everything held identical: windows, recording split, the G1
gate knee (1 s pre-roll, 1 s hangover), threshold rule, metrics. **`cnn` is now trained —
see §9b**, which supersedes this section's ranking. §8 and §9 are kept because the *route* to
the answer — a fixed threshold failing, then being fixed — is most of what was learned.

### The cochlea loses. The simplest arm wins.

Duty cycle needed to reach a given birdsong retention, test split (lower is better):

| arm | window AUC | AP | duty@95% | duty@96% | duty@97% | duty@98% | MMAC/s | % of M4F |
|---|---|---|---|---|---|---|---|---|
| `carihc` | 0.80 | 0.87 | 82.8% | 86.8% | 91.3% | 94.7% | 9.0 | 18.7% |
| **`bands`** | **0.84** | **0.90** | **77.3%** | **83.5%** | **88.2%** | **92.8%** | 7.7 | 16.0% |
| `mfcc` | 0.83 | 0.89 | 80.4% | 84.6% | 89.1% | 93.4% | **1.5** | **3.2%** |

**`bands` beats `carihc` at every operating point, on every metric.** The cascade of
asymmetric resonators — the cochlea's defining feature — is not merely unnecessary here, it
is *worse* than a plain bank of independent ERB-spaced bandpass biquads at the same channel
count, same rectification, same window summing, same normaliser. This is the control doing
exactly the job it was built for. Ship `bands`: better, cheaper, and far easier to get right
in Q15 (no 40-stage error compounding — the G5 risk largely evaporates).

⚠️ **The usual cost argument against MFCC does not survive our parameters either.** MFCC
runs at **1.5 MMAC/s vs the cascade's 9.0** — 3.2% of a 48 MHz M4F against 18.7% — because
the cascade's cost scales with the *sample rate* (32 kHz) while the FFT's scales with the
*frame rate* (100 fps). At 8 kHz with 30 channels the balance is different; at
ours it inverts. And `mfcc` also out-ranks `carihc` on accuracy. (Caveat: MACs undercount
`log()`/DCT, which are slower per-op on an M4F, so treat the 6× as an upper bound on the
advantage.)

### ⚠️ But the real finding is that a single threshold does not transfer across recordings

At one θ chosen on val for 96% retention, applied per test recording (`bands`):

| recording | occupancy | retention | duty | saving | oracle saving |
|---|---|---|---|---|---|
| SNE_032 | 31.4% | **54.9%** ❌ | 22.5% | 77.5% | 49.3% |
| SNE_016 | 36.3% | 97.9% | 59.2% | 40.8% | 45.8% |
| SNE_030 | 56.9% | 99.3% | 91.2% | 8.8% | 22.6% |
| SNE_012 | 59.4% | 96.0% | 79.5% | 20.5% | 21.6% |
| SNE_021 | 60.0% | 94.5% | 78.1% | 21.9% | 16.4% |
| SNE_004 | 76.1% | 97.7% | 89.5% | 10.5% | 11.1% |

**Retention swings from 54.9% to 99.3% at a fixed threshold.** On SNE_032 the gate would have
thrown away nearly half the birdsong while congratulating itself on a 77.5% saving. The
aggregate 96% hides a recording where the device would have failed outright — and in the
field that failure is silent and unrecoverable, which is precisely the risk §10 names as the
main danger of this whole idea.

This is the thing to fix before anything else, and it is more important than the arm choice:

- **Hypothesis:** recording-to-recording differences in gain, mic, and background level move
  the score distribution bodily, so one absolute θ means different things in different hours.
  The linear CAR has **no AGC** — we deliberately dropped CARFAC's automatic gain control to
  keep the front-end linear — and this is exactly the symptom its absence would produce.
- **Cheap fixes to measure first:** per-recording (or rolling-window) score standardisation;
  θ set as a *quantile* of a trailing score window rather than an absolute; per-channel
  background subtraction before the template sum.
- **Why it matters more for Scotland than here:** a single AudioMoth at a fixed gain in one
  location has less between-file variance than 33 recordings across a season — but weather,
  season and battery voltage all move levels over a months-long deployment, and there is no
  one to notice.

### Absolute performance is weak, and that is honest

At 98% retention on this test split, the best arm still records 92.8% of the time — a 7%
saving where the oracle achieves 35%. The test recordings average 59.6% occupancy, so the
headroom was never large (§6), but the model captures only about half of what is there.
`bands` at 96% retention saves 18.4% against an oracle 35% — **roughly half of the achievable
saving**. Reported as-is; the Highland deployment's much lower occupancy is where this gets
interesting, and the SNE_032 failure is what stands between here and there.

---

## 9. G3+ results ✅ — adaptive thresholding fixes the transfer problem

**A gain control, done differently.** The linear CAR has none; CARFAC does, and we
dropped it to keep the front-end linear. §8 showed the cost of that decision. The fix
below acts on the score rather than the audio, so it is front-end-agnostic.

**The fix is CFAR** (constant false-alarm rate, from radar): threshold the score *relative to
the local background* rather than absolutely.

    z_t = (s_t - b_t) / d_t     b_t = trailing background level, d_t = its spread

then apply one global θ to `z`. When it rains, `b_t` rises and the gate demands a
correspondingly louder bird — no retraining, no site survey.

⚠️ **Anchor on a LOW quantile (0.3), never the median or higher.** A threshold set at the
q-th percentile of recent scores is a *duty-cycle controller*, not a detector: it records q%
of the time regardless, so continuous song would still be q% recorded and pure silence would
still trigger q% of the time. 0.3 keeps the anchor inside the background population even at
SNE's 76%-occupancy hours.

### It works, and the trade is small

`bands`, test split, θ matched to give the same overall retention:

| threshold mode | device cost | saving @96% | **worst recording @96%** |
|---|---|---|---|
| fixed (G3) | 0 B | **16.9%** | **67.5%** ❌ |
| **ema (α=0.03)** | **8 B** | 14.2% | **90.4%** |
| quantile, 300 s | 1.2 KB | 12.9% | 91.9% |
| quantile, 120 s | 480 B | 12.3% | 92.8% |

At a fixed θ chosen on val, per-recording retention ran **54.9% → 99.3%, a 44.4-point
spread**. With the trailing-quantile tracker that collapses to **4.7 points**. The catastrophic
recording (SNE_032, 54.9%) comes up to 92.8%.

**Recommendation: `ema` with α = 0.03.** It gives up only 2.7 points of saving against the
fixed threshold, lifts the worst recording from 67.5% to 90.4%, and costs **two floats and
four operations per second** — against 480 B of ring buffer for the quantile version. On a
device that will sit unattended for months, bounded worst-case behaviour is worth far more
than 2.7 points of card. Validated on a synthetic rain step: the fixed threshold's trigger
rate trebles when the noise floor jumps, `quantile` and `ema` hold it flat.

⚠️ **α is a real tuning risk.** Adapting fast enough to track weather (α = 0.03, ~30 s) is
also fast enough that a long continuous chorus starts inflating its own background. The
background updates 10× slower while the gate is open, which mitigates but does not remove it.
Watch for it on dense Highland dawns.

### The recording algorithm is already optimal — the shortfall is the model

Suspicion was that a long minimum-on time or hangover was wasting the saving on dense
recordings. Measured, driving the **real model** rather than the oracle — saving at 96%
retention:

| pre-roll ↓ / hangover → | 0 s | 1 s | 2 s | 3 s | 5 s |
|---|---|---|---|---|---|
| 0 s | 9.7% | 11.0% | 10.9% | 10.3% | 10.0% |
| **1 s** | 11.3% | **14.0%** | 13.1% | 12.2% | 11.2% |
| 2 s | 11.2% | 13.6% | 12.4% | 11.9% | 11.4% |

**The optimum is pre-roll 1 s, hangover 1 s — exactly the G1 oracle knee.** The state machine
needs no re-tuning, and the gap to the oracle is the model's discrimination, not the
recording policy. Note the sign flips versus the oracle: for a *perfect* detector hangover 0 s
maximised saving, but for a real one hangover 1 s wins, because it bridges momentary score
dropouts inside a call that the oracle never has. G3 used no minimum-on time, so that was
never the cause.

### Re-run with EMA and a 10 s minimum-on

The G3 arm ranking was measured under the broken fixed threshold, so it was re-run with the
G3+ defaults. Matched at **96% retention**, test split:

| arm | saving | **worst recording** | whole boxes | files/hr | mean file | MMAC/s |
|---|---|---|---|---|---|---|
| `carihc` | 10.5% | **80.5%** | 90.9% | 96 | 33 s | 9.0 |
| **`bands`** | **14.2%** | **90.3%** | 90.9% | 128 | 24 s | 7.7 |
| `mfcc` | 11.8% | 88.9% | 90.6% | 120 | 26 s | 1.5 |

**The ranking is unchanged: `bands` wins, `carihc` is last.** Adaptive thresholding did not
rescue the cochlea — it is now behind on saving *and* has the worst per-recording floor
(80.5%). The conclusion from §8 stands and is now measured under the better threshold policy.

### `min_on_s` is free up to 10 s

`bands`, matched 96% retention:

| min-on | saving | worst rec | whole boxes | files/hr | mean file |
|---|---|---|---|---|---|
| 0 s | 14.1% | 90.4% | 91.5% | 158 | 20 s |
| 5 s | 14.2% | 90.9% | 91.1% | 152 | 20 s |
| **10 s** | **14.2%** | 90.3% | 90.9% | **128** | **24 s** |
| 20 s | 12.1% | 90.9% | 92.5% | 83 | 38 s |
| 30 s | 11.2% | **93.5%** | **94.0%** | 62 | 51 s |

**Up to 10 s the minimum-on time costs nothing** — identical saving to 0 s, with 19% fewer
files. That is a free improvement in filesystem behaviour and listenability, so **10 s is the
default**. Past that the price is roughly 2 points of saving per extra 10 s, but it is not
pure loss: at 30 s the worst-recording floor rises to 93.5% and whole-box capture to 94.0%,
because longer recordings capture complete calls. If the Highland site turns out to be
card-rich and attention-poor, 30 s may be the better buy. Tunable in the app.

### ⚠️ Correction: do not divide by a *local* spread

The first version of the adaptive threshold did `z = (s - b)/d` with **both** `b` and `d`
estimated in the trailing window. Spotted by eye in the app: on SNE_026 — a **training**
recording whose first 45 minutes are 0.1% bird — the gate kept switching on through the
silence.

The model was not at fault. On that recording the raw score separates cleanly: **AUC 0.925**,
silence at −0.27 ± 0.15, birds at +1.13. The normaliser destroyed it:

| normalisation | recording AUC | silence z max | bird z mean |
|---|---|---|---|
| raw / fixed | **0.925** | +0.78 | +1.13 |
| ÷ trailing MAD (`ema`) | 0.843 | +12.6 | +10.5 |
| ÷ trailing MAD (`quantile`) | 0.739 | **+337** | +3.7 |

**Mechanism:** in a long quiet stretch the local spread collapses (0.084 here), so dividing
by it amplifies meaningless fluctuation into huge z, and the tail of silence overtakes real
birds. `min_on_s` then turns each spurious blip into ten seconds of recording. The symptom is
a gate that chatters through precisely the silence it exists to skip.

**What varies between recordings is the offset — gain, mic, background level — not the
spread of the score.** So: subtract the trailing background, divide by **one global constant
fitted on train** (which travels with the model into `model_params.h`).

Sweeping the anchor quantile and window, measured as the fraction of SNE_026's 45-minute
silence that gets recorded:

| config | saving | worst recording | silence recorded |
|---|---|---|---|
| fixed | 18.4% | 70.1% | **54.4%** |
| q120 s, anchor 0.2, global | 16.3% | 86.4% | 3.0% |
| **q300 s, anchor 0.2, global** | **16.1%** | **88.6%** | **5.2%** |
| q300 s, anchor 0.3, global | 15.3% | 89.4% | 17.1% |
| q300 s, anchor 0.5, global | 7.7% | 90.4% | **75.0%** |
| q300 s, anchor 0.2, ÷local MAD | 14.2% | 87.5% | 20.4% |

**The anchor quantile is the single most consequential number in this module**, and the
0.5 row is the duty-cycle-controller failure predicted above arriving exactly as described:
anchor at the median and the gate records three-quarters of pure silence regardless of
content. **Defaults: `quantile`, 300 s, anchor 0.20, `scale="global"`.**

On SNE_026 itself the fix is dramatic — `bands` now saves **77.5%** of that hour against an
oracle ceiling of 79.8%, i.e. **97% of what a perfect detector achieves**, where the
local-spread version managed 5.3%.

### Final arm table (corrected thresholding, min-on 10 s, θ on val for ≥98% retention)

| arm | retention | worst rec | whole boxes | saving | species floor | MMAC/s |
|---|---|---|---|---|---|---|
| `carihc` | 97.1% | **83.4%** | 94.8% | 6.6% | 92.7% | 9.0 |
| **`bands`** | **98.1%** | 92.6% | **96.2%** | 8.4% | **96.0%** | 7.7 |
| `mfcc` | 97.8% | **94.7%** | 94.4% | **9.5%** | 95.8% | **1.5** |

`carihc` remains last on every safety metric. `bands` and `mfcc` are close; `mfcc` is 5×
cheaper and has the best worst-recording floor, `bands` the best whole-box capture and
species floor. Either is a defensible ship; the cochlea is not.

### Window length and score pooling ✅

**Longer feature windows are decisively worse.** Tested by deriving T-second templates as
sums of consecutive 1 s templates (exact, since Φ is a sum), with the decision rate held at
1 Hz so window length is isolated from decision granularity:

| feature window | saving | worst rec | silence recorded |
|---|---|---|---|
| **1 s** | **15.4%** | 88.9% | **12.6%** |
| 2 s | 13.6% | 92.4% | 52.6% |
| 5 s | 5.3% | 92.1% | 75.0% |

This is the dilution effect: 85% of annotation boxes are shorter than 5 s, so a longer window
buries the call in background. The apparent gain in worst-recording retention is an artefact
of recording nearly everything. **A 5 s window is also physically impossible** — it needs >5 s
of pre-roll to avoid clipping onsets, and the ring buffer holds 4.1 s at 32 kHz.

**Score pooling over 1 s features does help, via the opposite mechanism to the obvious one:**

| pooling | saving | worst rec | whole boxes | silence recorded |
|---|---|---|---|---|
| none | 15.4% | 88.9% | 93.4% | 12.6% |
| **max-3** | 15.0% | 87.7% | **94.6%** | **4.5%** |
| max-5 | 13.4% | 86.7% | 95.9% | 5.6% |
| mean-3 | 11.1% | 86.4% | 94.2% | 27.5% |
| median-3 | 10.4% | 88.6% | 93.3% | 53.1% |

⚠️ **`mean`/`median` look like the principled choice and measure worse.** Requiring sustained
evidence suppresses short calls, so θ must come *down* to keep recall, and recorded silence
rises to 27-53%. `max` does the reverse: it makes the score more sensitive, so the same recall
is reachable at a **higher** θ — and the higher θ is what rejects silence. Not redundant with
`hangover_s`, which only extends an already-open gate. **Default: `max`, n=3** (three floats).

### Final G3 table — corrected thresholding, max-3 pooling, min-on 10 s

| arm | retention | worst rec | whole boxes | saving | species floor | MMAC/s |
|---|---|---|---|---|---|---|
| `carihc` | 96.9% | **81.1%** | 95.3% | 7.8% | 92.2% | 9.0 |
| `bands` | **98.1%** | 91.3% | **97.2%** | 8.1% | **95.8%** | 7.7 |
| `mfcc` | 97.9% | **93.2%** | 95.7% | **8.9%** | 94.6% | **1.5** |

Per-recording retention, θ from val:

| recording | occupancy | `carihc` | `bands` | `mfcc` |
|---|---|---|---|---|
| SNE_032 | 31.4% | **80.7%** | 89.5% | **93.1%** |
| SNE_016 | 36.3% | 99.7% | 99.4% | 99.3% |
| SNE_030 | 56.9% | 98.1% | 97.7% | 97.4% |
| SNE_012 | 59.4% | 97.7% | 97.5% | 97.3% |
| SNE_021 | 60.0% | 98.9% | 99.2% | 98.8% |
| SNE_004 | 76.1% | 98.5% | 99.0% | 98.5% |

**`carihc` is clearly last** — its 81.1% floor is 10 points below the others and concentrated
entirely on the hardest recording. **`bands` vs `mfcc` is genuinely close, not a rout**: bands
wins 5 of 6 recordings but by 0.1-0.4 points, while mfcc wins the one recording that matters
(the hard, sparse SNE_032) by 3.6 points and has the better worst-case floor. `mfcc` is also
**5x cheaper**. Bands leads on whole-box capture and species floor.

⚠️ **Seed variance is exactly zero** — LinearSVC's objective is convex, so it converges to the
same optimum regardless of seed. Good for repeatability (§11), but it means seeds give **no
error bars**. The dominant uncertainty is the recording split, and we have n=1 on that. Do not
read the bands-vs-mfcc gap as established; a different 6-recording holdout could reorder them.

### Defaults locked in ✅ — and one bug that made the first `min_on_s` sweep wrong

Everything G3+ measured is now the default in code, not just the recommendation in
this document: `AdaptParams(mode="quantile", window_s=300, bg_quantile=0.20,
scale="global", agg="max", agg_n=3)` in `adapt.py`, matched by the app's sliders.
The sliders stay — the point of the compare tab is that these are arguable — but
opening the app now shows the configuration the numbers below were measured at.

⚠️ **`min_on_s` moves from 10 s to 20 s, and the earlier 10 s sweep was measured
with a bug.** `train.pick_theta` read the module-level `GATE`, so passing
`run(gate=...)` changed the state machine used for the **test** evaluation but not
the one used to **select theta** on val. Every `min_on_s` row in the "free up to
10 s" table above was therefore scored at a theta chosen for a 10 s gate. Fixed —
`pick_theta` now takes the gate it will be reported under, and refuses to be
confused about it in the docstring. Re-measured, θ on val for ≥98% retention:

| arm | min-on | retention | worst rec | whole boxes | saving | species floor | files/hr |
|---|---|---|---|---|---|---|---|
| `mfcc` | 0 s | 97.6% | 92.1% | 95.2% | 8.9% | 95.0% | 39.2 |
| `mfcc` | 10 s | 97.9% | 93.2% | 95.7% | 8.9% | 94.6% | 37.0 |
| **`mfcc`** | **20 s** | **98.0%** | **94.1%** | **96.1%** | **8.8%** | **95.5%** | **34.5** |
| `mfcc` | 30 s | 98.0% | 95.2% | 96.5% | 9.1% | 91.6% | 31.5 |
| `bands` | 10 s | 98.1% | 91.3% | 97.2% | 8.1% | 95.8% | 18.2 |
| `bands` | 20 s | 97.7% | 89.6% | 96.5% | 12.1% | 95.2% | 22.2 |
| `bands` | 30 s | 97.1% | 89.0% | 95.8% | 13.6% | 85.9% | 26.3 |

On `mfcc` the 20 s row is better than the 10 s row on **every** metric except 0.16
points of saving, so it is close to free and it is the default. The apparent cost
of a long minimum-on in the old table was mostly the mismatched theta.

Confirmed separately by sweeping theta directly on test — a diagnostic, not a
headline, since it selects on the split it reports — which removes val→test
calibration noise from the comparison. Saving at **matched** 98.0% retention:

| min-on | `mfcc` saving | `mfcc` worst rec | `bands` saving | `bands` worst rec |
|---|---|---|---|---|
| 0 s | 6.4% | 93.5% | 7.2% | 91.8% |
| 10 s | 7.7% | 93.2% | 7.8% | 91.4% |
| **20 s** | **7.7%** | **94.4%** | **8.0%** | **92.3%** |
| 30 s | 8.7% | 95.4% | 7.8% | 92.3% |

At equal retention 20 s costs nothing at all against 10 s and buys about a point
on the worst recording, for both arms. It is still a **preference, not an
optimum** — 30 s keeps improving `mfcc`'s worst recording — and it is worth taking
only because the binding metrics here are the worst recording and the species
floor rather than the mean saving (§10).

⚠️ **`bands` is unstable under this sweep and `mfcc` is not.** At 20 s and 30 s the
`bands` rows come in *below* the 98% target they were selected for (97.7%, 97.1%)
and its species floor collapses to 85.9% at 30 s, while its saving apparently
jumps to 13.6%. That is not a better model; it is a theta chosen on val landing in
a worse place on test. It is one more small piece of evidence for `mfcc` over
`bands` — the two are within noise of each other on accuracy (§9), but `mfcc`
calibrates more predictably, and it is 5x cheaper.

---

## 9b. G3 results ✅ — the CNN arm, and the data-scarcity premise does not survive

The fourth arm, and the only one that tests the *argument* for a tiny linear model
rather than a front-end. The usual case for one on this class of device is **data
scarcity** — a few dozen clips per class — and we have ~119,000 labelled seconds and a
binary target. So: with abundant data, does the SVM's advantage survive?

**It does not.** At matched retention the CNN saves roughly **60% more card** than
the best SVM arm and has a better worst-recording floor. But it loses on the two
things that actually bind — **working SRAM** and the **per-species floor** — so
the answer is not "ship the CNN", it is "the small model is right for the wrong reason".

### What was built

`birdgate/cnn.py`, `scripts/train_cnn.py`, six tests. The input is the full
32 x 100 log-mel patch, which is the one thing this arm changes: every other arm
sums the window to a vector before the classifier sees it, throwing away all time
structure inside the second.

    (1, 32, 100)  log-mel dB, globally standardised on train
      conv 3x3 x8,  stride (1,2) -> BN -> ReLU -> maxpool 2x2
      conv 3x3 x16              -> BN -> ReLU -> maxpool 2x2
      conv 3x3 x32              -> BN -> ReLU -> maxpool 2x2
      global average pool -> 32 -> linear -> 1 logit

**5,865 deployable parameters**, BatchNorm folded away, CMSIS-NN-compatible ops
only. Everything outside the model is imported from `scripts/train.py` — same
windows, same split, same adaptive threshold, same theta-on-val rule, and the same
`report()` function, which was factored out of `train.run` precisely so the arms
cannot drift apart in their metric code.

### The headline table — all four arms, θ on val for ≥98% retention

| arm | retention | worst rec | whole boxes | saving | species floor | files/hr | MMAC/s | % M4F | model |
|---|---|---|---|---|---|---|---|---|---|
| `carihc` | 97.3% | **82.1%** | 95.9% | 7.8% | 92.7% | 21.7 | 9.0 | 18.7% | 172 B |
| `bands` | 97.7% | 89.6% | **96.5%** | 12.1% | 95.2% | 22.2 | 7.7 | 16.0% | 172 B |
| `mfcc` | 98.0% | 94.1% | 96.1% | 8.8% | **95.5%** | 34.5 | **1.5** | **3.2%** | 164 B |
| `cnn` | 97.7% | 94.3% | 95.0% | **14.1%** | **90.4%** ❌ | 52.2 | 3.2 | 6.6% | 5.9 KB |
| **`cnn-int8`** | **98.3%** | **97.1%** | 96.4% | 12.2% | 93.7% | 48.2 | 3.2 | 6.6% | 5.9 KB |

### Matched retention, which is the comparison that means something

The table above compares arms at whatever retention each happens to land on, and
near 98% the trade is steep enough that a point of retention is worth more than
the entire gap between arms. Sweeping θ on test — **a diagnostic, not a headline,
since it selects on the split it reports** — isolates the model from that
calibration noise:

| arm | saving @97.5% | saving @98.0% | saving @98.5% | worst rec @98.0% |
|---|---|---|---|---|
| `carihc` | 6.2% | 4.3% | 3.3% | 86.1% |
| `bands` | 11.7% | 8.0% | 7.4% | 92.3% |
| `mfcc` | 9.7% | 7.7% | 5.7% | 94.4% |
| `cnn` | 13.9% | **12.7%** | **11.2%** | 96.8% |
| **`cnn-int8`** | **14.3%** | 12.2% | 11.5% | **97.1%** |

**At 98% retention the CNN saves 12.2-12.7% where the best SVM arm manages 8.0%**,
and its worst recording holds at 97.1% against 92.3-94.4%. The gap widens as the
retention target rises, which is the direction that matters: at 98.5% the CNN
still saves 11.5% while `mfcc` is down to 5.7%. This is not a marginal win.

Window-level discrimination on the normalised score agrees but understates it —
AUC `cnn` 0.84, `bands` 0.83, `carihc` 0.80, `mfcc` 0.79 — which is §10's point
restated: the arm that ranks windows best is not automatically the arm that gates
best, and only the post-gate numbers decide. (These AUCs are computed on the
background-normalised `z`, not the raw margins of §8, so they are not comparable
to that table.)

### int8 is free

`cnn-int8` is within seed noise of the float model on every metric, and better on
several. Post-training quantisation — BN folded, per-output-channel int8 weights,
per-tensor int8 activations calibrated at the 99.9th percentile of a train batch —
costs nothing measurable here. ⚠️ `cnn.quantise` **simulates** the int8 grid and
computes in float; bit-exact CMSIS-NN parity is still G5's job, on the same terms
`test_parity.c` gives the gate.

### ⚠️ Seed variance is real, unlike every other arm

LinearSVC is convex, so §9 could report that seed variance is exactly zero. The
CNN has no such guarantee, so it was run at three seeds:

| arm | retention | worst rec | whole boxes | saving | species floor |
|---|---|---|---|---|---|
| `cnn` | 97.75 ± 0.22 | 95.09 ± 1.21 | 95.65 ± 0.66 | 13.47 ± 0.57 | **88.92 ± 3.87** |
| `cnn-int8` | 97.88 ± 0.45 | 95.68 ± 1.28 | 95.94 ± 0.60 | 12.91 ± 0.62 | 91.73 ± 1.96 |

The saving advantage (13.5% vs 8.0%) is about nine times its own standard
deviation, so it is not a seed artefact. The **species floor is another matter**:
±3.9 points on the float model, with seed 1 landing at 84.5%.

**Re-earned 2026-09-22 for the ported arm** — the one the firmware runs, on the causal
`logmel-dev` front-end and without the two artefact quantisers (ROADMAP §4):

| arm | retention | worst rec | saving | species floor |
|---|---|---|---|---|
| `cnn-int8` on `logmel-dev` | 97.82 ± 0.64 | 94.95 ± 1.90 | 13.28 ± 1.34 | **94.37 ± 2.59** |

Causality cost this arm nothing and helped the column that decides it: the floor is
~2.6 points above the centred-front-end figure above, though still the widest-spread
number here and still below `mfcc`'s 96.22. Seeds land at 95.29, 96.37 and 91.45, and
the species setting the floor changes between them, which is the same thin-measurement
problem §13 G7b is about rather than a property of the model.

### ⚠️ The two things that actually stop it

**1. Per-species retention floor — §10 names this the main risk of the whole
idea.** The CNN is the *worst* arm on it: 88.9 ± 3.9 (float) and 91.7 ± 2.0
(int8), against 95.5 for `mfcc` and 95.2 for `bands`. It saves more card and holds
the worst *recording* better while dropping more of the rarest *species*, which is
the exact failure that is silent in the field and unrecoverable afterwards. A gate
that loses 15% of a scarce species' calls to buy 4 points of card is a bad trade
for this project, whatever the headline says.

**2. ~~Working SRAM — 23.9 KB against a <8 KB target~~ — SOLVED, see §9c.** The
first version of this section called 23.9 KB the blocker. It was measuring the
wrong thing: that is what the network costs when each feature map is built whole,
which is what a training framework does and has nothing to do with what a device
must do. Streamed, the same network — bit-identical output — needs **7.1 KB**.

So the species floor is the **only** remaining objection, and it is enough on its
own: a gate that loses 15% of a scarce species' calls to buy 4 points of card is a
bad trade for this project, whatever the headline says. Class-balanced or
species-aware training is the obvious first attack and has not been tried.

### ⚠️ Apple MPS trains this network worse, and does not replicate

Worth recording because it cost an hour and would silently corrupt any conclusion
drawn from a GPU run. One epoch, seed 0, identical config:

| device | loss | val AP |
|---|---|---|
| **cpu** | **0.4825** | **0.9288** |
| cpu, repeat | 0.4825 | 0.9288 |
| mps | 0.6436 | 0.8327 |
| mps, repeat | 0.6699 | 0.7768 |
| mps, repeat | 0.6187 | 0.9162 |

Two failures at once: MPS does not reproduce (Metal fixes no reduction order, so
the same seed in the same process diverges) and it converges somewhere worse.
Gradient clipping is not the cause — the spread is identical with it off. Not
chased further: the run is 20 minutes on CPU and `CnnParams.device` now defaults
to `"cpu"`, because §11 asks for numbers that replicate and MPS cannot give one.

### So what does this mean for the tiny-linear-model case?

The *conclusion* — use a tiny linear model over a cheap filterbank — is
the right call on this hardware. The *usual reason* is not: data scarcity is not
why the SVM should win here, because with 33 hours of annotation the CNN clearly
out-detects it. The real reasons are different ones — a
convolution's activation memory, and, on this data, a per-species floor that the
linear arms hold better. `mfcc` remains the arm to ship; the CNN is now the honest
measure of how much that decision costs, which is about 4 points of card at 98%
retention.

---

## 9c. The SRAM objection was an artefact of execution order ✅

§9b reported 23.9 KB of working SRAM against a <8 KB target and called it the thing
that stops the CNN shipping. **That was wrong, and wrong in an instructive way.**

`cnn.activation_bytes` costs the network the way a training framework runs it:
materialise conv1's whole output, then pool it, then conv2, and so on. On that
schedule the peak is one layer's input and output held together — 16 KB in conv1
alone, since `(1,32,100)` in and `(8,32,50)` out is 3.2 + 12.8 KB.

**A device has no reason to run it that way.** The log-mel front-end emits one
frame every 10 ms and evaluation windows do not overlap
(`WindowParams.eval_hop_s == win_s`), so each frame can be consumed as it arrives.
Each layer keeps a ring of its last `k` input columns, emits an output column when
its stride phase comes round, and the global average pool accumulates into 32
int32s. No feature map is ever built, and there is no patch buffer to hold.

| | whole-tensor | **streamed** |
|---|---|---|
| log-mel patch buffer | 3,200 B | **0** — frames are consumed on arrival |
| peak feature maps | 16,000 B | **0** — one column per layer |
| per-layer column rings | — | 2,400 B |
| transient window + emitted column | 576 B | 640 B |
| GAP accumulator | — | 128 B |
| FFT working buffer | 4,096 B | 4,096 B |
| **total** | **23.3 KB** | **7.1 KB** ✅ |

**Inside the <8 KB target**, and 4 KB of the remaining 7.1 KB is the FFT buffer,
which is a front-end cost `mfcc` pays too — the CNN's own working set is 3.2 KB
against `carihc`/`bands` at 640 B.

**MACs are unchanged.** Every position is computed exactly once. This is the point
that makes it free: the obvious alternative, *tiling* the patch and recomputing the
overlaps, was measured first and is a much worse deal, because the receptive field
spans 41 of the 100 frames so tiles overlap heavily:

| execution | peak pair | SRAM | MACs | vs whole-tensor |
|---|---|---|---|---|
| whole-tensor | 16,000 B | 23.3 KB | 1.02M | 1.00x |
| tiled, 6 output columns | 19,232 B | 25.9 KB | 1.10M | 1.08x |
| tiled, 1 output column | 6,432 B | 13.4 KB | 1.60M | 1.57x |
| **streamed** | — | **7.1 KB** | **1.02M** | **1.00x** |

### ⚠️ The trap: padding is a per-layer boundary condition

The first tiled implementation was wrong, and wrong *quietly* — every interior
column bit-exact, only the two edge columns off, by 4e-3 in the logit. The cause:
`padding=1` applies at **every layer's** input, not just the network's. A tile
whose span reaches past the patch edge must see zeros at that layer, but a naive
implementation hands it `conv(zeros) + bias`, which is not zero because the conv
has a bias. Each stage therefore owns its own padding (`_Stage`), and
`test_streaming_is_the_same_arithmetic` asserts bit-equality rather than trusting
the reasoning — which is the only reason the error was found at all.

`cnn.stream_forward` is the reference implementation, agreeing with the folded
model to 3e-8 (float32 accumulation order in the average pool). The C version is
G5's job; the geometry it needs is `cnn.layer_spec` / `cnn.geometry`, which
`test_geometry_matches_the_real_model` pins to what `build()` actually returns.

### What this changes

The memory argument is gone from both sides of the ledger. §9b's conclusion that
"the small model is right for the wrong reason" now rests on **one** measured objection
rather than two: the CNN's per-species retention floor, 88.9 ± 3.9 against 95.5
for `mfcc`. That is a training problem, not an architecture problem, and it has
not been attacked yet.

---

---

## 10. Metrics

Judged on what ends up on the card, all computed **after** the state machine, in continuous
time. A model compared on raw AUC can lose to a worse model that interacts better with the
gate.

1. **Birdsong retention** — fraction of bird-seconds (union) inside a recorded region. The
   headline. Target ≥ 98% against the oracle's 100%.
2. **Event recall @ k** — fraction of boxes with ≥ k% of their duration retained,
   k ∈ {50, 90, 100}. **The binding metric** (see §6).
3. **Duty cycle / saving** — never reported without the oracle row at the same pre-roll and
   hangover beside it. "We saved 30%" is meaningless; "30% of the oracle's 35.1%, at 98.5%
   retention" is a result.
4. **Per-species retention floor.** ⚠️ **The main risk of this entire idea.** A gate at 97%
   overall that keeps 99% of a loud abundant species and 45% of a quiet one has injected a
   species bias into every analysis those recordings will ever support — invisibly, because
   the discarded audio is gone. Report the **minimum** over species with ≥20 boxes, not the
   mean. A gate is acceptable only if the floor is high. This matters more in Scotland than
   here: the point of a rewilding project is often the *rare* returning species.
5. **The operating curve** — retention vs duty cycle over θ. This is the deliverable.
6. **Deployment life** — card-days and battery-days from the measured duty cycle. Note the
   cochlea runs every second regardless, so the gate buys **card life and analysis time**,
   and battery only via avoided writes.

---

## 11. Repeatability and traceability

A first-class requirement, because the deliverable is a device left in a forest whose
recordings become primary ecological data. If the gate on a recovered card cannot be
identified, the audio it kept is of unknown provenance — and the audio it discarded cannot
be re-examined.

`provenance.py` writes `runs/<name>/manifest.json` for every run, capturing the **resolved**
config (not CLI args), a stable config hash used as the run's identity, the split seed and
the exact recording list per split, SHA-256 digests of the label files consumed, tracked
package versions, and the metrics achieved. `export.py` stamps the config hash into
`model_params.h`, so a `.bin` recovered from a device resolves back to that manifest.

✅ **Fixed: `git_rev` was recording `None` because it asked the wrong directory.**
`provenance.git_rev` ran `git rev-parse` in `D.REPO` — `BirdSets/`, the shared parent, which
is not a repository — while `birdgate/` itself has been version-controlled all along. Every
manifest written before this said "not reproducible" about a repo that was fine. It now asks
`D.PROJECT` and appends `-dirty` when the tree has uncommitted changes, because a manifest
naming a commit the code did not match is worse than one that admits it does not know. The
app warns on both cases rather than hiding them.

⚠️ **Still true: commit before any deployment build.** A `-dirty` rev pins nothing.

⚠️ **The CNN arm is the one thing here that does not replicate on a GPU.** `CnnParams.device`
defaults to `"cpu"` for exactly this reason — see §9b. On CPU the same seed reproduces to the
fourth decimal across processes; on Apple MPS it does not reproduce at all.

---

## 12. Resource budget

| item | estimate |
|---|---|
| CAR-IHC state | 40 channels × 4 float32 = **640 B** |
| SVM weights + bias | 42 × float32 = **168 B** |
| Ring buffer | 8 × 32 KB — **external** SRAM, unchanged from stock AudioMoth |
| Working SRAM | target **< 8 KB** — what stock firmware leaves free in the 32 KB part, less a stack margin |
| Flash | target **< 100 KB** — the headroom above stock's 126 KB in a 240 KB part |
| Compute | 40 biquads × 32 kS/s ≈ 6-13 MMAC/s on a 48 MHz Cortex-M4F. Tight but feasible; the standard CARFAC decimation trick (later, narrower channels run at a reduced rate) buys 2-4× if needed. |
| CNN arm — flash | 5,865 int8 weights = **5.9 KB**, plus ~4 KB of FFT twiddles and mel table. Comfortably inside the 32 KB target. |
| CNN arm — compute | 1.0 MMAC/s of convolution + 2.2 MMAC/s of log-mel = **3.2 MMAC/s, 6.6% of a 48 MHz M4F**. Cheaper than `carihc` (9.0) and `bands` (7.7); 2x `mfcc` (1.5). |
| CNN arm — **SRAM** | **7.1 KB streamed** — 2.4 KB of per-layer column rings + 0.6 KB transient + 128 B accumulator + 4 KB FFT buffer. Inside the <8 KB target, and 4 KB of it is the FFT buffer `mfcc` pays too. **23.3 KB if executed whole-tensor**, which is what a framework does and what an unwary estimate reports. See `cnn.streaming_bytes` vs `cnn.activation_bytes`, and §9c. |

---

## 13. Stages

### G0 — Frame the problem ✅
Duty cycle, gap structure, high-frequency content, the 32 kHz justification.

### G1 — Binary windows and the oracle gate ✅
`windows.py`, `intervals.py`, `gate.py`, `metrics.py`, `provenance.py`, `app.py`. Results in
§6. **Fixed for the rest of the project: pre-roll 1 s, hangover 1 s, 1 s decision window.**

### G2 — CAR-IHC, from scratch ✅
See §7 for results. `carfac.py`, `frontends.py`, 16 passing validation tests, and a Cochlea
tab in the app.

### G3 — Template-SVM, and the three controls ✅
All four arms trained on identical windows and split; results in §8, §9 and §9b.
**Answers:** the cochlea does *not* earn its keep — `bands`, the plain ERB bandpass control,
beats it on every metric at lower cost, and `carihc` is last on every safety metric in every
table. `mfcc` is the arm to ship: within noise of `bands` on accuracy, 5x cheaper, and the
most predictable under calibration. And the data-scarcity argument for a tiny model does not survive
contact with 33 hours of annotation — a 5.9 KB int8 CNN out-detects every SVM arm by a wide
margin, and is blocked by **activation memory** and its **per-species floor** instead.

### G4 — Calibrate the knob ✅
`scripts/sweep_threshold.py` sweeps θ on **val** and names the operating points, carrying the
two metrics `pick_theta` omits: the per-species floor and the worst recording. Run
`g4-calibrate-mfcc` (`mfcc`, 60 thetas):

| point | θ | retention | saving | worst rec | species floor | set by |
|---|---|---|---|---|---|---|
| `paranoid` | 0.165 | 99.55% | 0.76% | 98.29% | 98.23% | American Robin |
| `balanced` | 0.319 | 98.19% | 8.62% | 95.47% | 95.25% | American Robin |
| `frugal` | 0.423 | 95.86% | 20.88% | 89.48% | 91.38% | Red-breasted Nuthatch |

**The finding that made the table usable:** `species_floor` was being set by whichever species
the split under-sampled. On val that was Brown-headed Cowbird — 21 boxes, **41 seconds** of
song, 61% retention, 95% CI **32–95**. `min_boxes=20` did not catch it because it guards on
box *count* while precision depends on total *duration*; 21 clears 20. Guarded on bootstrap CI
width instead (`metrics.admissible`, `max_ci_width=0.10`), `frugal` went from **0.76% saving to
20.88%** — the same model, the same gate, a floor that means something.

No default point is chosen: PLAN §14's "what is actually scarce" is still open, and it selects
a row rather than changing the table.

### G4b — Closed: the CNN's floor was an artefact ✅
- ✅ **SRAM: solved.** 23.9 KB was whole-tensor execution. Streamed, 7.1 KB, bit-identical.
- ✅ **Species floor: resolved.** The blocking `88.9 ± 3.9` was produced by the count-only
  guard above. Recomputed with the CI guard, `cnn-int8` scores **95.54%** — second only to
  `mfcc`, and with the **best worst-recording figure of any arm at 97.08%**.

| arm | retention | worst rec | saving | species floor | set by |
|---|---|---|---|---|---|
| carihc | 97.28 | 82.07 | 7.75 | 92.72 | Golden-crowned Kinglet |
| bands | 97.65 | 89.57 | 12.13 | 95.23 | Mountain Chickadee |
| mfcc | 97.95 | 94.11 | 8.75 | 96.22 | Mountain Chickadee |
| cnn | 97.65 | 94.33 | 14.09 | 90.43 | Red-breasted Nuthatch |
| **cnn-int8** | **98.31** | **97.08** | **12.19** | **95.54** | Spotted Towhee |

**So the decision is now a genuine trade, not a disqualification.** `cnn-int8` buys ~3.4 more
points of card at higher retention and a better floor; `mfcc` is ~2× cheaper in compute, far
simpler in C, and has no 5.9 KB of weights. `mfcc` is being ported **first** on those grounds —
cost and portability — and explicitly *not* because the CNN was ruled out. Revisit once
something is running on hardware.

⚠️ `n_admissible` is 17–20 of 20 species and differs between arms, so the floors are not
strictly like-for-like. The spread is small in the shipping region; it would matter more if
these were closer.

### G5 — Export ✅ (float), quantisation deferred
`birdgate/export.py` generates `firmware/arms/<arm>/model_params.h` from the trained model,
with a provenance block so a recovered card resolves to the run that produced it. Done in
**float**, not Q15/Q31, and that reordering is deliberate: quantising while integrating
confounds a port bug with fixed-point drift. Integrate, prove it works on hardware, *then*
quantise as an isolated change with a measured before/after.

⚠️ When quantisation does happen — the CAR cascade is a 40-stage IIR chain where
**fixed-point error compounds down it**, and this is where a faithful-looking port silently
degrades. Measure per-channel drift on *real audio*, not white noise, which does not excite
marginally-stable poles. Budget: ≤ 0.2% absolute retention lost. (`mfcc` has no such chain,
which is one more reason it is the first arm ported.)

### G6 — Hardware ◐ in progress
Build chain **proven**: `firmware/build.sh stock` reproduces the official 1.12.1 release
byte-for-byte, and a modified build was flashed to a dev AudioMoth and confirmed three ways.

Whole chain **ported and verified** (2026-09-22): front-end, scorer, adaptive threshold,
state machine and the device's ring framer produce *identical gate output* to Python on
synthetic audio and on 15 minutes of the busiest recording — 659/900 windows recorded,
3 openings, 0 differing, retaining 100% of the 248 annotated bird-seconds in that stretch.
Features agree to ~2e-6 relative; the closest window sits **79×** the numerical error from
theta. `tests/test_c_parity.py` builds the harness from the device sources and pins all of it.

⚠️ **Parity must be measured on audio where the gate actually fires, at the deployed
threshold.** Two earlier runs flattered themselves. The first recorded zero windows on both
sides and so agreed only about silence. The second — the source of the "893/900 windows,
628× headroom" figure this section used to report — was taken near θ = 0, where the gate is
open 99.9% of the time and agreement again mostly measures a constant. At the deployed θ the
honest headroom is 79×: still comfortable, an order of magnitude below what was claimed. The
test now asserts both conditions rather than trusting the operator to choose good audio.

Remaining: `app/main.c` is written and builds (green LED follows the gate, SD calls stubbed);
it has **not been flashed**. See `ROADMAP.md` §1. The next milestone is
deliberately **not** recording: drive the green LED from gate state, so the whole detection
chain can be confirmed live without involving the SD path at all. Then the write path with
pre-roll rewind, and `simulate_device.py` — a real hour in, the byte stream a card would hold
out, plus a loss report listing every dropped box by species.
**Deliverable:** a listenable before/after — the only demo that convinces anyone.

**Note on `adapt.c`:** the deployed config is `mode="quantile"`, 0.2 quantile over a trailing
300 s. At one window per second that is 300 floats and one sort per second — affordable, so
implement the **exact** quantile. Switching to `ema` to save memory would change the
statistic theta was calibrated against and force every G4 number to be re-earned.


### G6b — The device front-end, and what is deliberately deferred

Porting mfcc exposed that the Python arm computes two things a device cannot, and one
thing nobody chose. `mfcc-dev` fixes only the first category; the rest is listed here so
it is deferred on purpose rather than forgotten.

**Fixed now (causality — `arms = "mfcc-dev"`):**
- `center=False`. librosa pads by `n_fft//2` and centres frame *i* on sample *i·hop*; a
  device has no future samples to centre on. Also drops frames per 1 s window from 100 to
  97, since the last frame must *end* inside the window — using 100 silently pulled three
  frames of the next window's audio into this one.
- `top_db=None`. `power_to_db` clamps 80 dB below the maximum of the **whole array**, so
  later audio rewrites features already computed. Measured: fires on 1 recording in 8, and
  0.42% of that recording's frames. Nearly inert here — but it is the non-causality that
  would make C/Python parity fail *intermittently*, on exactly the loud-transient
  recordings a field deployment produces most.

**Deferred, both worth doing, both to be measured before adoption:**

- **32 mel bands instead of 128.** The mfcc arm passes no `n_mels` to librosa, so it
  inherited the default of **128** — while `logmel` explicitly chose 32 with the comment
  "this has to fit a microcontroller, and the mel filterbank table itself is flash." Nobody
  decided 128; it arrived. Measured cost of the difference is **8.6 KB flash** (20.6 → 12.2
  KB), and it is *not* mostly the mel table — that is dominated by FFT bins, not bands (873
  non-zeros at 128 vs 837 at 32). It is the **DCT matrix**: 20 × 128 f32 = 10.2 KB, half the
  entire front-end budget, and it scales linearly with bands. ⚠️ This is a real feature
  change — 20 DCT coefficients over 32 bands is a different feature space — so mfcc's G3
  numbers would have to be re-earned, not assumed.
- **Welford running mean/variance instead of the frame ring.** The window features are the
  mean and std of each coefficient over the window's frames, and nothing else is ever read
  from the ring — so the 100 × 20 f32 buffer (**8 KB**, half the SRAM) can be replaced by
  streaming accumulators. Numerically exact up to float ordering, unlike the 32-band change.
  Do it *separately* from the band change: bundling a lossy change with an exact one makes
  any retention drop unattributable.

**Measured budget as it stands** (correcting `firmware/README.md`, which listed ~0.2 KB
flash and ~4.7 KB SRAM — the former counted only the weights, the latter missed the ring):

| | flash | SRAM |
|---|---|---|
| `mfcc-dev`, 128 bands, frame ring | 20.6 KB | 16.3 KB |
| + 32 bands | 12.2 KB | 16.3 KB |
| + Welford | 12.2 KB | ~8.3 KB |
| target | < 100 KB | **< 8 KB** |

Flash is never the binding constraint. SRAM is, and only both changes together reach the
stated target — which is worth noting against the CNN's 7.1 KB streamed figure, because it
means the memory argument between the arms is much closer than the README implies.


### G7 — Retrain on the Highland site
The actual goal. What changes, and what deliberately does not:

- **Changes:** the training set, the species vocabulary, the negatives (Atlantic weather —
  far more wind and rain than Sierra Nevada), the calibrated threshold, the deployment prior.
- **Does not change:** the front-end, the state machine, the metrics, the quantisation, the
  export path, the manifest format. That invariance is the point of building it this way.
- **What the Highland data will need to be useful:** strong-ish annotations on at least a few
  hours, spanning **night, midday and dawn** — not just the chorus. Occupancy there will be a
  few percent rather than 49%, which is a much better setting for a gate and a much worse one
  for training (positives become scarce). Expect to need targeted labelling of *positives*
  and to take negatives for free.
- **Cross-site check worth doing:** train on SNE, evaluate on Highland audio before
  retraining. That number is the honest estimate of how much a gate transfers between
  forests, and nobody has it. Whatever it is, it is worth knowing.


### G7b — Enrich the training data until every species can set the floor

G4 exposed the limit of the current data rather than of the model. `species_floor`
is a minimum over species, so it is only as trustworthy as the *thinnest* species in
the split — and on val, three of 22 species cannot be measured to within ten points
at all. Brown-headed Cowbird has 21 boxes and **41 seconds** of song in the entire
split; its retention is 61% with a 95% CI of 32-95. `metrics.admissible` now excludes
species like that and says how many it dropped, which makes the shipped number honest,
but honest is not the same as sufficient: excluding a species means this dataset can
make **no claim** about it, and a gate deployed at the Highland site will still be
deciding that species' fate every second.

The fix is data, and it is the natural point to merge with the sibling project:

- **Target coverage per species, not in aggregate.** The acceptance criterion becomes
  "every species likely at the site has enough song to pin its retention to within X
  points", which is a *collection* target derived from `per_species_retention`'s CI,
  not a wish. Run the CI backwards to get the required seconds per species and collect
  against that list.
- **Use the sibling SED model as a teacher** (§14) to find and label far more song per
  species than hand-annotation can reach, especially for the quiet and infrequent ones
  that are exactly the species the floor is about.
- **Augment rather than only collect.** Add the negatives the Atlantic site will
  actually produce — wind, rain, sea noise — mixed against known-clean calls at
  controlled SNR, plus semi-synthetic mixtures that place scarce species into realistic
  backgrounds. The point is not more data in general; it is more data *at the operating
  conditions where the floor is set*, which is quiet distant calls against loud weather.
- **Re-test the guard afterwards.** If enrichment works, `max_ci_width` stops excluding
  anyone and the floor becomes a real constraint over the full species list rather than
  over the well-sampled subset. That, not a better headline retention, is the measure of
  whether the enrichment succeeded.

⚠️ Until that lands, one consequence stands: the `species_floor` column in the G3 arm
comparison (`runs/g3-compare/manifest.json`) was computed with the count-only guard and
is therefore hostage to whichever species each split happens to under-sample. It did not
bite on test only by luck — test's worst species, Pacific-slope Flycatcher, is also thin
(37 boxes, 49 s) and merely happened to score well. Those numbers should be recomputed
with `max_ci_width` before the arm comparison is cited as a safety result.

---

## 14. Open questions

- **Which prior do we train against?** SNE negatives are ~45-48% of windows; a Highland
  24-hour deployment will be 90-98%. G1 found SNE *does* contain low-occupancy recordings
  (1.4%, 10.9%, 11.0% — the 04:00 and 17:00 hours), so **post-hoc threshold calibration on
  those recordings is available and is probably the right answer**, being cheaper and more
  transferable than reweighting the loss.
- **Does the gate need an explicit "loud non-bird" class?** Rain and wind are what will fill
  a card in Scotland. SNE has no rain labels; we may need to hand-mark a few minutes, and
  should plan to mark them properly in the Highland data from the start.
- **Normaliser** (§5) — settle empirically; one-line sweep.
- **Is 1 s the right decision unit?** Free from the ring buffer. 0.5 s would reach more of
  the many ~1.9 s gaps, at 2× inference rate. Worth one measurement, but note the answer may
  differ in Scotland where gaps are long — in which case 1 s is fine and this is moot.
- **What is actually scarce at the Highland site?** A 128 GB card holds ~23 days of ungated
  32 kHz mono; 3×AA gives ~48 days. If the card outlasts the deployment, the gate buys
  *analysis time*, not storage, and θ should sit far more generously than a storage-driven
  sweep would put it. **Decide this before tuning θ** — it is a site question, not a
  modelling one.
- **Should the gate be distilled from the sibling project's SED model?** That model already
  reaches event AP 0.894 vs 0.558 prevalence. Using it as a *teacher* to label far more audio
  than 33 h — including unannotated Highland audio — is the cheapest route to training data
  and uses only our own models. Strong candidate if G3 disappoints, and a natural fit for
  bootstrapping the Highland site before hand-labelling exists.
