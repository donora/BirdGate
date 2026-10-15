# BirdGate — roadmap

What is next, in order, on each of three fronts: the firmware port, the hardware
work around it, and the cleanup needed before this goes online. `docs/lab-notebook.md` is the
record of what has been done and why; this file is only what has not.

Last revised 2026-10-15.

---

## 0. Where things stand

**Python side — complete.** Four front-end arms compared under controls, the gate state
machine, the adaptive threshold, threshold calibration, the energy model, and a float
export path. `mfcc-dev` is the shipping arm: MFCC means and stds over a 1 s window, a
40-weight linear model, CFAR background subtraction, and the gate. Every number in
`SUMMARY.md` re-derives from the caches.

**Firmware — the whole chain is written, host-verified, and runs live on an AudioMoth as
a real-time detector** for both `mfcc-dev` and `cnn` (2026-09-22): the green LED follows
each one-second decision. The build is `BG_LED_ONLY`; nothing is written to the card yet
(F8). The table below is the state at 2026-09-16 and §4 has the CNN arm since.

| piece | state |
|---|---|
| build chain (`firmware/build.sh`) | proven: `stock` reproduces the official 1.12.1 release byte-for-byte; a version bump was flashed and confirmed on hardware |
| `arms/mfcc-dev/mfcc.c` | streaming front-end: real-input FFT, Welford window statistics, no frame ring |
| `common/scorer.c`, `adapt.c`, `gate.c` | written; mirror `svm.py`, `adapt.py`, `gate.py` |
| `common/birdgate.c`, `bg_stream.c` | the chain behind one call, and the int16 ring framer the device uses |
| `app/main.c` | stock `main.c` with the chain hooked into the recording loop; `BG_LED_ONLY=1` stubs every SD call so it runs with no card; `BG_LED_RAW=1` points the green LED at the per-window detection rather than the gate |
| parity | `tests/test_c_parity.py` builds the harness from the device sources and compares the whole chain against Python, including the per-window detection the LED shows. **Masks identical on synthetic audio and on 15 min of the busiest real recording** (659/900 windows, 3 openings, 100% of the 248 annotated bird-seconds retained); features within 4.3e-4, z within 2.8e-6, closest window 79x the numerical error from theta. Verified 2026-09-22 |
| device build | links the chain (`arm-none-eabi-size` differs from stock); SRAM and flash below |

**Measured footprint** (from the linker map, our objects only):

| | LED-only build | full-recording build |
|---|---|---|
| our SRAM (`.bss`) | 8,768 B | 8,768 B |
| our flash (tables + code) | 22.8 KB | 22.8 KB |
| total `.bss` + `.data` + heap | 24.5 KB | 26.9 KB |
| left for stack in the 32 KB part | ~8.3 KB | ~5.9 KB |
| `.bin` | 126.8 KB | 151.0 KB (of 240 KB) |

Our SRAM: 4 KB frame buffer, 2 KB power spectrum, 2.4 KB adaptive-threshold ring and
sort scratch, ~0.3 KB state. The LED-only build is smaller than stock because the
SD-card code is garbage-collected.

---

## 1. Firmware plan

### Done on the host (F1–F7, 2026-09-16)

- **F1 `scorer.c`** — dot product and bias.
- **F2 export** — `BG_WIN_S`, `BG_ADAPT_N`, `BG_MIN_SCALE`, `BG_AGG_N`, `BG_RECHECK_S`
  added; the exporter refuses any adaptive-threshold configuration other than
  quantile / global scale / max-of-N, which is all the C implements. Regenerated:
  theta and scale unchanged.
- **F3 `adapt.c`** — 300-score ring of strictly-past scores, pre-filled with the first
  score; `np.quantile`'s linear interpolation reproduced including its two-sided lerp;
  one global scale with the `BG_MIN_SCALE` floor; max over the last three.
- **F4 `gate.c`** — `gate.run` step for step. Reports the live state and, on opening,
  the pre-roll backfill count; the recorded mask is the union, as in Python.
- **F5 SRAM** — Welford accumulators replaced the 7.6 KB frame ring; a HALF-point
  complex FFT with the standard real-input post-processing halved the FFT buffers.
  Twiddles are read through `fft.c` so the header's `static` tables link once.
  Parity moved from 1.8e-4 to 1.2e-3 on features (Welford's summation order) and the
  gate mask is unchanged.
- **F6 parity test** — `tests/test_c_parity.py`. The int16 ring path (`--i16`) is
  asserted bit-identical to the float path on the same samples.
- **F7 `app/main.c`** — written and building. Hook points: `bg_stream_init` after DMA
  init; after each completed slot in the write loop, `bg_stream_samples_arrived` then
  `bg_stream_advance` in a loop driving `AudioMoth_setGreenLED`. The gate runs only
  when the configured effective rate is 32 kHz; otherwise the green LED blinked once
  per slot (since changed to a fast red blink with green dark, see C8b).

### To reach the green LED (the hardware milestone)

1. ~~**Run the real-audio parity case.**~~ ✅ 2026-09-22 — masks identical over
   15 minutes of SNE_001 at the deployed threshold, 79x headroom over numerical
   error. Re-run it after any front-end change: `pytest tests/test_c_parity.py -s`.
2. **Configure the device** with the Configuration App: sample rate 32 kHz
   (`sampleRate 384000`, `sampleRateDivider 12`), one recording period covering the
   whole day, the longest record duration and shortest sleep the app allows, LED
   enabled, amplitude threshold and frequency trigger off, filter off. Save.
3. **Flash `firmware/out/birdgate-mfcc-dev.bin`** with the Flash App (device in USB/OFF).
   Rollback is the stock 1.12.1 binary in `AM_Software/`.
4. **Observe.** Switch to CUSTOM. The red LED behaves as stock (it flashes on the
   simulated SD writes, which are no-ops). The green LED is the **detector**
   (`BG_LED_RAW 1`): it lights for each one-second window scored above theta and goes
   out on the first window that is not, so a sound can be tried over and over without
   waiting out a recording. Decisions land once a second, so the LED follows about a
   second behind. If the red LED blinks fast with green dark instead, the sample rate
   is not 32 kHz and the gate is not running.

   Set `BG_LED_RAW 0` and rebuild to watch the **gate** instead — hysteresis,
   hangover and the 20 s minimum-on, i.e. what would actually reach the card. Use
   that when testing the recording policy; the detector when testing the classifier.

   Two things to expect, both the adaptive threshold working as designed. Start the
   device in relative quiet: the background is seeded from the first window, so a
   noisy start desensitises it until the 300 s ring turns over. And leave gaps —
   the background is the 0.2 quantile of the last 300 s, so continuous loud input
   eventually becomes the background. Measured, it still detects with 80% of the last
   five minutes loud and fails at 95%; intermittent testing is well clear of that.
5. **Film it against the simulator.** Play a clip with a known Python gate mask at the
   device and compare LED timing to the mask. Sub-second agreement is enough.

Known differences from the simulator to keep in mind on the device: stock applies its
48 Hz DC-blocking filter before samples reach the ring (below the 150 Hz mel floor, so
inert), and the microphone gain is not the training data's gain — CFAR removes the
offset, which is the point of it, but `H4` is where that gets checked properly.

### F8. The SD write path with pre-roll
Set `BG_LED_ONLY 0`. On an open transition `bgResult.backfill` says how many 1 s windows
to recover; mark those slots and the current one via stock's `writeIndicator`, and let
stock's per-slot write decision do the rest. Then `scripts/simulate_device.py` (a real
hour in, the bytes a card would hold out, and a per-species loss report) and the
listenable before/after pair. Full-recording build leaves ~5.9 KB for stack; check the
map after any change.

### F9. Firmware identity
Set `firmwareDescription[]` to `"BirdGate"` and our own version triple. Both go into
`CONFIG.TXT` and every WAV's GUANO block, which is the provenance hook. Untested
whether the Configuration App still talks to a device with a non-stock description;
settle that on hardware (H3) before relying on it.

### F10. Quantisation
Only after the float chain runs on the device, as an isolated change with a measured
before/after. Budget ≤ 0.2 % retention. Probably never necessary for `mfcc-dev`: the
M4F has an FPU and this arm uses a few percent of it.

### F11. Measure the stack, which nothing ever has
The app's silicon tab reads the linker map symbol by symbol and can say exactly what
the statics cost — 8,776 B of ours on `mfcc-dev`, leaving 8,284 B of the 32 KB, and
3,840 B on `cnn`. What it cannot say is what the **stack** actually needs, because the linker does
not know: `.stack_dummy` is a 1 KB placeholder, not a measurement, and no
`-fstack-usage` pass or fill-and-watch probe has ever been run on this firmware.

So every "stack left" figure in this file and in the app is a ceiling on what is
available, not a margin that has been verified. On `mfcc-dev` that is very likely fine.
On `cnn` at 3,840 B it is the tightest number the project has and it is unmeasured,
which is the wrong combination — the failure mode is a stack that grows into `.bss`
and corrupts the adaptive threshold's ring rather than faulting, so it would present
as a detector that gradually stops working. Cheapest version: `-fstack-usage` over our
objects for the static part, then fill the region with a pattern at boot and read the
high-water mark after an hour of recording. Do it before `BG_LED_ONLY 0` goes to a
card, and before the CNN arm is flashed at all.

---

## 2. Hardware plan

### H1. Green-LED milestone on the dev AudioMoth ✅
Reached 2026-09-22: the whole chain runs on the device and the green LED follows the
detector. The steps are under §1 "To reach the green LED"; bench testing by speaker
playback is in `firmware/README.md`.

### H1b. Settle the adaptive threshold's warm-up
Found while bench testing (2026-09-22): the background ring is pre-filled with 300 copies
of the first window's score, so the anchor cannot move for 60 s and is not clear of the
pre-fill for 300 s. On the bench this made playback-with-gaps fire on background;
`firmware/README.md`, "Bench testing by playing recordings", has the measurements.

The field question it raises is unmeasured and worth one experiment: a device waking into
night silence anchors low, so the first minutes of the dawn chorus are over-recorded. That
errs toward recording and is therefore safe, but it is unquantified, and a short
`recordDuration` makes it permanent because `bg_stream_init` re-seeds every recording
period — with the stock 55 s default the anchor never escapes the pre-fill at all.

Run the simulator over a full night-into-dawn recording, reset per recording period as the
device does, and compare duty and retention against the continuous case. If the cost is
real, the fix is either to seed the ring from the first N windows instead of copies of the
first one, or to persist the background across periods. Both change what theta was
calibrated against, so measure before changing either.

### H2. The energy bench
Three measurements turn `energy.py`'s estimates into deployment-days, and they decide
whether the arm comparison means anything energetically. `scripts/energy_report.py`
prints the protocol.

| | measures | needs |
|---|---|---|
| M1 | SD energy per file open/close vs per sequential byte | current probe, a card, a scripted write pattern |
| M2 | EM0 and EM1 currents, and what the peripheral clock setting costs | current probe, a way to pin the core in each mode |
| M3 | the analog floor: mic + preamp + ADC running with the core asleep | the same |

A Nordic PPK2 or a µCurrent into a scope is enough; supply at 4.5 V through the
battery contacts. Record every number into `energy.py`'s parameter table with its
date and setup.

### H3. Configuration App compatibility with a custom firmware description
Ten minutes on hardware. Decides F9.

### H4. Side-by-side field trial
Two devices for 24 h in the same spot, one stock and one gated. Compare what the cards
hold: the gated card's content should be a subset of the stock card's at the predicted
duty cycle. Also the first check of whether the device's microphone gain sits where the
training data did.

### H5. Highland data
Unchanged from `docs/lab-notebook.md` G7: collect, re-fit training set and threshold only.

---

## 3. Cleanup before this goes online

### Done (2026-09-16)
- Reframed the project on its own terms in `README.md`, `SUMMARY.md`, `docs/lab-notebook.md`,
  `firmware/README.md`, `app.py` and module docstrings. The Sabbella et al. paper is
  a one-line acknowledgement as the inspiration; nothing measures itself against
  its numbers or its method. Removed the "resume here" status blocks.
- **Provenance.** `GateConfig` now carries `adapt`, and every script writes the
  config it actually ran with (arm, gate parameters, theta, adaptive-threshold
  parameters). Config hashes change as a result, so the next `export_model` run
  stamps a new hash into `model_params.h`; the old manifests under `runs/` keep
  their old hashes and are not re-derived.
- **Licence.** `LICENSE` (MIT) for this code; the Open Acoustic Devices MIT notice
  is reproduced at `firmware/LICENSE-AudioMoth` and referenced from `LICENSE`.
- **Dataset attribution** in `README.md` "Data".
- **Decoupled from the sibling project.** `birdgate/data.py`, `birdgate/splits.py`
  and `birdgate/audio.py` replace the `sys.path` import of `birdsong`. The split is
  the same algorithm and seed and is pinned by `tests/test_split.py` to the six val
  and six test recordings every manifest records. `data/recordings.csv` carries
  sample rate and duration so nothing but feature extraction needs the audio.
  `pyproject.toml` declares the dependencies; `cnn`, `app` and `dev` are extras.
- **Dead scaffolding removed:** `firmware/arms/carihc/`, `firmware/arms/mfcc/`,
  `birdgate/quantize.py`, `tests/test_gate_parity.py`, `configs/`. Nothing loaded
  the YAML; the config lives in `config.py` defaults and `scripts/train.py`.
- **Stage tags** stripped from module docstrings, printed strings and app tabs; they
  remain in `docs/lab-notebook.md` (the log) and in `runs/` directory names, which the app reads.
- `SUMMARY.md` is no longer a draft: the banner and `[fill]` markers are replaced by
  pointers to the section of `docs/lab-notebook.md` holding the detail.
- `README.md` gained hardware, flashing, regeneration and setup sections.

### Still open
- [x] **Dataset licence.** CC BY 4.0 (checked 2026-10-15). Labels, derived scores and
      eight 30 s clips ship in `data/` and `demo/`; `data/ATTRIBUTION.md` has the
      citation and the changes.
- [x] **Runs from a fresh clone.** `demo/` (12 MB) stands in for the audio and the
      feature cache; `scripts/build_demo.py` rebuilds it.
- [ ] **Public v1.0.0.** A fresh public repository without the working history, a
      tour (`TOUR.md`), and the detector `.bin` as a release asset, labelled as not
      recording to the card.

---

## 4. The CNN arm on hardware

The goal is the same green-LED detector, driven by the 5.9 KB int8 CNN instead of the
40-weight linear model, so the two can be compared on the device rather than only in the
simulator. **This is an experiment, not a change of shipping decision.** `mfcc-dev` still
ships: the CNN out-detects it (12.2% saving against 8.8% at matched retention, a better
worst-recording floor) but loses on the per-species floor, 91.7 ± 2.0 against 95.5, and
that metric is the one PLAN §10 calls the main risk of the whole idea. What a device
comparison buys is whether the CNN's advantage survives a real acquisition chain.

Everything downstream of the score is shared and already ported: the adaptive threshold,
the gate, the ring framer, the LED modes, the bench knobs. Only the front-end and the
scorer change.

### Two things that make this more than a C port

**The CNN's front-end is not causal, so the model has to be retrained first.**
`frontends.logmel` calls `power_to_db` without `top_db`, which defaults to clamping 80 dB
below the maximum of *the whole array*, and without `center=False`, so every frame is
centred on samples that have not arrived. This is exactly what forced `mfcc-dev`, and it
needs the same treatment. The architecture survives it — global average pooling means the
patch can be 97 frames instead of 100 with no shape change, verified — but the weights
were fit on a different feature distribution, so they have to be refit and recalibrated.

**⚠️ Corrected 2026-09-22: a float port does not fit, so the int8 grid is not optional.**
This section first said "port it in float first, as `mfcc-dev` was". That was wrong, and
wrong because of a figure in this repo: PLAN §12's **7.1 KB streamed** is an *int8*
number. The per-stage rings it counts as 2,400 B are 2,400 *activations*; held as floats
they are 9,600 B, and the arm then needs 19,324 B against the 17,056 B that stock
firmware leaves for us and the stack combined. It does not fit, with or without the
fallbacks listed below.

**What fits is int8 storage with float arithmetic**, and that turns out to be a better
fit for this project than either alternative. A stored code is read back as
`(code - offset) * scale`, which is exactly the value `cnn.ActQuant` produces, so the C
runs the int8 *grid* without CMSIS-NN's integer requantisation — no per-channel
multipliers, no int32 accumulators, no shift arithmetic. The FPU does the multiplying.
Two consequences worth having:

- `cnn.quantise` documents itself as "simulates int8; not a bit-exact CMSIS-NN run",
  which reads as a limitation. For *this* design it is the specification. The existing
  `cnn-int8` numbers therefore describe the implementation rather than approximating it.
- `cnn-int8` is the better arm anyway — 98.3% retention against the float model's 97.7%,
  and a better worst-recording floor — so the arm being ported is the one worth porting.

The cost is one Python reference that does not exist yet: `stream_forward` has the
streaming schedule and `QuantGate.forward_np` has the quantiser placement, and nothing
combines them. C6 needs `cnn.stream_forward_quant` first, asserted against
`QuantGate.forward_np` on a batch, before any C can be checked against anything.

### Measured budget, before writing any C

Stock firmware leaves **17,056 B** of the 32 KB for our statics *and* the stack.

| | `mfcc-dev` (shipping) | `cnn`, float rings | `cnn`, int8 rings |
|---|---|---|---|
| front-end (FFT buffer + spectrum) | 6,148 B | 6,148 B | 6,148 B |
| stage rings | — | 9,600 B | **2,400 B** |
| widest transient column, accumulator | — | 1,152 B | 1,152 B |
| adaptive threshold ring | 2,424 B | 2,424 B | 2,424 B |
| **our SRAM** | **8,772 B** | **19,324 B** | **12,124 B** |
| **stack left** | 8,284 B | **−2,268 B** ❌ | **4,932 B** |
| our flash | 20.8 KB | ~15 KB tables + 23.5 KB weights | same |
| compute | ~3–6 % of the M4F | 6.6 % — two thirds of it the log-mel, not the network | same |

The CNN is *cheaper in tables* than the linear arm: the 10.2 KB DCT matrix disappears
entirely, which pays for the float weights on its own. Flash is not the constraint here,
113 KB is free. **4,932 B of stack is the number to watch**, and it is the tightest this
project has run. Confirm it on the linker map at C8, and if it bites, the spectrum can be
overlaid on the FFT buffer (−2,052 B) and the adaptive ring's sort scratch replaced by an
order-maintained insert (−1,200 B).

### Progress (2026-09-22)

C1 through C8 are done: **the CNN arm builds, fits, and matches its reference.** What
remains is C9, the device comparison, which needs hardware.

| | `mfcc-dev` | `cnn` |
|---|---|---|
| flash | 125,700 B | 140,892 B of 245,760 |
| RAM used | 24,484 B | 27,392 B |
| **stack left** | 8,284 B | **5,376 B** |
| gate masks vs Python | identical | identical |
| per-window detections | identical | 1 flip in 900, on a window 2.9e-3 from theta |

Flash `out/birdgate-cnn.bin` exactly as for the other arm; `app/main.c` needed no
change, because `common/arm.h` means nothing downstream knows which arm it is running.

**C2 — the causal front-end did not cost the CNN anything.** Three seeds on
`logmel-dev`, test split, against the published centred-`logmel` figures:

| | retention | saving | worst rec | species floor |
|---|---|---|---|---|
| `cnn-int8` on `logmel-dev`, as ported | 97.82 ± 0.64 | 13.28 ± 1.34 | 94.95 ± 1.90 | **94.37 ± 2.59** |
| `cnn-int8` as published | 97.88 ± 0.45 | 12.91 ± 0.62 | — | 91.73 ± 1.96 |
| `mfcc` (ships) | 97.95 | 8.75 | 94.11 | 96.22 |

The floor is the column that decides this arm, and causality moved it the *right* way,
by about 2.3 points. It is still the unstable one — seed 2 lands at 91.22 — and still
below `mfcc`. On **val** at the shipping operating point the exported arm reads 98.47%
retention, 13.84% saving and a 95.79% floor, which beats `mfcc-dev`'s 98.25 / 8.44 /
95.25 on all three; val and test disagree about the floor ordering, which is itself a
statement about how thin that measurement is.

**What the port cost, and what it did not.** The CNN is *cheaper in flash tables* than
the linear arm — the 10.2 KB DCT matrix does not exist here — and dearer in working
memory, landing at 5,376 B of stack against the linear arm's 8,284 B. Compute is 6.6% of
the M4F by MAC count, two thirds of it the log-mel rather than the network, and
unmeasured on silicon.

**A finding about int8 and parity, and a decision taken.** The linear arm agrees with
Python to 4e-4 and the gate masks are identical. The CNN cannot work that way: any
float difference, however small, occasionally flips an activation code, and a flipped
code moves the output by a whole step. Measured with the arm as first defined: the
score differed on ~3% of windows, never by more than one step, but the gate's 20 s
minimum-on turns a single flipped opening into 20 recorded windows, and masks came out
619 against 598 over 15 minutes.

Two of the thirteen quantisers were artefacts of applying one after *every* module
rather than anything a device does — after `Flatten`, which changes no values, and
after the final `Linear`, whose single logit goes straight to a comparison and which a
device reads out of the int32 accumulator. The second spread 8 bits across the whole
logit range: a step of ~0.07 against a threshold of 0.54, and the dominant cause of the
disagreement. Both are now pass-through in `cnn.quantise` and skipped in the firmware,
which changes `cnn-int8`'s definition and therefore its numbers; they are being
re-earned rather than carried over.

⚠️ **That change is a parity fix, not an accuracy win, and the first reading said
otherwise.** On seed 0 it looked like a real gain — the species floor went 94.36 to
95.29 and the worst recording 96.51 to 97.07. Across three seeds the floor moved 94.06
to 94.37 and the worst recording 94.78 to 94.95, both far inside a ±2.6 spread, with
seeds 1 and 2 barely moving at all. Dropping the two quantisers removed the dominant
source of device/simulator disagreement and cost nothing; the seed-0 gain does not
generalise.

What is left is irreducible, and worth stating as a general lesson about porting
quantised networks. The C and PyTorch sum a convolution in different orders and differ
in the seventh digit; float would absorb that, but a rounding boundary turns it into a
whole-step jump. The linear arm agrees to 3e-6 with **79x** headroom over its closest
window to theta. The CNN's closest window sits *inside* its numerical error. Measured
over 15 minutes of real audio: gate masks identical, and exactly **one** window in 900
flips its detection — a window 2.9e-3 from theta against an error of 1.8e-2, so it was
on the threshold either way. `tests/test_c_parity.py` asserts precisely that: masks
identical, and every disagreement explained by threshold proximity rather than by the
implementation.

### Stages

**C1. `logmel-dev`, the device-faithful front-end.** Add it to `frontends.py` beside
`mfcc-dev`: `center=False`, `top_db=None`, 97 frames per window. Extend
`scripts/build_cache.py` and rebuild that arm's cache. Needs the audio mounted.

**C2. Retrain, recalibrate, re-compare.** `train_cnn` on `logmel-dev`, three seeds
(the CNN's seed variance is real, unlike the convex arms), then `sweep_threshold` for
theta. Re-run `compare_arms` so the published table describes the arm actually being
ported. ~20 min per seed on CPU; do not use MPS, it neither replicates nor converges as
well. **Expect the numbers to move** — the species floor is the unstable column and the
one that matters, so re-read it before spending effort on the port.

**C3. `export_cnn`.** A second exporter emitting a 32-band mel bank, Hann and twiddles,
the folded per-layer conv weights and biases, the final linear layer, `InputNorm`'s mean
and std, and the same adaptive-threshold and gate constants and provenance block the
linear exporter writes. Reuse `export_header`'s table emitters rather than forking them.

**C4. An arm interface in the firmware.** `birdgate.c` and `bg_stream.c` include
`mfcc.h` and `scorer.h` directly today. Replace with four calls each arm provides —
begin, get the frame buffer, push, finish-and-score. `bg_result_t.feat` survives
unchanged if the CNN reports its 32 globally-average-pooled channels there, which is
genuinely the analogous quantity: it is the vector the final linear layer consumes. The
harness, the LED modes and `main.c` then need no arm-specific code at all.

**C5. `arms/cnn/logmel.c`.** The existing `fft.c` and the real-input packing carry over;
drop the DCT, use 32 bands, keep `power_to_db`'s clamp-before-log.

**C6. `arms/cnn/cnn.c`.** The streaming convolution. It has to mirror two references at
once: `cnn.stream_forward` for the **schedule** (each stage owns its own time padding)
and `QuantGate.forward_np` for the **arithmetic** (a quantiser after every module, so the
order is conv, quantise, ReLU, quantise, pool, quantise). Write
`cnn.stream_forward_quant` first, assert it against `QuantGate.forward_np`, and only then
write the C against it. `cnn.layer_spec` / `cnn.geometry` give the per-stage ring sizes. ⚠️ The known trap is
recorded in PLAN §9c: padding is a **per-layer** boundary condition, not a network-level
one, and the first Python attempt was bit-exact on every interior column and wrong on the
two edges. Only a bit-equality test caught it. Write that test first.

**C7. Parity.** Parametrise `tests/test_c_parity.py` over arms. Same bar as the linear
arm: identical gate masks and identical per-window detections against Python, on
synthetic audio and on 15 minutes of real audio where the gate actually fires, with the
closest window clear of numerical error by 10x.

**C8. Build and measure.** `./firmware/build.sh cnn`. Confirm on the linker map that our
sections are present and that stack headroom is real. Watch the ~6 KB figure above.

**C8b. ⚠️ The CNN arm does not keep up on hardware — open.** Flashed, the detector
updates roughly once every ten seconds rather than once a second, the power trace
cycles on the same ten-second period, and detections look unresponsive to birdsong.

*Found and fixed so far.* The convolution's inner loop was dequantising a stored
activation on every multiply-accumulate, so every output channel re-converted the same
bytes. The generated code per MAC was a hardware divide for the ring index, a byte
load, an integer-to-float convert, a subtract and two multiplies in a serial chain,
fourteen instructions with no fused multiply-add, about 940,000 times per window.
Dequantising the kernel's input columns once per output column instead leaves a
seven-instruction loop with the divide hoisted out. Numerically identical — parity is
unchanged to the digit. Costs 1,536 B of scratch, taking stack headroom to 3,840 B,
which is fine for the LED-only build (no filesystem linked) and needs re-checking
before `BG_LED_ONLY 0`.

*Not yet explained.* That should be worth several times, and the work itself is right —
939,984 MACs measured against a theoretical 995,360, so nothing is being done twice.
But no arithmetic I can do reaches ten seconds a window: the core runs from the 48 MHz
HFXO unless energy-saver mode halves it, and the honest estimate even before the fix
was around a second. Host timing cannot settle it either, because a laptop hides
exactly the divide and dependency stalls that cost a Cortex-M4.

*Confirmed compute-bound (device, 2026-09-22).* With `BG_PROFILE_LED 1` the red LED is
on continuously and the power trace never cycles, so the core never sleeps. The green
LED never lights, which the overrun guard explains: the detector is reading windows the
DMA has already overwritten, so the features are not of anything.

*Still unexplained, and now the whole question.* Counting instructions gets nowhere near
this. The core runs from a 48 MHz crystal (`EFM32_HFXO_FREQ`), halved only in
energy-saver mode. The convolution's inner loop is now seven instructions with a
loop-carried add, so ~10 cycles a multiply-accumulate, ~195 ms for the window's 940,000.
The front-end is ~110 ms on the same reckoning. That totals ~350 ms against a 1 s budget
— it should fit with room to spare, and it does not. Some assumption in that chain is
wrong by a large factor and no amount of further arithmetic will find which.

*Measured on the device, 2026-09-22.* `BG_SELFTEST`, ten windows per phase:

| | per window | my estimate | wrong by |
|---|---|---|---|
| front-end (FFT, filterbank, log) | **450 ms** | 110 ms | 4.1x |
| model (convolutions, quantisers) | **550 ms** | 250 ms | 2.2x |
| total | **1000 ms** | ~350 ms | against a 1000 ms budget |

So the arm sits exactly on the budget and tips over it, which is why it never sleeps,
never catches up, and ends up reading audio the DMA has already overwritten. And the
bad assumption was the **front-end**, not the network — the part shared with the arm
that works. The linear arm fits only because its model is a 40-weight dot product;
it has been running on ~55% headroom, not the ~90% the compute model claimed.

Worth keeping: naive instruction counting was wrong by 4x on code whose inner loops I
had read. On a Cortex-M4 with data in flash, no cache and no reordering, cycles are not
instructions — measure before optimising, and before believing a budget.

*Optimisations applied, all bit-identical* (parity numbers unchanged to the digit):

- **FFT twiddles hoisted out of the butterfly loop.** They were fetched from flash once
  per butterfly and indexed with a stride that defeats the prefetch; now once per stage.
- **Branchless Hann window** — two straight loops over the symmetric halves.
- **The 3-tap kernel unrolled.** The inner loop body was three instructions and the
  compare, branch and increment around it cost about as much again.
- **`roundf` inlined.** An out-of-line newlib call ran ~49,000 times per window.

*After those four, re-measured on the device:*

| | before | after | |
|---|---|---|---|
| front-end | 450 ms | **380 ms** | 16% faster |
| model | 550 ms | **410 ms** | 25% faster |
| total | 1000 ms | **790 ms** | 21% under the 1 s budget |

Not in the self-test and still to be paid in the real detector: `fetch_frame`'s 99,328
external-SRAM reads per window, the adaptive threshold's sort, and the gate. Call it
830-850 ms all in, so roughly 15% headroom — enough to keep up, not enough to be
comfortable, and it should be confirmed by the normal build running with the green LED
responding and the red one staying dark.

⚠️ **The self-test build must be gated on the switch.** The first one looped
unconditionally inside `makeRecording`, which never returns to the main loop, so the
switch was never re-read and the device could only be reflashed by pulling power and
racing the bootloader. It now runs only in CUSTOM and breaks out as soon as the switch
moves. Any future build that takes over the main loop needs the same escape.

*⚠️ A bookkeeping bug the LEDs exposed, 2026-09-22.* On the optimised build the device
showed red solid, green blinking like a clock, and steady power. Red solid should mean
the overrun guard had fired, but with `avail` counted the way it was — one buffer added
per pass of the read loop — the guard's condition is arithmetically unreachable, so the
two observations could not both be true of the code as written. That contradiction was
the finding.

`avail` was counting **what this loop had got around to reading**, not what the DMA had
captured. When the detector is slower than real time those diverge: the read loop
advances one buffer per pass and so does `avail`, while the DMA runs on ahead. Every
internal check stays self-consistent — `avail` and `next_end` track each other perfectly
— while the absolute read position drifts further and further into audio that has
already been overwritten. A guard comparing our own counter against our own counter
cannot see that, which is why it looked healthy while producing nonsense.

`bg_stream_set_captured` now takes the figure from the DMA transfer counter, so
`avail` is real elapsed audio and the guard compares a read position against the ring's
true extent. When the detector falls behind it now skips forward to the newest complete
window and says so, instead of quietly sliding into stale samples.

The LEDs were part of the problem: "wrong sample rate" was a slow green blink, which is
indistinguishable from a detector firing regularly. The chain now announces its state
before recording — two slow green flashes for armed at 32 kHz, six fast red for not —
and the wrong-rate signal is a fast *red* blink with green dark.

*⚠️ Why the CNN detected badly on hardware, even once it was keeping up.* With the
chain armed, red dark (no overruns) and the timing inside budget, the detector still
responded poorly to birdsong. The cause is not the port, which matches Python; it is
that **this arm depends on the absolute recording level and the linear arm does not**.

The network's input was standardised by one global mean fitted on the corpus, and the
input quantiser spans only about 46 dB (−70.5 to −24.3 dB of log-mel). A device quieter
than the training recordings walks off the end of that grid and the input saturates.
Measured in the simulator on 248 bird-seconds:

| acquisition level | CNN as built | CNN, level tracked | MFCC (linear) |
|---|---|---|---|
| 0 dB | 91.5% hit, 75 false | 87.9%, 31 | 86.7%, 51 |
| −6 dB | 79.0% | 87.9% | 86.7% |
| −12 dB | 61.7% | 87.9% | 86.7% |
| −20 dB | 19.0% | 87.9% | 86.7% |
| −30 dB | **0.0%** | 87.9% | 86.7% |

The linear arm is invariant because its features are spectral shape plus a log-energy
term the model barely weights; a level change is a constant the adaptive threshold then
removes. The CNN reads standardised log-mel directly and has no such protection. **The
four-arm comparison never tested this**, because every arm saw the same corpus at the
same level — so a property that decides whether an arm works in the field was invisible
to the whole study.

*Fixed by tracking the level.* `CnnParams.input_centre="ema"` replaces the fixed mean
with a causal per-window estimate, seeded from the first frame, updated once per window
from that window's own mean, applied to the window after. Per *frame* would be the
obvious choice and is not reproducible on the device, which computes 97 frames of every
100; per window uses exactly the frames the firmware has. Retrained and re-exported:
test retention 97.71%, saving 12.61%, species floor **95.70%** — slightly better than
the fixed-mean version's 95.29% — and now level-invariant. Parity improved too: zero
detection flips on both synthetic and real audio, against one before.

*If that is not enough,* in order of appeal: contract multiply-add (`-ffp-contract=fast`,
changes the last bit so parity must be re-measured); halve the decision rate to one
window every two seconds (free, but re-scales the adaptive ring and the gate constants,
so theta must be re-earned); or integer SIMD for the convolution, which is what CMSIS-NN
exists for and would be the real fix. **C10 was wrong to call that "if ever".**

*The measurement that settles it.* `BG_SELFTEST 1` ignores audio, the gate and the card,
and times the chain against a fixed synthetic signal: red is `BG_SELFTEST_WINDOWS`
windows of the front-end alone, green is the same count through the whole chain, one
second apart. Each LED's ON time divided by the window count is milliseconds per window,
and the difference between them is what the model costs. Built for both arms, so the
linear arm — which is known to keep up — calibrates the reading.

*A second fault this exposed, now guarded.* Falling behind did not merely make the
detector late, it made it **wrong**: `bg_stream` went on reading a window whose audio
the DMA had already overwritten, so the features came from a buffer being rewritten
underneath it. That alone would look like a model that had stopped recognising
anything. `bg_stream_advance` now skips to the newest complete window instead and
counts it, and in the normal build the red LED latches on the first time it happens.
Red dark means every green flash was computed from real audio.

*Consistent with the symptoms.* At one window per ten seconds the adaptive threshold's
300-window ring spans fifty minutes rather than five, so it stays near its seeded value
for most of an hour — which is why the detector appeared to grow more sensitive as it
ran rather than settling within minutes.

**C9. The device A/B.** Flash each arm in turn and compare on identical audio. The
protocol matters more than anything else here — see "Bench testing by playing recordings"
in `firmware/README.md`. Loop continuously with no gaps, or freeze the background after a
representative warm-up, and hold the clip, the level, the room and the device position
fixed between the two runs. Both arms share the adaptive threshold and the gate, so a
difference is attributable to the front-end and the model — but only under one protocol.

**C10. Integer requantisation, if ever.** Not needed to ship this arm: C6 already runs
the int8 grid, and the only thing left on the table is CMSIS-NN's integer-only
arithmetic, which buys speed on a part without an FPU. This one has an FPU. Revisit only
if C8 says the compute does not fit.

### What could make this not worth finishing

- **C2 moves the species floor the wrong way.** It is already the reason this arm does
  not ship. Re-read it before C3 and stop if it has got worse.
- **Stack headroom at C8.** If ~6 KB proves too tight, the 4 KB FFT buffer and the 2 KB
  spectrum can be overlapped, or the adaptive ring's sort scratch removed with an
  order-maintained insert. Both are known and neither has been needed yet.
- **C6 is the real work.** Everything else is plumbing that already has a shape.

