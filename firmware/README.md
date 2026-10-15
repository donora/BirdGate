# firmware/ — BirdGate on the AudioMoth

Bare-metal C mirroring `birdgate/` exactly, so the Python simulator and the device
agree. Two arms are ported: `mfcc-dev` (the one that ships) and `cnn`. The whole
chain — front-end, scorer, adaptive threshold and gate — is verified against Python on
the host and **runs live on an AudioMoth as a real-time detector**: the green LED
follows each one-second decision. The build is `BG_LED_ONLY`, so **it does not write
to the SD card yet**; the write path with pre-roll is next (`../ROADMAP.md` §1, F8).
This file is the reference for what exists and how to build it.

---

## What is verified

| | evidence |
|---|---|
| Build chain | `./firmware/build.sh stock` reproduces the official 1.12.1 release **byte-for-byte** (sha256 `8683ba69…51a42`) |
| Flash loop | a version bump to `9.9.9` was flashed and confirmed on hardware three ways — Configuration App, `CONFIG.TXT`, WAV GUANO block — then reverted |
| Model export | `model_params.h` generated from the trained model with every constant the C reads; the exporter refuses configurations the C does not implement |
| **Whole-chain parity** | `tests/test_c_parity.py` builds the harness from the device sources and runs audio through C and Python. **Gate masks identical** on both synthetic audio and 15 min of the busiest real recording. The int16 ring path the device uses is bit-identical to the float path |
| Device build | `./firmware/build.sh mfcc-dev` links the chain: 8,768 B of SRAM and 22.8 KB of flash are ours (linker map) |

Measured 2026-09-22, `mfcc-dev`, 900 one-second windows each:

| | synthetic | SNE_001, first 15 min |
|---|---|---|
| gate mask | identical | identical |
| windows recorded | 72 / 240 | 659 / 900, 3 openings |
| feature max abs diff | 1.21e-03 | 4.27e-04 (~2e-6 relative) |
| z max abs diff | 2.49e-04 | 2.84e-06 |
| closest window to theta | 9.05e-03 | 2.24e-04 |
| headroom over numerical error | 36x | **79x** |

⚠️ **Always run parity on audio where the gate actually fires, at the deployed
threshold.** Two earlier runs looked better than they were: a quiet-audio run agreed
perfectly because both sides recorded *zero* windows, and the run that produced the
"628x headroom" figure these docs used to quote was taken near theta = 0, where the
gate is open 99.9% of the time and agreement mostly proves both sides agree about
"on". At the deployed theta the real headroom is 79x — comfortable, and an order of
magnitude less than was previously claimed. `test_c_parity.py` now asserts both
conditions: the gate must fire, and the closest window must sit at least 10x the
numerical error away from theta.

In that 15-minute stretch the gate retains **100% of the 248 annotated bird-seconds**
at a 73% duty cycle. It is the busiest recording in the corpus, so that duty is
expected and is not a saving figure.

## Rebuild everything from cold

```sh
cd AM_Software                       # if the clones are missing
git clone https://github.com/OpenAcousticDevices/AudioMoth-Project.git
git clone https://github.com/OpenAcousticDevices/AudioMoth-Firmware-Basic.git
cd ..

./firmware/build.sh stock            # must say "reproduces ... byte-for-byte"
../.venv/bin/python -m scripts.export_model --arm mfcc-dev    # regenerate model_params.h
../.venv/bin/python -m pytest tests/test_c_parity.py -s       # host parity, whole chain
./firmware/build.sh mfcc-dev         # the device build -> out/birdgate-mfcc-dev.bin
```

---

## Bench testing by playing recordings

A device fed recordings through a speaker fires far more than the simulator does on
the same audio, and it fires on the *background* of the recording. This is not the
model and not the port. It is the adaptive threshold meeting a signal whose statistics
no field recording has.

**The mechanism.** A window fires when its score sits more than `theta * scale` = 0.50
above the background anchor, and the anchor is the 0.2 quantile of the last 300 s — the
quietest fifth of the last five minutes. Measured on this corpus:

| | score |
|---|---|
| a quiet indoor room (stationary low-frequency rumble) | −5.57 |
| the recording's own forest background | −0.60 |
| birdsong in the recording | +2.00 |

Leave the room silent between takes and that silence *is* the quietest fifth, so the
anchor sits near −5.57 and the recording's own background lands 5.0 above it — ten
times the 0.50 needed to fire. Everything lights the LED. Measured, playing the same
five-minute clip four times over:

| quiet between takes | windows firing during playback |
|---|---|
| none, looped back to back | 26 / 1200 — matches the simulator |
| 30 s | 299 / 1200 |
| 60 s or more | 992 / 1200, essentially all on background |

The error budget is small in both directions: an anchor only 0.25 too low already puts
the LED on half the time, and 1.0 too low pins it on permanently.

**What does *not* cause this**, all measured and all absorbed by the adaptive threshold
exactly as intended: playback level (the model's weight on overall level is 0.005, two
orders below its shape weights), speaker roll-off below 300 Hz, and room reverberation.
A stationary change to the acoustic path is the one thing CFAR is built to remove. Only
*non-stationarity* — the silence between takes — breaks it.

**So test with continuous audio.** Loop the recording with no gaps, and give it about
five minutes before believing the LED, so the ring flushes whatever it heard first. Under
those conditions the device is testing what the simulator tested.

**If gaps are unavoidable** — poking it with live whistles, say — two knobs in
`app/main.c`, both off by default:

- `BG_ADAPT_FREEZE_S` — hold the background after this many seconds. Warm it up on
  representative ambience first, because whatever it hears in that window becomes the
  reference permanently. This is the one that makes repeated poking reproducible.
- `BG_THETA_OFFSET` — added to theta. Blunt compensation; the right value depends on how
  disciplined the gaps are, so prefer the freeze.

⚠️ **The ring is slow to start.** It is pre-filled with 300 copies of the first window's
score, so the anchor cannot move at all until more than a fifth of those are overwritten
(60 s) and is not clear of them until 300 s. A device that wakes into an unrepresentative
moment carries it for minutes. In the field the same pattern appears at dawn: the trailing
five minutes are night silence, so the chorus onset is over-recorded for a few minutes.
That errs toward recording, which is the safe direction here, but it is unmeasured —
ROADMAP §2 carries it as an open question.

## Where this plugs into stock AudioMoth

Stock Firmware-Basic already has almost all of the data path. Samples arrive by DMA
into internal SRAM and are copied into an 8-slot ring in external SRAM (8 × 16,384
samples, 0.512 s per slot at 32 kHz). A per-slot `writeIndicator` flag, set by the
stock amplitude/frequency trigger, decides whether each slot is written to the card
as the read pointer advances. BirdGate replaces that trigger:

```
MEMS mic --DMA--> internal SRAM (ping-pong) --> 8-slot ring (external SRAM)
                                                   |  writeBuffer
                                                   v
                       read cursor --> bg_fe_* (arm.h)    1 s of samples -> features
                                                   |
                                                   v
                                            bg_score()      w·φ + b
                                                   |
                                                   v
                                            bg_adapt()      CFAR: subtract trailing
                                                   |        0.2-quantile, /global scale,
                                                   v        max over last 3
                                            bg_gate_step()  hysteresis / pre-roll /
                                                   |        hangover / min-on
                                    open? --yes--> mark slots indicated (rewind the
                                                   |   read cursor by pre_roll on open)
                                     no ---------> leave slots unindicated
```

The one change to the stock write path: on a gate *open* transition, rewind the read
cursor by `pre_roll` slots before marking. The buffer already holds that audio, so
pre-roll costs zero extra memory — see `birdgate/gate.py`.

## Layout

Split by **what varies between arms**. Everything that actually differs is *data* —
weights, theta, the adaptive threshold's `global_scale` — and all of it is generated
into one per-arm header. The logic that consumes it is shared.

| path | | mirrors |
|---|---|---|
| `common/gate.c` | shared | `birdgate/gate.py` — **must be bit-identical in behaviour** |
| `common/adapt.c` | shared | `birdgate/adapt.py` — quantile background, global scale, max-of-3 |
| `arms/mfcc-dev/scorer.c` | per arm | `birdgate/svm.py` — `w·φ + b` |
| `common/fft.c` | shared | self-contained radix-2 FFT, plus the windowed real-input power spectrum both arms start from |
| `common/arm.h` | shared | the four streaming calls plus a scorer that every arm provides |
| `common/birdgate.c` | shared | the chain behind one call: features → score → z → gate |
| `common/bg_stream.c` | shared | frames 1 s windows out of the device's int16 sample ring |
| `arms/<arm>/model_params.h` | per arm | generated by `birdgate/export.py`; do not hand-edit |
| `arms/cnn/logmel.c`, `cnn.c` | per arm | `birdgate/frontends.py::logmel(dev=True)` and `birdgate/cnn.py` — 32-band log-mel, then the streamed convolutions |
| `arms/mfcc-dev/mfcc.c` | per arm | `birdgate/frontends.py` for `kind="mfcc-dev"` — streaming: real-input FFT, mel, log, DCT, Welford mean/std |
| `app/main.c` | shared | Firmware-Basic's `main.c` with the chain hooked into the recording loop |
| `test/test_parity.c` | | host harness; `tests/test_c_parity.py` drives it |

`common/` means "shared by everything that *can* share it", not shared by definition:
a non-linear arm brings its own scorer, so a CNN port would get one under `arms/cnn/`.

The state machine is single-sourced because of `gate.py`'s opening note: `gate.c` is
mirrored from it exactly, and **if the two drift, the simulator lies and every number
in every report is wrong**. One copy is one parity obligation instead of N.

### `app/main.c`

Four anchored insertions into stock, so an upstream update can be re-applied by hand:
the include block with the `BG_LED_ONLY` stubs; `bg_stream_init` after the DMA is
initialised; the per-slot `bg_stream_samples_arrived` / `bg_stream_advance` loop in
the write loop, driving `AudioMoth_setGreenLED`; and the LED off before the file is
closed. `BG_LED_ONLY 1` turns every SD-card call into a no-op that reports success, so
the milestone build runs the chain with no card and writes nothing. The chain only runs
when the configured effective sample rate is 32 kHz; otherwise the red LED blinks fast
with green dark, so a misconfigured device is obvious rather than silent and cannot be
mistaken for a detector firing.

Two meanings for the green LED, chosen by `BG_LED_RAW`:

| | LED is on while | good for |
|---|---|---|
| `BG_LED_RAW 1` | this one-second window scores above theta (`z_inst > BG_THETA_ON`) | testing the classifier — try a sound, see the answer, try again |
| `BG_LED_RAW 0` | the gate is open: hysteresis, hangover, 20 s minimum-on | testing the recording policy — what reaches the card |

Both read the same score through the same adaptive threshold. The detector deliberately
uses `z_inst` rather than the gate's max-of-`BG_AGG_N` pooled `z`, which would hold the
LED on for two further seconds after every detection. `tests/test_c_parity.py` pins the
detector signal against Python in its own right, not merely as a by-product of the gate
mask, because it is what a person reads the model's behaviour off.

On 15 minutes of the busiest recording the detector lights on 87% of annotated
bird-seconds (215 of 248) and on 51 further seconds, some of which are birds the strong
annotations do not cover. The gate, over the same audio, records 659 seconds — the
difference between the two columns above.

## Building

    ./firmware/build.sh stock       # vendor only — reproduces the official 1.12.1 release
    ./firmware/build.sh mfcc-dev    # vendor + common/ + arms/mfcc-dev/  -> out/birdgate-mfcc-dev.bin

`build.sh` assembles `AM_Software/AudioMoth-<arm>/` from the two pristine upstream
clones plus `firmware/`, then builds. Those trees are **disposable output** — every run
deletes and recreates them, so an edit made inside one vanishes without warning. Edit
here; the trees are generated.

`stock` is the regression check on the assembly itself: it must keep reproducing the
official release byte-for-byte. Run it whenever the vendor clones or `build.sh` change —
if it ever stops matching, every arm built from that chain is suspect.

Override the toolchain with `TOOLCHAIN_PATH=... TOOLCHAIN_VERSION=... ./firmware/build.sh`.

Check what a build actually contains with
`arm-none-eabi-size AM_Software/AudioMoth-<arm>/build/audiomoth.axf` — if `text` and
`bss` equal stock's, our code was garbage-collected because nothing calls it.

## Two arms

`./firmware/build.sh mfcc-dev` and `./firmware/build.sh cnn` produce the two detectors.
They share everything except the front-end and the scorer, which is what `common/arm.h`
buys: `app/main.c`, the gate, the adaptive threshold and the ring framer are identical
and unaware of which is running.

| | `mfcc-dev` (ships) | `cnn` |
|---|---|---|
| front-end | 128 mel bands, DCT, 20 MFCCs, window mean and std | 32 mel bands, log, no DCT |
| model | 40 weights and a bias, float | three 3x3 convolutions on the int8 grid, 5,865 weights |
| flash | 125,700 B | 140,892 B |
| RAM used / stack left | 24,484 / 8,284 B | 27,392 / **5,376 B** |
| agreement with Python | features 4e-4, masks and detections identical | masks identical, 1 detection in 900 flips |

⚠️ **The CNN arm cannot be bit-identical to its reference, and that is structural.**
Quantisation turns a seventh-digit difference in summation order into a whole-step jump,
so the device and the simulator disagree on windows that sit within numerical error of
the threshold. Measured: one window in 900, and the gate masks still match because the
state machine smooths isolated flips. The linear arm has no such amplifier and runs with
79x headroom. Treat the CNN's simulator numbers as accurate to about a window in a
thousand rather than exact.

## Budget

Measured from the linker map (`AM_Software/AudioMoth-mfcc-dev/build/audiomoth.map`),
our objects only. Stock leaves ~14.7 KB of the 32 KB internal SRAM free.

| | ours | LED-only build total | full-recording build total |
|---|---|---|---|
| SRAM `.bss` | **8,768 B** — 4 KB frame, 2 KB power spectrum, 2.4 KB adaptive-threshold ring + sort scratch | 20.3 KB `.bss` + 1.1 KB `.data` + 3 KB heap = 24.5 KB → ~8.3 KB for stack | 26.9 KB → ~5.9 KB for stack |
| Flash | **22.8 KB** — 10.2 KB DCT, 4 KB twiddles, 3.5 KB mel, 2 KB Hann, 2 KB code | 126.8 KB `.bin` | 151.0 KB `.bin` of 240 KB |
| Compute | ~97 HALF-point complex FFTs + mel + DCT per second, estimated 2–3 % of the 48 MHz M4F; one 300-float sort per second | | |

Check any change with `arm-none-eabi-size` on the `.axf`, not by arithmetic — if
`text` and `bss` equal stock's, our code was garbage-collected because nothing calls it.

⚠️ A CNN port's SRAM figure depends entirely on **execution order**: build each feature
map whole and it needs 23.3 KB, stream it one log-mel frame at a time and it needs
7.1 KB for bit-identical output and the same MAC count. `birdgate/cnn.py::stream_forward`
is the reference the C would have to match. See docs/lab-notebook.md §9c.

## Deliberately not done

- **Quantisation.** Float, on purpose. The M4F has a hardware FPU and this arm uses a
  few percent of it. Quantising *while* integrating would confound a port bug with
  fixed-point drift — integrate first, prove it works, then quantise as an isolated
  change with a measured before/after, if ever.
- **32 mel bands.** A flash saving and a feature change; flash is not the constraint.
- **Firmware identity.** `firmwareDescription[]` is still `"AudioMoth-Firmware-Basic"`.
  Changing it puts BirdGate provenance into `CONFIG.TXT` and every WAV's GUANO block at
  one line — but **untested** whether a non-standard description costs Configuration
  App support. Settle it on hardware before committing (ROADMAP H3).

---

## Build environment (verified 2026-09-14)

`AM_Software/` is gitignored — upstream clones and generated trees, never our source:

| | |
|---|---|
| `AudioMoth-Project` | upstream skeleton, kept **pristine**; `build.sh` copies from it |
| `AudioMoth-Firmware-Basic` | upstream app layer, master = 1.12.0 |
| `AudioMoth-Firmware-Basic-1.12.1.bin` | the official release — rollback, and the build oracle |
| `AudioMoth-<arm>/` | generated build trees, disposable |
| toolchain | `~/Library/Toolchains/gcc-arm-none-eabi-10.3-2021.10` (x86_64, via Rosetta 2) |

Three things worth knowing, all handled by `build.sh`:

- `AudioMoth-Firmware-Basic` master is **1.12.0**; the released `.bin` is **1.12.1**, and
  the only difference between them is the version triple at `main.c:1480`. GitHub does not
  currently carry source for the exact binary OAD ships — it carries source for something
  one byte away from it. `build.sh stock` patches that byte so the comparison is exact.
- Firmware-Basic ships its own `gps.c`/`gps.h`, which are **not** the same module as the
  Project's `gps/` folder (`gpsinterface`, `nmeaparser`, `gpsutilities`, `sunrise`). No
  basename collision; both are needed, and `build.sh` verifies the Makefile patch landed.
- The toolchain path must be absolute — `make` will not expand `~`.

Flashing uses the OAD AudioMoth Flash App over USB.

Upstream code is MIT-licensed (Open Acoustic Devices, 2017); a fork must keep that
notice.
