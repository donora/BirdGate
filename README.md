# BirdGate

**A tiny, always-on model that decides, second by second, whether an AudioMoth
should be recording birdsong — in 164 bytes.**

Developed in collaboration with a rewilding project in the Scottish Highlands.
**Work in progress:** the results so far are preliminary, from one dataset;
see [the note below](#the-four-approaches).

**[Take the tour →](TOUR.md)** A walk through the problem, the four approaches
and their audio maths, the silicon budget, and the firmware, with the app as the
guide.

![The gate, end to end: one second of audio becomes features, a score, an adaptive threshold and a record-or-skip decision](docs/tour/flow.svg)

---

## Why tiny

Passive acoustic monitoring is heading towards very large numbers of cheap,
low-power recorders left in the field for months. Most of the attention goes to
large models — BirdNET, Perch — that identify hundreds of species after the fact,
on a server. That leaves a second problem at the other end of the scale: what can
a 48 MHz microcontroller running on three AA batteries decide *for itself*,
reliably, every second, with no second chance at the audio it throws away?

BirdGate is a study of that kind of insect-scale intelligence. It answers one
small question — *is there birdsong right now?* — well enough to trust in the
field, so the recorder writes only what is worth keeping. At a busy site that
saves card and listening time; at a sparse one it stretches the battery by over a third.

## The four approaches

> [!NOTE]
> **Work in progress: these are preliminary results.** Every model was trained
> and tested on one dataset: 33 hours from one area of the Sierra Nevada, all
> recorded in May 2018, with 21 recordings for training, six for tuning and six
> held out. The CNN figures come from a single training run of a network that has
> had little tuning, and it could improve substantially with more varied data and
> better training. Treat the comparison as where things stand today, not a verdict
> on which approach is best. Retraining on Highland recordings is planned.

Four front-ends were compared under identical controls — the same one-second
windows, the same recording-level split, the same gate and threshold policy and
the same metric code — so any difference comes from the one thing that changed.

| approach | what it computes | model | silence skipped* | birdsong missed* | worst species kept* | verdict |
|---|---|---|---|---|---|---|
| **`mfcc`** | FFT → mel filterbank → 20 MFCCs, mean and spread over the second → linear score | **164 B** | 28% (14–35%) | 1.7% (worst 5.6%) | **96.9%** | **ships** — best on quiet species, cheapest to run |
| `bands` | 40 independent bandpass filters, rectified and summed | 172 B | 32% (5–51%) | 1.6% (worst 7.7%) | 96.3% | similar on average at 5× the compute, erratic recording to recording |
| `carihc` | the same 40 channels as a cascade of resonators — a model cochlea, after Lyon's CARFAC | 172 B | 11% (0–22%) | 1.9% (worst 13.9%) | 94.1% | last on every metric; the cascade does not earn its keep |
| `cnn` | a small int8 convolutional network over log-mel | 5.9 KB | **53%** (16–67%) | 2.0% (worst 5.1%) | 92.2% | best detector, but drops more of the quietest species; runs on the device in 790 ms per 1 s window |

\* On the six held-out recordings, with the threshold set to keep 98% of birdsong
seconds overall. **Silence skipped** is the share of the skippable silence the gate
skips, where skippable is what a perfect detector skips behind the same gate; only
birdsong-free seconds count, and the bracket is the range across the six recordings.
How much silence there is to skip depends on the hour: 7% to 33% of the card here.
**Birdsong missed** is the share of annotated birdsong seconds not written to the
card, overall and on the worst recording. **Worst species kept** is the lowest
retention among species with enough calls to measure. In card terms MFCC saves
7.7% and the CNN 13.2%. Computed from the bundled scores by
`scripts/tour/scoreboard_numbers.py`; [SUMMARY.md](SUMMARY.md) has the full tables
(its CNN figures come from a different training seed).

## What we found

1. **Measure the ceiling before building the model.** Running ground truth
   through the real gate showed the best possible saving (35.1%), and showed that
   99.7% of seconds kept can still clip 8.3% of whole calls.
2. **The cochlea loses, and MFCC is cheaper than its reputation.** At 32 kHz a
   filter cascade costs per *sample* while an FFT costs per *frame*, so MFCC runs
   5× cheaper — the usual microcontroller argument inverts above 8 kHz.
3. **A fixed threshold fails silently.** At one sensible threshold, retention
   ranged from 54.9% to 99.3% across recordings while the average looked fine.
   A CFAR threshold, borrowed from radar, collapses that spread to under 5 points.
4. **The metric that binds is the worst species, not the average.** A gate that
   keeps 99% of a loud species and 45% of a quiet one quietly biases every later
   analysis. That metric is why the CNN does not ship.
5. **Memory estimates depend on schedule.** The CNN "needed" 23.9 KB of RAM;
   streamed a column at a time, the identical network needs 7.1 KB.
6. **The battery runs out before the card does.** At this site the gate buys
   1.6% more battery life; at a sparse site, about 39%.

## Status

- **Python study:** complete. Every headline number re-derives from the cached
  features.
- **Firmware:** both `mfcc` (as `mfcc-dev`) and `cnn` are ported to C, match
  Python window for window on the host, and **run live on an AudioMoth as
  real-time detectors** — the green LED follows each decision.
- **Next:** write gated audio to the SD card, measure energy on the bench, run a
  side-by-side field trial, then retrain on Highland recordings.
  [ROADMAP.md](ROADMAP.md) has the order.

## Run the app

The repository carries a 12 MB demo bundle (`demo/`: frozen model scores, linker
maps and eight 30 s clips), so the app runs without the 3 GB dataset.

```bash
git clone https://github.com/donora/BirdGate.git && cd BirdGate
python -m venv .venv && .venv/bin/pip install -e '.[app]'
.venv/bin/streamlit run app.py
```

To rebuild everything from the audio, download `soundscape_data.zip` from
[Zenodo](https://doi.org/10.5281/zenodo.7050014), point `data/audio` at the
unzipped folder, and:

```bash
.venv/bin/pip install -e '.[cnn,app,dev]'
.venv/bin/python -m scripts.build_cache              # every arm's features, once
.venv/bin/python -m scripts.compare_arms             # the linear arms, one table
.venv/bin/python -m scripts.train_cnn --save-scores  # the CNN (~20 min on CPU)
.venv/bin/python -m scripts.build_demo --clips       # refresh demo/
.venv/bin/pytest -q
```

## Put it on an AudioMoth

You need an AudioMoth 1.x, the Open Acoustic Devices
[Flash and Configuration apps](https://www.openacousticdevices.info/applications),
and, to build from source, `gcc-arm-none-eabi-10.3-2021.10`. Each release carries
a prebuilt detector `.bin`. **It lights the green LED on birdsong and does not
record to the card yet.** Building, flashing, the host parity test and bench
testing are in [firmware/README.md](firmware/README.md); the walkthrough is the
last chapter of the [tour](TOUR.md).

## Repository

| | |
|---|---|
| [TOUR.md](TOUR.md) | the guided tour |
| [SUMMARY.md](SUMMARY.md) | the findings in full, with the maths in Appendix A |
| [ROADMAP.md](ROADMAP.md) | what comes next |
| [docs/lab-notebook.md](docs/lab-notebook.md) | the complete working record, every experiment and table |
| `birdgate/` | front-ends, models, adaptive threshold, gate, metrics, silicon budget |
| `firmware/` | the C port, build script and parity harness |
| `app.py` | the Streamlit explorer |
| `data/`, `demo/` | labels and the demo bundle (CC BY 4.0, see below) |

## Data and licence

Code: MIT ([LICENSE](LICENSE)); the AudioMoth firmware it builds on is MIT,
© Open Acoustic Devices ([firmware/LICENSE-AudioMoth](firmware/LICENSE-AudioMoth)).

Data: Kahl, Wood, Chaon, Peery & Klinck (2022), *A collection of fully-annotated
soundscape recordings from the Western United States*, Zenodo,
[10.5281/zenodo.7050014](https://doi.org/10.5281/zenodo.7050014), CC BY 4.0 —
33 hours from the Lassen and Plumas National Forests, California, with 20,147
annotated calls. The labels, the derived scores and the short clips here are
redistributed under the same licence; [data/ATTRIBUTION.md](data/ATTRIBUTION.md)
lists what was changed.

## Acknowledgements

The idea of a tiny always-on classifier in front of the AudioMoth's SD write comes
from Sabbella et al., *An Always-On tinyML Acoustic Classifier for Ecological
Applications* (ISCAS 2022). This project kept the idea and rebuilt the rest. The
cochlea arm follows Richard F. Lyon's CARFAC. The hardware and its firmware are
[Open Acoustic Devices](https://www.openacousticdevices.info/)' AudioMoth.
