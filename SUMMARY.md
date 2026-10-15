# BirdGate — what we did, what we found, why it matters

> Each section is the argument and the conclusion; the supporting tables live in
> `docs/lab-notebook.md`, and each section says where.

---

## 0. The one-paragraph version

An AudioMoth left in a forest records continuously, fills its SD card with wind and
silence, and buries the few minutes that matter. BirdGate is a classifier small enough to
run always-on inside the recorder that answers one question every second — *is a bird
singing right now?* — and writes to the card only when the answer is yes. We compared four front-ends
under controls — a neuromorphic cochlea (the inspiration, after Sabbella et al., ISCAS
2022), a plain bandpass bank, MFCCs and a tiny CNN — all trained on 33 hours of
continuously annotated soundscape rather than curated clips. **The cochlea turned out to
be the worst of the four, and a plain bank of bandpass filters beat it at lower cost.** Three further findings matter more than
the arm ranking: a single fixed threshold does not transfer between recordings and would
have failed silently in the field; the usual data-scarcity case for preferring a tiny linear
model does not survive contact with abundant data; and — found last, and the most
uncomfortable — **the quantity we spent the project optimising is a small share of the
device's energy budget**, which sits mostly in firmware power management nobody costed.

**Everything below reproduces.** Every headline number in this document was re-derived
from the cached features and logits today, from scratch, and matched to the digit.

---

## 1. The setup, and why it is not what it looks like

**Exec summary.** The dataset is Sierra Nevada, but the target is a rewilding project in
the Scottish Highlands. SNE is the test rig — 33 h of continuous soundscape with 20,147
strongly-localised annotations — chosen because it is the only way to answer *can
continuous field recordings train a gate this small?* before labelled Highland audio
exists. That framing drove every design choice: where a parameter could be tuned to
squeeze another 2% out of Sierra Nevada, we took the choice that transfers instead.

**The three constraints:** don't miss birdsong (discarded audio does not exist and no
later cleverness recovers it); byte-for-byte repeatable; and traceable — any recording on
a recovered card must resolve back to the exact model and threshold that let it through.

**So what:** the deliverable is *a method that survives a change of continent*, not a
leaderboard number. That is why the controls exist and why the negative results below are
the valuable part.

*Detail: docs/lab-notebook.md §4 (the 32 kHz decision) and §5 (what it cost the pre-roll budget).*

---

## 2. What was built

**Exec summary.** A complete Python pipeline — windowing, exact continuous-time interval
arithmetic, the gate state machine, metrics, adaptive thresholding, four model arms, a
provenance layer, and a Streamlit app to turn every knob interactively — plus a C
scaffold for the firmware port. 27 tests pass, 16 of which validate the cochlea front-end
against signals whose answer is derivable by hand.

**So what:** the front-end is tested in isolation because an IIR cascade degrades
*silently*, and the loss then gets blamed on the classifier. That is a general lesson, not
a project detail.

### The four arms, and why each one exists

An **arm** is one variant in a controlled comparison — the term is borrowed from
clinical trials, and it carries the same discipline: everything except the named
difference is held identical. All four share the same 1 s windows, the same
recording-level split, the same gate state machine, the same adaptive threshold and
the same metric code, so a difference in the result is attributable to the one thing
that changed.

| arm | front-end algorithm | classifier | size | why it is here |
|---|---|---|---|---|
| `carihc` | Cascade of 40 asymmetric 2-pole-2-zero resonators, ERB-spaced 12 kHz→150 Hz. Each stage's output is half-wave rectified (the "inner hair cell"), summed over the window, L1-normalised, plus log-energy and log-flatness → **42 features** | linear SVM | 172 B | the neuromorphic cochlea — the front-end that inspired the project |
| `bands` | Identical, except the 40 biquads are *independent* ERB-spaced bandpasses rather than a cascade. Same rectification, same summing, same normaliser → **42 features** | linear SVM | 172 B | **the control that matters.** Only the cascade-and-zeros differs, so it isolates what the cascade itself contributes |
| `mfcc` | 1024-pt FFT → mel filterbank → 20 MFCCs at 100 fps; mean and standard deviation of each over the window → **40 features** | linear SVM | 164 B | the standard front-end, **usually rejected on cost for this class of device**. Tests whether that argument holds at our sample rate |
| `cnn` | 32-band log-mel patch, 32×100 per window, globally standardised. Three conv blocks (8→16→32 channels, 3×3, BN+ReLU+maxpool), global average pool, linear → 1 logit | tiny CNN, int8 | 5.9 KB | tests whether a tiny linear model is the right **family** once labelled data is abundant |

All three linear arms use the same classifier: `StandardScaler` → `LinearSVC` (hinge
loss, C=1), with the scaler folded into the weights afterwards so the deployed model is
a bare dot product — `P` weights and a bias, and nothing to normalise at run time.

**→ [Appendix A](#appendix-a--the-three-linear-arms-in-detail) works through the three
linear arms properly** — the filter mathematics, the cochlear physics being imitated,
what the DCT is actually for, and why the cost comparison inverts at 32 kHz.

**The logic of the set.** `carihc` is the headline and the other three are what make it
falsifiable:

- **`bands` isolates the cascade.** The cochlea's claim is not "a filterbank" — it
  is specifically a *cascade of asymmetric resonators*, where each stage filters the
  previous stage's output and the zeros sit above the poles. `bands` keeps the channel
  count, the spacing, the rectification, the window summing and the normaliser, and
  removes only that. If it ties, the cochlea is decoration; if it wins, the cochlea is
  worse than decoration. (§4 reports which.)
- **`mfcc` tests the cost argument, not the accuracy argument.** The usual case against
  MFCC on a microcontroller is that it needs an FFT and log-domain ops. That is a claim about a *device*, and
  device claims are sample-rate dependent — worth checking rather than inheriting.
- **⚠️ `cnn` is deliberately the weakest evidence in the set, and is reported as such.**
  Arms 1–3 vary *only* the front-end, so a difference between them is attributable. The
  CNN varies the front-end **and** the classifier, so if it wins it cannot tell us which
  change did it. That is accepted on purpose: it asks "would a different approach be
  better?" rather than "why?", and that is the right question to ask of a design
  premise.

*Detail: docs/lab-notebook.md §13 for the module map by stage; the app is `app.py`; the cost model is `frontends.cost_model`.*

---

## 3. Finding 1 — Measure the ceiling before you build the model

**Exec summary.** Before any model existed we ran the ground truth through the real gate
state machine. This fixes the denominator for everything afterwards: a perfect detector
on this data saves **35.1%** of the card at 100% retention. So "we saved 8%" is not a
result; "8% of an achievable 35%" is.

It also separates state-machine wins from model wins. Pre-roll of 1 s and hangover of 1 s
is the knee — perfect retention, perfect whole-box capture — and every longer hangover
costs saving and buys literally nothing. Fixing that in G1 means we could never later
credit a state-machine change to the model.

**Two things the oracle exposed that we would otherwise have missed:**
- Second-level retention reads a comfortable 99.7% while **8.3% of annotation boxes are
  being clipped**. Retention is dominated by a few very long boxes; event recall weights a
  0.3 s chip equally. The box metric is the binding one.
- Retention *cannot* reach 100% at the tightest setting — it caps at 99.71%, and that
  residue is the labelling rule, not the gate. Every model number has to be read against
  that, not against a naive 100%.

**So what:** this is the cheapest and most reusable idea in the project. An oracle run
costs an afternoon and prevents a whole class of overclaiming.

*Detail: docs/lab-notebook.md §6 — the oracle table, the 1.4–86.1% occupancy range, and the files-per-hour argument for a minimum-on time.*

---

## 4. Finding 2 — The cochlea does not earn its keep

**Exec summary.** The front-end that inspired the project is a cascade of asymmetric
resonators — a neuromorphic cochlea. We built it from scratch, validated it to machine precision, and
then built the control that matters: a bank of *independent* ERB-spaced bandpass biquads,
same channel count, same rectification, same summing, same normaliser. Only the cascade
differs.

**The cochlea loses on every metric, at every operating point, in every table, at higher
compute cost.** It is last on every safety metric. MFCC — usually rejected on cost for
this class of device — also beats it, and runs **5× cheaper** at our sample rate,
because the cascade's cost scales with the sample rate while the FFT's scales with the
frame rate. At 8 kHz that argument holds; at 32 kHz it inverts.

**So what:** this is a controlled negative result, and it is directly actionable — the simpler filterbank is also far easier to get right in fixed
point, so choosing it removes most of the firmware risk rather than adding to it.

*Detail: docs/lab-notebook.md §7 (the validation tests and the two-ERB-definitions trap) and §8 (the arm table).*

---

## 5. Finding 3 — A fixed threshold would have failed in the field, silently

**Exec summary.** This is the most important result in the project. At one threshold
chosen properly on held-out data, per-recording retention swung from **54.9% to 99.3%**.
The aggregate looked fine at 96%. On one recording the device would have thrown away
nearly half the birdsong while reporting a healthy 77% saving.

**Why it happens:** gain, microphone, weather and background level move the whole score
distribution bodily, so one absolute threshold means something different in every hour.
The fix is borrowed from radar — CFAR: threshold the score *relative to a trailing
estimate of the local background*, so when it starts raining the gate simply demands a
correspondingly louder bird. No retraining, no site survey. **The 44-point spread
collapses to under 5 points**, for a cost of a few hundred bytes.

**So what:** in an unattended months-long deployment this failure is invisible and
unrecoverable — there is nobody there to notice. Bounded worst-case behaviour is worth
far more than the couple of points of card it costs. This is also the finding most likely
to transfer to any always-on acoustic sensor, not just this one.

*Detail: docs/lab-notebook.md §9 — the mode comparison, the corrections, and the final tables.*

---

## 6. Finding 4 — Two subtleties inside that fix, both counterintuitive, both measured

**Exec summary.** Getting adaptive thresholding *nearly* right made things worse, in two
ways that would have been easy to ship.

**(a) Do not divide by a local spread.** The obvious formulation standardises the score:
subtract the local background, divide by the local spread. Measured, this destroyed
within-recording discrimination — on one training recording the raw model separated
cleanly at AUC 0.925 and the normaliser dropped it to 0.74. In a long quiet stretch the
local spread collapses, so dividing by it amplifies meaningless fluctuation into huge
scores and the tail of *silence* overtakes real birds. The symptom is a gate that chatters
through precisely the silence it exists to skip. **What varies between recordings is the
offset, not the spread** — so subtract locally, scale by one global constant fitted on
training data. Caught by eye in the app, not by a metric.

**(b) The anchor quantile is a detector/duty-cycle-controller switch.** Anchor the
background at the median and the gate records ~75% of pure silence regardless of content —
it stops being a detector at all and becomes a device that records q% of the time. At 0.20
it records 5%. Same code, one number, completely different machine.

**(c) `max` pooling beats `mean`, for the opposite reason to the obvious one.** Requiring
sustained evidence suppresses short calls, so the threshold has to come *down* to keep
recall, and recorded silence rises. `max` makes the score more sensitive, so the same
recall is reachable at a *higher* threshold — and the higher threshold is what rejects
silence.

**So what:** three cases where the principled-looking choice measures worse. Worth saying
out loud because each was a coin-flip decision that a reasonable person would have made
the other way and never checked.

---

## 7. Finding 5 — The CNN wins on detection and still does not ship

**Exec summary.** The usual case for a tiny linear model on this hardware is **data
scarcity** — a few dozen clips per class. We have ~119,000 labelled seconds. So we built the fourth arm: a 5.9 KB int8 CNN
over log-mel, using the one thing every other arm throws away — the time structure
*inside* the second.

**It out-detects every linear arm by a wide margin.** At matched 98% retention it saves
**12.2%** where the best linear arm manages 8.0%, with a better worst-recording floor
(97.1% vs 92.3%), and the gap *widens* as the retention target rises — which is the
direction that matters. Nine times its own seed-to-seed standard deviation, so not an
artefact. Post-training int8 quantisation is free.

**And we still recommend shipping the linear arm** — because it loses on the metric that
binds (§8), not on detection.

**So what:** the small model is the right choice on this hardware, but not for the reason
usually given. Data scarcity is not why it wins here. Knowing the actual price of the
choice (~4 points of card) is more useful than either verdict alone, and it only became
visible because we tested the argument rather than only the model.

*Detail: docs/lab-notebook.md §9b (architecture, the four-arm table, int8 parity, the MPS finding) and §9c (streaming).*

---

## 8. Finding 6 — The metric that actually binds is the one nobody reports

**Exec summary.** A gate at 97% overall retention that keeps 99% of a loud abundant
species and 45% of a quiet one has injected a **species bias into every analysis those
recordings will ever support** — invisibly, because the discarded audio is gone. So we
report the *minimum* per-species retention, not the mean.

On that metric the CNN is the **worst** arm (88.9 ± 3.9) despite winning everything else,
and it is the single reason it does not ship. A gate that loses 15% of a scarce species'
calls to buy 4 points of card is a bad trade for this project whatever the headline says.

**So what:** this matters more at the target site than the test site — the point of a
rewilding project is often the *rare returning species*. It is also the metric most likely
to be missing from comparable published work, and it reversed our arm ranking.

---

## 9. Finding 7 — A blocker that was a measurement artefact

**Exec summary.** We first costed the CNN at 23.9 KB of working memory against an 8 KB
budget and called it the thing that stops it shipping. **That was wrong, and wrong in an
instructive way:** it measures what the network costs when each feature map is built
whole, which is what a *training framework* does and has nothing to do with what a device
must do. Streamed one column at a time as frames arrive, the same network — bit-identical
output, not one extra multiply — needs **7.1 KB**, inside budget.

The obvious alternative, tiling and recomputing overlaps, was measured first and is a
much worse deal (1.6× the compute). And the first streaming implementation was wrong
*quietly*: every interior column bit-exact, only the two edge columns off, because
padding is a **per-layer** boundary condition, not a network-level one. Only a
bit-equality test found it.

**So what:** we nearly closed out a viable option on a number that was measuring the
wrong thing. Memory estimates for tinyML are execution-schedule-dependent, and reporting
one without saying which schedule you assumed is close to meaningless.

---

## 10. Finding 8 — we have been optimising the wrong quantity

**Exec summary.** The objective has been *bytes written to the card*, with compute
treated as a pass/fail feasibility check. Neither is energy, and there was no energy
model: `deployment_life()` returned the same **48.1 battery-days whether the gate
recorded everything or nothing**, because its current draw was a constant that never
read the duty cycle. Writing the model down changed the conclusion.

**(a) The battery capacity is wrong by ~3.5×, and it inverts the premise.** The
inherited figure is 10,500 mAh at 4.5 V — 47 Wh — which requires each of three
*series* AA cells to hold 10,500 mAh. Series sums voltage, not capacity: the pack's
mAh is one cell's. Real 3×AA is ~2,400–3,200 mAh, so **11–14 days at 9.1 mA, not 48**.

The consequence is structural. A 128 GB card holds 23 ungated days. With 48
battery-days the card fills first and a card-saving gate *is* the product; with ~12
battery-days **the battery dies first and the card is never the constraint** — at
Highland occupancy the card lasts 230–460 days against a battery lasting ~37. The gate
is then buying analysis time, not deployment time. PLAN §14 listed this as an open
question; the arithmetic says it is the default answer.

**(b) At SNE's occupancy the gate barely pays, and that is a property of the test
rig, not the method.** The SD path is 23–37% of a properly-configured budget, but the
gate can only act on the part of it that scales with duty. On test recordings averaging
60% occupancy, gating from 100% to 91.5% duty buys **+1.6%** of battery life. At
Highland occupancy (~10% duty) the same gate buys **+39%**. The gate's value is almost
entirely a function of how much silence there is to skip, which is exactly why SNE
understates it and why §1's framing — test rig, not target — matters here more than
anywhere else in the project.

**(c) ⚠️ Correction to my own earlier read — file fragmentation does *not* flip the
arm ranking.** It flips it on **write energy alone**, which is where I first computed
it: `cnn-int8` writes 2.8× the files of `bands` at matched retention, and stops being
cheaper once a file close costs more than ~5 s of sequential-write energy (~314 kB of
card programming — entirely plausible, since an SDHC allocation unit is 4 MB). That
break-even is real and worth keeping.

But once the MCU is counted, **`mfcc` is the cheapest arm at every per-file cost and in
both clock configurations** — its EM0 advantage (0.32 mA of wake bursts against
`cnn-int8`'s 0.67 and `carihc`'s 1.89) is larger than any plausible fragmentation
penalty. So the fragmentation finding survives as a caution about `cnn-int8`, not as a
reordering, and it *strengthens* rather than complicates the existing recommendation:
`mfcc` was already the arm to ship on accuracy-per-cost, and it wins on energy too.

**(d) "Does the core sleep" is the wrong question — compute is a duty cycle.** The core
never runs continuously in a competent always-on design: DMA fills a buffer while the
core sits in EM1, an interrupt wakes it to process the block, and it sleeps again. An arm
is *running* and the core is *asleep* at the same time, time-sliced at millisecond
granularity. So **MMAC/s is literally the fraction of wall-clock time the core spends
awake** — 3.2% for `mfcc`, 18.7% for `carihc` — not an abstract feasibility ratio.

That reframes the budget. The arms cost 0.3–1.9 mA in wake bursts; the floor beneath
them — the EM1 clock tree plus the mic/preamp/ADC chain that must run continuously or
the pre-roll ring buffer does not exist — is 2.0–4.5 mA and is **identical for every
arm**. The floor dominates, and configuring the peripheral clock is worth more than
choosing the front-end.

**Where the current actually goes** (peripheral clock at 8 MHz, mA):

| arm | EM0 duty | EM0 burst | EM1 floor | analog | SD | **total** |
|---|---|---|---|---|---|---|
| `carihc` | 18.7% | 1.89 | 0.50 | 1.50 | 1.74 | **5.54** |
| `bands` | 16.0% | 1.62 | 0.50 | 1.50 | 1.70 | 5.24 |
| `cnn-int8` | 6.6% | 0.67 | 0.50 | 1.50 | 1.69 | 4.33 |
| **`mfcc`** | **3.2%** | **0.32** | 0.50 | 1.50 | 1.72 | **4.03** |

Two columns are identical down the table and together they are half the budget. No
model choice and no gate setting can touch either.

**Ranked levers, at Highland occupancy:**

| configuration | mean current | battery-days |
|---|---|---|
| core pinned in EM0 (a missing sleep loop, not a design) | 12.3 mA | 8.8 |
| DMA batching + EM1, clock tree at 48 MHz | 5.4 mA | 20.1 |
| + peripheral clock reduced to what the ADC needs | **2.95 mA** | **36.7** |
| the same, with `carihc` instead of `mfcc` | 4.4 mA | 24.4 |
| tiered wake-up: a cheap detector gates the expensive arm | bounded by ~1.5 mA | — |
| CARFAC decimation (PLAN §12), 2–4× on `carihc`'s burst only | bounded by ~1.4 mA | — |
| + drop the ADC when idle — **destroys pre-roll** | 1.1 mA | 98.3 |

The two middle rows are worth naming because they are the obvious clever ideas and both
are *bounded by the EM0 burst term*, which is at most 1.9 mA. No amount of cleverness in
scheduling the classifier can beat simply configuring the clock tree.

**So what.** Two corrections to my own first pass. The "does it sleep, 8.4 vs 18.6 days"
framing was comparing against a straw-man firmware; stock AudioMoth almost certainly
batches and sleeps already, so the honest levers are peripheral-clock configuration
(≈ +80%), then arm choice (≈ +50% between `mfcc` and `carihc`), then the gate. And the
gate looks *better* once sleep is modelled properly, not worse: at Highland occupancy it
buys **+39%** of deployment life, not the +10% I first reported, because the SD path is a
much larger share of a 4 mA budget than of a 12 mA one. Under that configuration
`mfcc` also becomes the first setup in any of these runs where the **card binds before
the battery** (25.3 d against 26.9 d) — i.e. the only one where the project's original
premise, that saving card extends the deployment, is actually true.

The one genuinely huge lever — powering down the ADC between triggers, worth ~2.7× —
is the one the project has already ruled out by choosing pre-roll, and §3 measured what
that choice is worth in recall (whole-box capture 91.7% → 97.0%). **That is the real
energy/recall trade in this system, and it is a design decision rather than an
optimisation.** The model lives in `birdgate/energy.py`; `scripts/energy_report.py`
prints the bench protocol for the three constants that would settle it.

*Detail: `birdgate/energy.py` holds the parameter table and the break-even arithmetic; `scripts/energy_report.py` prints the M1–M3 bench protocol.*

---

## 11. Proposal — a density-aware recording policy

> Not a result. A design, with the tuning plan and the failure modes it has to be
> tested against, so the next stage is specified rather than improvised.

**Exec summary.** The current policy is two fixed constants: hold the gate open at
least `min_on_s` (20 s), then close once the score has been quiet for `hangover_s`
(1 s). Both are blind to context — an isolated tick in a silent hour is treated exactly
like a gap inside a dawn chorus. The proposal is to let the *close* condition depend on
recent birdsong **density**:

```
rho <- (1-beta)*rho + beta*[z_t > theta_off]     # one float, updated each recheck
hangover_t = h_min + (h_max - h_min) * rho
close when quiet_for > hangover_t
```

Opening stays instantaneous — one positive opens the gate, as now, so onset recall is
unchanged. Sustained song drives ρ→1 and the gate bridges long gaps into a single long
file; an isolated trigger leaves ρ near 0 and the gate shuts almost at once. **This
replaces `min_on_s` outright.**

**Why it is structurally better, not merely tuned differently.** `min_on_s` slides
monotonically along a duty-versus-files trade: raising it cuts file count but spends
duty on windows already known to be empty. The density gate *decouples* the two — it
adds duty only where song is dense (where that duty is mostly true positive anyway) and
removes it in sparse stretches, which is exactly where `min_on_s` was pure waste. So it
can reach a point that **dominates the current policy on both axes at once**, which no
setting of `min_on_s` can. Both axes are the two terms of the energy equation in §10.

**How to tune it:**
- Three parameters (β, `h_min`, `h_max`) plus a hard cap on continuous on-time.
- **G1's gap statistics are the physical prior, not a blind sweep.** Within-recording
  silent gaps have median 1.9 s and 75th percentile 4.1 s, so `h_min` belongs near the
  median gap (bridge inter-syllable) and `h_max` at the inter-bout scale.
- Objective: minimise `energy.energy_equivalent_seconds(duty, files, c)` subject to
  retention ≥98% and species floor ≥95%, selected on **val** only. Since `c` is
  unmeasured, sweep it and report the optimum *as a function of c* — a robustness
  surface, not a point estimate. If the optimum is stable across `c`, the bench
  measurement drops off the critical path entirely.
- Evaluate the four sparse recordings separately. The whole point is behaviour that
  varies with occupancy, and SNE's 49% mean hides it — at 76% occupancy "stay on while
  birds sing" is indistinguishable from "stay on".

**Failure modes to design against, up front:**
1. **Lock-on.** A gate that stays open because it has been open is positive feedback —
   structurally the same trap as the background tracker's α risk in §6. Hence the hard
   cap, and an explicit check that it does not degenerate to always-on on dawn chorus.
2. **ρ must be driven by the score, never by the gate's own output**, or the feedback
   loop is closed rather than merely risky.
3. **Two adaptive loops now interact.** The background tracker already updates 10×
   slower while the gate is open; longer opens mean a staler background for longer.
   They have to be validated jointly, not one at a time.
4. Anything below `recheck_period_s` is inert — the gate can only close at a recheck
   instant.

**Cost on device:** one float and one multiply-add per recheck — strictly cheaper than
`min_on_s`, which needs a timer.

---

## 12. Repeatability — and what today's audit found

**Exec summary.** Traceability is a first-class requirement here, not hygiene: these
recordings become primary ecological data, and audio the gate discarded cannot be
re-examined. Every run writes a manifest with the resolved config, a config hash, the
exact recording list per split, digests of the label files, and package versions.

**Verified today, from scratch, with the audio drive unplugged:**

| checked | result |
|---|---|
| All dataset-level statistics | ✅ exact — 20,147 boxes, 33 recordings, 16.26 h birdsong, 49.3% occupancy, 1.4–86.1% per-recording range |
| Final linear-arm table (§9 of PLAN) | ✅ reproduced digit-for-digit from the feature cache |
| Four-arm headline table (§9b) | ✅ reproduced digit-for-digit |
| Matched-retention frontier, all 5 arms | ✅ all 15 rows exact |
| CNN results | ✅ replayed from cached logits; row alignment asserted, matches to 3 dp |
| Memory and compute figures | ✅ 7,264 B streamed / 23,872 B whole-tensor; 5,865 deployable params |
| Test suite | ✅ 27 pass |

**Three defects found, one of which matters:**

1. ⚠️ **Manifests did not record the gate or threshold policy actually used** (since
   fixed: `GateConfig` carries the adaptive-threshold parameters and every script
   writes the config it ran with). The
   provenance call writes a *fresh default* config rather than the one the run executed.
   The archived comparison manifest claims a minimum-on time of 0 s; re-running proved its
   table was produced at 10 s. The adaptive-threshold parameters — including the anchor
   quantile we call the most consequential number in the system — appear in **no manifest
   at all**. The numbers are all correct; the *record of what produced them* is not. Given
   §11's promise that a card recovered from a device resolves back to the exact gate that
   let the audio through, this is the one thing here that would genuinely bite.
2. There was no Python/C parity *test* — the check was run by hand through the C
   harness. The empty `tests/test_gate_parity.py` placeholder has been removed;
   ROADMAP F6 turns the harness into a pytest.
3. One archived manifest holds superseded pre-correction numbers with nothing marking it
   stale, and roughly twenty result tables in `docs/lab-notebook.md` are backed by only four manifests.

**So what:** the science reproduced perfectly; the bookkeeping around it did not. Worth
saying plainly, because the project's own stated standard is the one it fell short of.

---

## 13. What is not done

- ~~**Calibration**~~ ✅ done — the operating-point table is in PLAN §13 G4 and
  `runs/g4-calibrate-mfcc`. The floor had to be guarded on bootstrap CI width first;
  before that a species with 41 s of song was setting it, and `frugal` bought 0.76%
  of card instead of 20.88%.
- ~~**The CNN's species floor**~~ ✅ resolved — the blocking `88.9` was an artefact of the
  same count-only guard. Guarded properly, `cnn-int8` is **95.54%** with the best
  worst-recording figure of any arm. The mfcc-vs-CNN choice is now a real trade, and
  `mfcc` is being ported first on cost and C-simplicity grounds, not because the CNN lost.
- **Firmware** ◐ in progress — build chain proven on hardware; the whole chain (front-end,
  scorer, adaptive threshold, state machine, device glue) is ported, host-verified with
  **identical gate output** to Python, and linked into a device build. Next milestone is
  flashing it and watching the green LED follow the gate — see `ROADMAP.md` §1.
- **Fixed-point** — deliberately deferred until something works on hardware in float, so a
  port bug cannot be confused with quantisation drift. Budget: ≤0.2% retention.
- **Per-species data coverage** — `species_floor` is a minimum, so it is only as good as
  the thinnest species in the split. Three of 22 val species cannot be measured to within
  ten points; the worst has 41 s of song. `metrics.admissible` now excludes them and says
  so, but the real fix is collection and augmentation — PLAN §13 G7b.
- **Highland data** — the actual goal. The front-end, state machine, metrics and export
  path carry over unchanged; only the training set and threshold get re-fit.
- **The energy bench (§10)** — three measurements (SD per-file cost, the EM0/EM1
  currents and peripheral clock config, and the analog floor) turn `energy.py`'s
  estimates into real deployment-days. Protocol is
  written and printed by `scripts/energy_report.py`. **This is now the highest-value
  unblocked task**, because it decides whether the arm comparison means anything
  energetically.
- **The density-aware gate (§11)** — designed, not built. Should be tuned against the
  energy objective rather than against saving, and it makes `min_on_s` redundant.

---

## 14. Why any of this is interesting

**Exec summary — the four claims worth making tomorrow.**

1. **A neuromorphic cochlea front-end was shown, under controls, not to earn its keep**
   — and the cheaper, simpler alternative is also the lower-risk one to deploy.
   The controls are the reason we know that rather than suspect it.
2. **A failure mode that is invisible in aggregate metrics and unrecoverable in the
   field** — threshold non-transfer across recordings — was found and fixed with a
   few-hundred-byte trick from radar. This generalises to essentially any always-on
   acoustic sensor.
3. **Testing the *argument* for a tiny model rather than only the model** gave a more
   useful answer than a leaderboard: right conclusion, wrong reason, and now we know
   the actual price of the decision (~4 points of card).
4. **The binding metric is the one that protects rare species**, and it reverses the
   ranking that every other metric gives. For a rewilding project — where the rare
   returning species *is* the point — a gate optimised on aggregate retention would have
   been actively harmful.

5. **We audited our own objective function and found it was measuring the wrong thing** —
   twice, in the same genre. The CNN's memory blocker was an artefact of execution order
   (§9), and the whole project's optimisation target turned out to be a ~1% term in the
   energy budget sitting on a battery constant that was wrong by 3.5× (§10). Both were
   found by asking "what does this number actually measure?" rather than by getting a
   better number. That habit is the most portable thing here.

**And the practical one:** the honest headline on this dataset is that we capture roughly
a quarter to a third of the achievable saving. That is a weak number and we report it as
one. But the test recordings average 60% occupancy, which is close to the worst possible
setting for a gate; a real 24-hour Highland deployment sits at a few percent, where the
same model has vastly more room. **The two numbers to be sceptical of are not the saving.** They are
whether a gate trained in California transfers to Scotland at all, and what the device's
idle floor actually is once the peripheral clock is configured properly. Both
measurements are cheap, both are unblocked, and nobody has either — and between them
they matter more than every result above.

---

## Appendix A — the three linear arms, in detail

All three share a skeleton and differ only in step 1:

1. **Decompose** 1 second of audio into ~40 frequency channels.
2. **Reduce** each channel's time course to one number per window.
3. **Classify** the resulting ~40-vector with a linear SVM.

### The classifier, shared by all three

A linear support vector machine. Given feature vector **x**, it computes one number

    f(x) = w·x + b = Σₚ wₚ xₚ + b

and the gate opens when `f(x) > θ`. Training minimises the regularised hinge loss

    min_w  ½‖w‖² + C Σᵢ max(0, 1 − yᵢ(w·xᵢ + b))

with `C = 1`. The hinge is zero once a sample is correctly classified *with margin*,
so only points near the boundary contribute — the "support vectors". The objective is
**convex**, which is why §12 can report that seed variance is exactly zero: there is one
optimum and the solver finds it every time.

Training standardises each feature first (`StandardScaler`: subtract the mean, divide
by the standard deviation). That scaler is then **folded into the weights**:

    w_deployed = w / σ        b_deployed = b − Σₚ wₚ μₚ / σₚ

so the device stores only `P` weights and one bias and does a bare dot product — no
normalisation at run time, no statistics to carry. 42 floats + 1 = **172 bytes**.

### Arm 1 — `carihc`: a cascade of asymmetric resonators

**The physics being imitated.** The cochlea is a tapered hydromechanical transmission
line. Sound enters at the *base* and a travelling wave propagates toward the *apex*.
Stiffness falls along the way, so each position resonates at its own characteristic
frequency — high at the base, low at the apex. The consequence that matters: **the wave
must pass through every higher-frequency region before it reaches a lower-frequency
one.** Energy above a place's characteristic frequency has already been absorbed
upstream. That is why real cochlear tuning curves are *asymmetric* — steep on the high
side, gradual on the low side — and it is the property the cascade reproduces
structurally rather than by fitting.

**The filter.** 40 biquad stages in series. Stage *p* has a conjugate pole pair at
radius `rₚ` and angle `θₚ = 2π fₚ / f_s` (radians per sample), so with
`aₚ = cos θₚ`, `cₚ = sin θₚ`:

    Hₚ(z) = gₚ · (1 + (hₚ rₚ cₚ − 2 rₚ aₚ) z⁻¹ + rₚ² z⁻²)
                 ─────────────────────────────────────────
                 (1 −              2 rₚ aₚ  z⁻¹ + rₚ² z⁻²)

- **The poles** set the resonance. `rₚ` controls sharpness: bandwidth ≈ `(1−rₚ)·f_s/π`,
  so `r → 1` means a sharp, long-ringing filter. Measured range here is
  `r ∈ [0.827, 0.996]` — the low channels sit very close to the unit circle, which is
  exactly why §4's fixed-point note insists the firmware use the coupled form.
- **The zeros** are the "asymmetric" in CAR. They sit at `ρ = √2` times the pole
  frequency — *above* it — via `hₚ = cₚ(ρ² − 1)`. A zero above the resonance puts a
  notch on the high side, steepening that flank and leaving the low side gradual.
- **`gₚ` normalises DC gain to 1** per stage, so 40 stages in series do not accumulate
  gain: `g = (1 − 2ra + r²)/(1 − 2ra + hrc + r²)`.

**Channel spacing** is uniform on the ERB-number scale
`E(f) = 21.4·log₁₀(1 + 0.00437 f)` — equal numbers of auditory filters apart, not equal
Hz — from 12 kHz down to 150 Hz. So low channels are packed tightly and high ones
spread out, matching how the ear allocates resolution.

**The cascade is the point.** Channel *p*'s output is
`yₚ = Hₚ(Hₚ₋₁(⋯H₀(x)))`, so it has *effective order 2(p+1)* while costing only one
biquad. One filter's worth of arithmetic buys a filter of order *p*. That is the
efficiency claim, and it is why the measured best frequency sits ~7.5% above the design
pole (the stages above have already attenuated, which drags the peak up).

**Inner hair cell.** Half-wave rectification, `max(0, yₚ[n])` — nothing more. Real hair
cells depolarise when the stereocilia bend one way and not the other, so they respond to
one direction of displacement. The model keeps just that.

**The template.** Channel *p*'s feature is its rectified output summed over the window:

    Φₚ = Σ_{n ∈ window} max(0, yₚ[n])

This *is* the classifier's input — the template-SVM formulation's neat trick is that
feature extraction and the SVM's kernel are the same computation.

**Normalisation, and why it needed changing (§1).** A species-ID front-end L1-normalises
across channels, `Φ/ΣΦ`, giving pure spectral *shape*. Correct for species ID — a bird at 5 m
and at 15 m should classify alike — and actively wrong for "is anything there", because
silence and birdsong both become a unit-sum vector. So two features are appended to put
the discarded information back:

- `log(ΣΦ)` — the level the normaliser threw away.
- `log(flatness)`, where flatness is the geometric mean over the arithmetic mean across
  channels, `exp(mean log Φ) / mean Φ`. Near 1 for noise-like spectra, near 0 for tonal
  ones. **Wind, rain and running water are loud but flat; birdsong is structured** — this
  feature is aimed squarely at the Highland failure modes rather than at Sierra Nevada.

**40 shape + 2 = 42 features.**

### Arm 2 — `bands`: the same thing without the cascade

Identical in every respect except step 1. Instead of 40 stages in series, 40
**independent** 2-pole Butterworth bandpass filters in parallel, each centred on the
same ERB-spaced `fₚ` and one ERB wide — bandwidth `(fₚ + 165.3)/9.2645`, the auditory
filter bandwidth at that frequency, so the bank tiles the spectrum the same way the
cascade does.

The audio goes into each filter *directly* rather than through its predecessors. Then
the same half-wave rectification, the same window summing, the same L1 normaliser, the
same two appended features. **42 features, 172 bytes.**

**What differs, and only this:**

| | `carihc` | `bands` |
|---|---|---|
| topology | series | parallel |
| effective order of channel *p* | 2(p+1) | 2 |
| response shape | asymmetric (zeros above poles) | roughly symmetric |
| error propagation | compounds down 40 stages | independent per channel |

That is what makes it the control worth having: it changes exactly one thing, so §4's
result is attributable to the cascade-and-zeros and nothing else.

### Arm 3 — `mfcc`: the FFT route

A completely different decomposition — this is the standard speech-recognition front-end,
and it is the one usually rejected on cost grounds for this class of device.

**Step 1 — frame and transform.** 1024-point FFT every 320 samples. At 32 kHz that hop
is exactly 10 ms, so **100 frames/second** and a 1 s window is exactly 100 frames — no
fractional-frame alignment anywhere. Take the power spectrum `|X(k)|²`.

**Step 2 — mel filterbank.** Project the 513 power bins onto triangular filters spaced
uniformly on the *mel* scale,

    m(f) = 2595 · log₁₀(1 + f/700)

between 150 Hz and 14 kHz. Like the ERB scale this is roughly logarithmic — it spends
resolution where hearing does. Output: a vector of band energies `S_j`.

**Step 3 — logarithm.** `L_j = log S_j`. This is the step that makes the whole thing
work, for a reason worth stating precisely: a microphone, a room, or a transmission
channel *convolves* the signal, which **multiplies** the spectrum, which **adds** in the
log spectrum. So a gain or channel change becomes an additive offset rather than a
multiplicative distortion — a far easier thing for a linear model to be robust to.

**Step 4 — discrete cosine transform.** Take the DCT-II of the log-mel vector and keep
the first 20 coefficients:

    cᵢ = Σⱼ Lⱼ · cos[ π i (j + ½) / J ]

Adjacent log-mel bands are strongly correlated — a harmonic stack lights up several at
once — and the DCT approximately **decorrelates** them (it is close to the optimal
Karhunen–Loève transform for this class of signal). So a handful of coefficients captures
most of the variance. The coefficients are interpretable as spectral *ripple rates*: `c₀`
is overall level, `c₁` spectral tilt, and higher indices describe progressively finer
structure across frequency. This is why "cepstrum" is an anagram of "spectrum" — it is a
spectrum *of* a spectrum.

**Step 5 — pool over the window.** For each of the 20 coefficients take the **mean and
the standard deviation** across the window's 100 frames → **40 features**.

The standard deviation is doing real work here and is the reason this arm is not simply a
worse filterbank: the mean captures average spectral shape, while the std captures
**modulation** — how much the spectrum moves *within* the second. Birdsong is strongly
modulated; steady wind and running water are not. It is the same discrimination the
`log_flatness` feature provides for the other arms, but measured across *time* rather
than across *frequency*.

**No normaliser.** MFCCs are signed (they are a cosine transform of logs), so L1
normalisation and geometric means are meaningless. The arm gets its raw coefficients,
standardised in the model pipeline like everything else. **40 features, 164 bytes.**

### Why the cost argument inverts — the one bit of physics that decides the ranking

The usual case against MFCC on a microcontroller is that it needs an FFT and log-domain
operations. That is a claim about a *device*, and it is **sample-rate dependent**:

- **Filterbank cost scales with the sample rate.** IIR filters are stateful — every
  single sample must pass through every filter, or the state is wrong. 40 channels ×
  32,000 samples/s × ~6 MAC = **7.7–9.0 MMAC/s**.
- **FFT cost scales with the frame rate.** 100 frames/s × (½·1024·log₂1024 FFT MACs +
  512×20 mel projection) = **1.54 MMAC/s**, of which the mel projection is two-thirds —
  the FFT itself is the cheap part.

At 8 kHz the filterbank costs a quarter as much while the FFT is unchanged,
and the cascade wins comfortably. **At our 32 kHz the filterbank quadruples and the FFT
does not move, so the ranking inverts and MFCC is 5–6× cheaper.** The argument is
correct at 8 kHz and does not survive the sample-rate decision
§1 forced on us for the sake of high, thin Highland calls.

(Caveat kept from PLAN §8: MAC counts undercount `log()` and the DCT, which are slower
per operation on an M4F than a multiply-accumulate, so treat 5–6× as an upper bound.)
