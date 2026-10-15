"""G7: where the AudioMoth's scarce resources actually go, by function.

**Why this module exists.** The project has carried resource *targets* since docs/lab-notebook.md
§4 ("< 8 KB SRAM", "< 100 KB flash") and a compute *model* since `frontends.cost_model`
("1.5 MMAC/s, 3.2% of a 48 MHz M4F"). Both were budgets in the abstract: a single
number per arm, with no statement of which part of the chain spent it. That is fine
until something does not fit, at which point the only actionable question is *what to
cut*, and a single number cannot answer it.

Three of the four resources here are read from the **linker map** of a real device
build, attributed symbol by symbol. That is not a model: `.bss.ring 0x4b0 adapt.o`
is 1,200 bytes of adaptive-threshold history that the part physically contains. The
fourth, compute, **is** a model, and a demonstrably poor one — see `CYCLES`.

The four resources, and which are measured:

| resource | size | source |
|---|---|---|
| internal SRAM | 32 KB | linker map, per symbol — MEASURED |
| flash | 240 KB | linker map, per symbol — MEASURED |
| external SRAM (the sample ring) | 256 KB | arithmetic from stock's `NUMBER_OF_BUFFERS` |
| core time per second of audio | 48 Mcycle | MODELLED, and known wrong by ~4x |

⚠️ **The map is the live build, not the tables in ROADMAP.md.** `AM_Software/` is
disposable build output, so these numbers move whenever `firmware/build.sh` runs and
will disagree with any figure written down by hand. When they disagree, the map is
right and the prose is stale — that is the point of reading it here.

⚠️ **The compute panel is the weak one, and it is weak by measurement, not by
suspicion.** `BG_SELFTEST` on the device timed the CNN arm's front-end at 450 ms per
window against 110 ms counted by hand: wrong by 4.1x, on code whose inner loops had
been read (ROADMAP.md §C). Operation *counts* here are exact — they come from the
exported constants — and cycle counts are not. Read the stage shares, not the totals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AM = ROOT / "AM_Software"
FIRMWARE = ROOT / "firmware"


# --------------------------------------------------------------------------- #
# The part                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Silicon:
    """EFM32WG380F256 on an AudioMoth 1.x, as the build actually sees it.

    `flash_bytes` is the **linker** FLASH region (0x4000..0x40000), not the part's
    256 KB: the first 16 KB holds the bootloader and is not ours to spend. Likewise
    `heap_bytes` is what `audiomoth.ld` reserves unconditionally whether or not
    anything mallocs, so it is spent SRAM from our point of view.
    """
    core_hz: int = 48_000_000
    flash_bytes: int = 0x3C000          # 245,760 — linker FLASH region
    flash_offset: int = 0x4000          # bootloader, not ours
    sram_bytes: int = 0x8000            # 32,768 — linker RAM region
    heap_bytes: int = 0xC00             # 3,072 — .heap in audiomoth.ld
    ext_sram_bytes: int = 256 * 1024    # AM_EXTERNAL_SRAM_SIZE_IN_BYTES
    n_ring_slots: int = 8               # NUMBER_OF_BUFFERS, stock main.c
    sample_rate: int = 32_000
    bytes_per_sample: int = 2


SI = Silicon()


def ring_model(si: Silicon = SI) -> dict:
    """The external SRAM as stock firmware divides it, and what that caps.

    Stock cuts the 256 KB into `NUMBER_OF_BUFFERS` equal slots and writes them to
    the card one slot at a time, so the ring is simultaneously the write queue and
    the only past audio that exists anywhere. **Pre-roll can only reach back into
    slots the DMA has not yet overwritten**, which is what makes 4.096 s a hard
    ceiling at 32 kHz rather than a tuning choice — and why the app's pre-roll
    slider stops where it does.
    """
    slot_b = si.ext_sram_bytes // si.n_ring_slots
    slot_n = slot_b // si.bytes_per_sample
    slot_s = slot_n / si.sample_rate
    return {
        "slot_bytes": slot_b,
        "slot_samples": slot_n,
        "slot_s": slot_s,
        "total_s": slot_s * si.n_ring_slots,
        # One slot is being filled by the DMA and cannot be read, so the usable
        # history is one slot short of the ring. The firmware backfills in whole
        # 1 s windows, so what pre-roll can actually claim is the floor of that.
        "readable_s": slot_s * (si.n_ring_slots - 1),
        "max_pre_roll_s": float(np.floor(slot_s * (si.n_ring_slots - 1))),
        "bytes_per_s": si.sample_rate * si.bytes_per_sample,
    }


# --------------------------------------------------------------------------- #
# Linker map                                                                   #
# --------------------------------------------------------------------------- #
# Output sections and which resource each one spends. `.data` is the awkward one:
# it is initialised RAM, so its contents sit in flash AND its footprint sits in
# SRAM. It is counted in both, which is correct and is why the two panels do not
# sum to one number.
_RAM_SECTIONS = {".data", ".bss"}
_FLASH_SECTIONS = {".text", ".data", ".ARM.exidx", ".ARM.extab"}

_OUT_SEC = re.compile(r"^(\.[A-Za-z0-9_.]+)")
_NAME_ONLY = re.compile(r"^ (\S+)\s*$")
_ROW = re.compile(r"^\s+(?:(\S+)\s+)?0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S+)\s*$")
_FILL = re.compile(r"^\s+\*fill\*\s+0x[0-9a-f]+\s+0x([0-9a-f]+)\s*$")
_SEC_HEADER = re.compile(r"^(\.[A-Za-z0-9_.]+)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)")


def arm_builds() -> dict[str, Path]:
    """Arms with a linker map on disk, i.e. arms that have actually been linked.

    `carihc` and `bands` are absent on purpose: their firmware was deleted as dead
    scaffolding (ROADMAP.md §2), so there is no honest footprint to show for them —
    only `frontends.cost_model`'s biquad arithmetic, which this module does not
    dress up as a measurement.
    """
    out = {}
    for d in sorted(AM.glob("AudioMoth-*/build/audiomoth.map")):
        arm = d.parent.parent.name.removeprefix("AudioMoth-")
        # A map is only admitted if `firmware/arms/<arm>/` is still there. Build
        # trees are never cleaned up, so `AudioMoth-mfcc/` outlived the sources it
        # was built from by a week — and a footprint attributed to code that no
        # longer exists is worse than no footprint, because nothing about it looks
        # stale. `stock` is the exception: it has no arm directory by definition.
        if arm == "stock" or (FIRMWARE / "arms" / arm / "model_params.h").exists():
            out[arm] = d
    if not out:
        # No local builds (a fresh clone): the snapshots `scripts.build_demo` took.
        out = {p.stem: p for p in sorted((ROOT / "demo" / "maps").glob("*.map"))}
    return out


def parse_map(path: Path) -> pd.DataFrame:
    """Every input section the linker placed, as (out_section, symbol, obj, bytes).

    Two shapes have to be handled: `name addr size obj` on one line, and — when the
    symbol name is long enough to crowd the column — the name alone followed by
    `addr size obj` on the next. Missing the second form silently loses whole
    symbols, which is how `.bss.freeze_after` and friends would vanish.

    `*fill*` rows are kept. They are alignment padding, they are real bytes, and
    dropping them is exactly why a naive sum comes in ~180 B under the section
    header — see `section_totals`.
    """
    rows, sec, pend = [], None, None
    for line in path.read_text().splitlines():
        if not line.startswith(" "):
            m = _OUT_SEC.match(line)
            if m:
                sec, pend = m.group(1), None
            continue
        if sec is None:
            continue
        m = _FILL.match(line)
        if m:
            rows.append((sec, "*fill*", int(m.group(1), 16), "*fill*"))
            continue
        m = _NAME_ONLY.match(line)
        if m and not m.group(1).startswith("0x"):
            pend = m.group(1)
            continue
        m = _ROW.match(line)
        if m:
            name, pend = (m.group(1) or pend), None
            if name is None or name.startswith("0x"):
                continue
            rows.append((sec, name, int(m.group(3), 16), m.group(4)))
    df = pd.DataFrame(rows, columns=["section", "symbol", "bytes", "path"])
    df["obj"] = df["path"].str.rsplit("/", n=1).str[-1]
    return df[df["bytes"] > 0].reset_index(drop=True)


def section_totals(path: Path) -> dict[str, int]:
    """Output-section sizes from the section headers — the authoritative figures.

    Used to *check* the per-symbol attribution rather than to replace it. If the
    symbols do not sum to the header the attribution is lossy, and every share
    computed from it is quietly wrong; `tests/test_budget.py` asserts they agree.
    """
    tot = {}
    for line in path.read_text().splitlines():
        m = _SEC_HEADER.match(line)
        if m and m.group(1) not in tot:
            tot[m.group(1)] = int(m.group(3), 16)
    return tot


# --------------------------------------------------------------------------- #
# Attribution                                                                  #
# --------------------------------------------------------------------------- #
# (object, symbol substring or None) -> (owner, stage). First match wins, so the
# symbol-specific rules for an object must precede its catch-all.
#
# The stage names are the pipeline as `birdgate.c` runs it, in order, because the
# whole point of the panel is to be read against the block diagram. Anything that
# is not ours is grouped by what it is FOR, not by which emlib file it came from:
# "does the SD stack or the USB stack cost more?" is a question worth answering,
# "how big is em_ldma.o?" is not.
_RULES: list[tuple[str, str | None, str, str]] = [
    # --- BirdGate: the chain, in execution order ---------------------------- #
    ("bg_stream.o", None,        "BirdGate", "1. Frame the sample ring"),
    ("fft.o",       "BG_HANN",   "BirdGate", "2. FFT (window + twiddle tables)"),
    ("fft.o",       "BG_TW",     "BirdGate", "2. FFT (window + twiddle tables)"),
    ("fft.o",       "bg_pow",    "BirdGate", "3. Power spectrum"),
    ("fft.o",       None,        "BirdGate", "2. FFT (window + twiddle tables)"),
    ("mfcc.o",      "BG_MEL",    "BirdGate", "4. Mel filterbank"),
    ("mfcc.o",      "BG_DCT",    "BirdGate", "5. DCT / cepstrum"),
    ("mfcc.o",      "frame",     "BirdGate", "1. Frame the sample ring"),
    ("mfcc.o",      "mean",      "BirdGate", "6. Window statistics (Welford)"),
    ("mfcc.o",      "m2",        "BirdGate", "6. Window statistics (Welford)"),
    ("mfcc.o",      "n_frames",  "BirdGate", "6. Window statistics (Welford)"),
    ("mfcc.o",      None,        "BirdGate", "4. Mel filterbank"),
    ("logmel.o",    "BG_MEL",    "BirdGate", "4. Mel filterbank"),
    ("logmel.o",    "frame",     "BirdGate", "1. Frame the sample ring"),
    ("logmel.o",    None,        "BirdGate", "4. Mel filterbank"),
    ("cnn.o",       "BG_CONV",   "BirdGate", "7. Model weights"),
    ("cnn.o",       "BG_LIN",    "BirdGate", "7. Model weights"),
    ("cnn.o",       "BG_ACT",    "BirdGate", "7. Model weights"),
    ("cnn.o",       "ring",      "BirdGate", "8. Model working memory"),
    ("cnn.o",       "dq",        "BirdGate", "8. Model working memory"),
    ("cnn.o",       "acc",       "BirdGate", "8. Model working memory"),
    ("cnn.o",       "out",       "BirdGate", "8. Model working memory"),
    ("cnn.o",       "col",       "BirdGate", "8. Model working memory"),
    ("cnn.o",       "stages",    "BirdGate", "8. Model working memory"),
    ("cnn.o",       None,        "BirdGate", "9. Model arithmetic"),
    ("scorer.o",    "BG_W",      "BirdGate", "7. Model weights"),
    ("scorer.o",    None,        "BirdGate", "9. Model arithmetic"),
    ("adapt.o",     None,        "BirdGate", "10. Adaptive threshold"),
    ("gate.o",      None,        "BirdGate", "11. Gate state machine"),
    ("birdgate.o",  "gate",      "BirdGate", "11. Gate state machine"),
    ("birdgate.o",  None,        "BirdGate", "12. Chain glue"),
    # --- stock, grouped by purpose ------------------------------------------ #
    ("ff.o",            None, "Stock", "SD card + FAT filesystem"),
    ("ffunicode.o",     None, "Stock", "SD card + FAT filesystem"),
    ("diskio.o",        None, "Stock", "SD card + FAT filesystem"),
    ("microsd.o",       None, "Stock", "SD card + FAT filesystem"),
    ("audioconfig.o",   None, "Stock", "USB + configuration"),
    ("digitalfilter.o", None, "Stock", "Audio path (ADC, DMA, filters)"),
    ("biquad.o",        None, "Stock", "Audio path (ADC, DMA, filters)"),
    ("butterworth.o",   None, "Stock", "Audio path (ADC, DMA, filters)"),
    ("dmactrl.o",       None, "Stock", "Audio path (ADC, DMA, filters)"),
    ("em_adc.o",        None, "Stock", "Audio path (ADC, DMA, filters)"),
    ("em_dma.o",        None, "Stock", "Audio path (ADC, DMA, filters)"),
    ("em_ebi.o",        None, "Stock", "Audio path (ADC, DMA, filters)"),
    ("gps.o",           None, "Stock", "GPS + sunrise"),
    ("gpsinterface.o",  None, "Stock", "GPS + sunrise"),
    ("gpsutilities.o",  None, "Stock", "GPS + sunrise"),
    ("nmeaparser.o",    None, "Stock", "GPS + sunrise"),
    ("sunrise.o",       None, "Stock", "GPS + sunrise"),
    ("main.o",          None, "Stock", "Recording loop + scheduler"),
    ("audiomoth.o",     None, "Stock", "Board support"),
]


def _classify(obj: str, symbol: str) -> tuple[str, str]:
    for o, sub, owner, stage in _RULES:
        if obj == o and (sub is None or sub in symbol):
            return owner, stage
    if obj.startswith("em_usb"):
        return "Stock", "USB + configuration"
    if obj.startswith("em_") or obj in ("startup_efm32wg.o", "system_efm32wg.o"):
        return "Stock", "MCU drivers (emlib)"
    if symbol == "*fill*":
        return "Padding", "Alignment padding"
    return "Toolchain", "Toolchain + libc"


def _role(section: str, symbol: str) -> str:
    """Code, constant table, or RAM state — the axis that says whether a line is
    cuttable by writing less code or only by holding less data."""
    if section == ".bss":
        return "RAM state"
    if section == ".data":
        return "RAM state (initialised)"
    return "table (const)" if symbol.startswith(".rodata") else "code"


def footprint(arm: str) -> pd.DataFrame:
    """Per-symbol flash and SRAM for one linked arm, attributed and labelled.

    Returns one row per (resource, owner, stage, role) with `bytes`, plus the
    free-space row for each resource, so the two treemaps tile their resource
    exactly. `.data` appears under both resources on purpose — see `_RAM_SECTIONS`.
    """
    maps = arm_builds()
    if arm not in maps:
        raise KeyError(f"no linker map for {arm!r}; built arms are {sorted(maps)}")
    df = parse_map(maps[arm])
    df[["owner", "stage"]] = [_classify(o, s) for o, s in zip(df["obj"], df["symbol"])]
    df["role"] = [_role(sec, sym) for sec, sym in zip(df["section"], df["symbol"])]

    totals = section_totals(maps[arm])
    out = []
    for resource, sections, cap in (("flash", _FLASH_SECTIONS, SI.flash_bytes),
                                    ("sram", _RAM_SECTIONS, SI.sram_bytes)):
        g = df[df["section"].isin(sections)]
        agg = (g.groupby(["owner", "stage", "role"], as_index=False)["bytes"].sum()
                .assign(resource=resource))
        out.append(agg)
        # The section HEADER is authoritative; the symbol rows are what can be
        # attributed. The difference — tens of bytes, from sections the linker
        # script places without naming an object (`.vectors`, `.glue_7`, `.iplt`)
        # — is carried as its own row rather than quietly inflating free space.
        used = sum(totals.get(s, 0) for s in sections)
        residual = used - int(g["bytes"].sum())
        if residual > 0:
            out.append(pd.DataFrame([{
                "owner": "Toolchain", "stage": "Unattributed (linker script)",
                "role": "code", "bytes": residual, "resource": resource}]))
        if resource == "sram":
            out.append(pd.DataFrame([{
                "owner": "Stock", "stage": "Reserved heap", "role": "RAM state",
                "bytes": SI.heap_bytes, "resource": "sram"}]))
            used += SI.heap_bytes
        out.append(pd.DataFrame([{
            "owner": "Free", "resource": resource, "role": "free",
            # The two free rows mean different things, and the SRAM one is the
            # number this project has actually come close to spending.
            "stage": "Left for the stack" if resource == "sram" else "Unused flash",
            "bytes": cap - used}]))
    res = pd.concat(out, ignore_index=True)
    return res[["resource", "owner", "stage", "role", "bytes"]]


def hook_cost(arm: str) -> int | None:
    """How much flash our hook into stock's recording loop costs, by differencing.

    `app/main.c` is stock's `main.c` with the chain wired in, so it links as one
    object and cannot be attributed symbolically. The difference against the stock
    build's `main.o` is the hook. ⚠️ An ESTIMATE: it also absorbs any compiler
    inlining decision that changed between the two builds.
    """
    maps = arm_builds()
    if arm not in maps or "stock" not in maps:
        return None
    def main_o(a):
        d = parse_map(maps[a])
        return int(d[(d["obj"] == "main.o") & d["section"].isin(_FLASH_SECTIONS)]["bytes"].sum())
    return main_o(arm) - main_o("stock")


# --------------------------------------------------------------------------- #
# Compute                                                                      #
# --------------------------------------------------------------------------- #
# ⚠️ EVERY NUMBER IN THIS DICT IS AN ASSUMPTION, and the one measurement this
# project has says they are collectively optimistic by 4.1x on the front-end and
# 2.2x on the model (ROADMAP.md §C, BG_SELFTEST on the device).
#
# They are datasheet-class reckoning for a Cortex-M4F running code from flash with
# no cache: a VFP multiply-accumulate is 1-3 cycles of arithmetic, and everything
# around it — the two loads, the index arithmetic, the loop compare and branch,
# and the flash wait states on all of it — is what makes 10 the honest figure for
# a MAC in a loop. `log10f` has no hardware support at all and goes to newlib.
#
# Keep them here, named, so that when the device disagrees the disagreement is
# attributable to a line rather than to "the model".
CYCLES = {
    "ring_sample": 4,      # ldrsh over the EBI, vcvt, vmul by 1/32768, vstr
    "mac": 10,             # one tap of a filter or convolution, loop overhead in
    "butterfly": 22,       # radix-2: 4 mul, 6 add, 2 loads, 2 stores, indexing
    "log10f": 90,          # newlib logf on M4F; no hardware log
    "compare": 8,          # one qsort comparison, call included
    "roundf": 20,          # inlined (it was ~120 out-of-line through newlib)
}


def _defines(path: Path) -> dict[str, float]:
    """Numeric `#define`s from an exported `model_params.h`.

    Read rather than duplicated: these are the constants the firmware was compiled
    with, so an operation count derived from them cannot drift from the binary the
    way a hand-copied table does.
    """
    out = {}
    for m in re.finditer(r"^#define\s+(BG_[A-Z0-9_]+)\s+(-?[0-9.]+(?:e-?[0-9]+)?)f?\s*(?:/\*|$)",
                         path.read_text(), re.M):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def stage_class(stage: str) -> str:
    """Which of the two things `BG_SELFTEST` timed a stage belongs to.

    The self-test splits the chain exactly once — red LED for the front-end alone,
    green for the whole chain — so those two are the only classes any measurement
    can be attached to. Stages 10-12 are in neither: the self-test skips the
    adaptive threshold and the gate, which is why the remainder is estimated.
    """
    n = int(stage.split(".")[0])
    return "front-end" if n <= 6 else "model" if n <= 9 else "decision"


def calibration() -> dict[str, float]:
    """How wrong `CYCLES` is, per stage class, from the one arm that was timed.

    Derived live from `compute_model("cnn")` against `MEASURED_MS["cnn"]` rather
    than hard-coded, so it tracks any change to `CYCLES` instead of silently
    describing a version of the model that no longer exists.

    ⚠️ Two of these factors, applied to the other arm, are an EXTRAPOLATION: the
    front-end one is defensible (both arms run the same `fft.c` over the same 97
    frames), the model one is not (a dot product and a convolution stack do not
    miss by the same factor). The app labels a calibrated column as such.
    """
    d = compute_model("cnn")
    d["cls"] = d["stage"].map(stage_class)
    got = d.groupby("cls")["ms"].sum()
    m = MEASURED_MS["cnn"]
    return {c: float(m[c] / got[c]) for c in ("front-end", "model") if got.get(c, 0) > 0}


def compute_model(arm: str, calibrated: bool = False) -> pd.DataFrame:
    """Modelled core cycles per one second of audio, stage by stage.

    One second is the right denominator because the chain is a hard real-time loop:
    it gets one window's worth of core time per window, or it falls behind and — as
    the device showed — starts scoring audio the DMA has already overwritten. So
    the resource is `Silicon.core_hz` cycles and the panel is a budget, not a rate.

    `ops` is exact. `cycles` is `ops x CYCLES[kind]` and is the modelled part.
    """
    p = _defines(FIRMWARE / "arms" / arm / "model_params.h")
    F = int(p["BG_FRAMES_PER_WINDOW"])
    N = int(p["BG_FFT_SIZE"])
    half = N // 2
    bins = int(p["BG_N_BINS"])
    mels = int(p["BG_N_MELS"])
    nnz = int(p["BG_MEL_NNZ"])
    adapt_n = int(p["BG_ADAPT_N"])

    rows = [
        # (stage, what, ops, kind)
        ("1. Frame the sample ring", f"{F} frames x {N} int16 reads from external SRAM",
         F * N, "ring_sample"),
        ("2. FFT", f"{F} x (Hann {N} + {half//2} x {int(np.log2(half))} butterflies)",
         F * half // 2 * int(np.log2(half)), "butterfly"),
        ("2. FFT", f"{F} x {N} Hann multiplies", F * N, "mac"),
        ("3. Power spectrum", f"{F} x {bins} bins, 2 mul + 1 add", F * bins, "mac"),
        ("4. Mel filterbank", f"{F} x {nnz} non-zero weights", F * nnz, "mac"),
        ("4. Mel filterbank", f"{F} x {mels} log10f calls", F * mels, "log10f"),
    ]
    if arm.startswith("mfcc"):
        n_mfcc = int(p["BG_N_MFCC"])
        rows += [
            ("5. DCT / cepstrum", f"{F} x {n_mfcc} x {mels} dense matrix-vector",
             F * n_mfcc * mels, "mac"),
            ("6. Window statistics (Welford)", f"{F} x {n_mfcc} coefficients, ~5 ops each",
             F * n_mfcc * 5, "mac"),
            ("9. Model arithmetic", f"{int(p['BG_N_FEATURES'])}-weight dot product, once",
             int(p["BG_N_FEATURES"]), "mac"),
        ]
    else:
        # (conv, ReLU, 2x2 max-pool) x BG_N_CONV, streamed one log-mel column at a
        # time. The pool halves frequency AND time, so the column count entering
        # conv i is the previous conv's output halved, divided again by this conv's
        # time stride. Getting that schedule right matters: it is the difference
        # between 995,328 MACs and a figure 40% higher.
        #
        # ✅ This reckoning reproduces the 995,360 MACs/window that ROADMAP.md §C
        # reports as the theoretical count against 939,984 measured on the device,
        # which is the one independent check available on any number in this
        # function. `cycles` is still a model; `ops` is not.
        cols, freq = F, mels
        acts = 0
        for i in range(int(p["BG_N_CONV"])):
            o, ci, k = (int(p[f"BG_CONV{i}_OUT"]), int(p[f"BG_CONV{i}_IN"]),
                        int(p[f"BG_CONV{i}_K"]))
            cols //= int(p[f"BG_CONV{i}_STRIDE_T"])
            rows.append(("9. Model arithmetic", f"conv{i}: {o}x{ci}x{k}x{k} over "
                         f"{cols} time x {freq} freq columns",
                         cols * freq * o * ci * k * k, "mac"))
            acts += cols * freq * o
            cols, freq = cols // 2, freq // 2           # the 2x2 max-pool
            acts += cols * freq * o
        rows += [
            ("9. Model arithmetic", "final linear unit over the pooled features",
             int(p["BG_N_FEATURES"]), "mac"),
            # Every module quantises its output onto an int8 grid, which is a
            # round() per activation. Cheap per call and not cheap in aggregate:
            # inlining `roundf` alone took the model from 550 ms to 410 ms.
            ("8. Quantisers (int8 grid)", f"{acts:,} activations, one round() each",
             acts, "roundf"),
        ]

    rows += [
        ("10. Adaptive threshold", f"copy + qsort of {adapt_n} scores, once per window",
         int(adapt_n * np.log2(adapt_n)), "compare"),
        ("11. Gate state machine", "a handful of comparisons, once per window", 20, "mac"),
    ]

    df = pd.DataFrame(rows, columns=["stage", "what", "ops", "kind"])
    df["cycles_per_op"] = df["kind"].map(CYCLES)
    df["cycles"] = df["ops"] * df["cycles_per_op"]
    df["cls"] = df["stage"].map(stage_class)
    if calibrated:
        df["cycles"] = df["cycles"] * df["cls"].map(calibration()).fillna(1.0)
    df["ms"] = 1000 * df["cycles"] / SI.core_hz
    df["pct_of_budget"] = 100 * df["cycles"] / SI.core_hz
    return df


# Device measurements. `BG_SELFTEST` times the chain against a fixed synthetic
# signal, ten windows per phase, reading each LED's ON time; ROADMAP.md §C.
# The linear arm has NOT been timed this way — the entry below records that
# absence rather than filling it with the CNN's front-end number, which would be
# defensible (they share `fft.c`) and still be a fabrication.
MEASURED_MS = {
    "cnn": {"front-end": 380.0, "model": 410.0, "unmeasured remainder": 45.0,
            "note": "device, BG_SELFTEST, 10 windows/phase, after the four "
                    "optimisations of 2026-09-22 (was 450 + 550). The remainder is "
                    "ROADMAP's own estimate for fetch_frame, the sort and the gate, "
                    "which the self-test does not cover: 'call it 830-850 ms all in'."},
    "mfcc-dev": {"note": "never timed with BG_SELFTEST. ROADMAP infers ~55% headroom "
                         "from the fact that it keeps up while the CNN does not, and "
                         "that its model is a 40-weight dot product — an inference, "
                         "not a measurement."},
}

def compute_free(arm: str, calibrated: bool = False) -> int:
    """Cycles per second the chain does NOT spend — the core's idle headroom.

    Negative means the chain cannot keep up, which is not an abstract overrun: the
    device showed that `bg_stream` then reads windows the DMA has already
    overwritten, so the features are of nothing and the detector goes quiet while
    every internal check still looks healthy (ROADMAP.md §C).
    """
    return int(SI.core_hz - compute_model(arm, calibrated)["cycles"].sum())
