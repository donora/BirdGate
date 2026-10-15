"""G4c: what a configuration actually costs in current, not in bytes.

**Why this module exists.** Until it was written the project optimised *bytes
written to the card* and called the result a deployment benefit. `metrics.deployment_life`
returned a `battery_days` that did not read `duty_cycle` at all — the same 48.1 days
whether the gate recorded everything or nothing. Every arm comparison in docs/lab-notebook.md §8-§9b
is therefore a comparison on card bytes with a compute *feasibility* check
(`pct_of_48mhz_m4f`) bolted on, and neither is energy.

Three corrections, in descending order of how much they change the answer:

**1. Writing is not proportional to bytes.** Energy per hour is
`per_byte x bytes + per_file x files + idle x time`. The per-file term is the one
that bites: a FAT32 close dirties the directory entry and both FAT copies, which
live in a different region of the card from the audio stream, so the controller
does a read-modify-write on a *separate* allocation unit (typically 4 MB on
SDHC/SDXC). If the firmware also power-cycles the card between recordings, each
file additionally pays a full re-init. Our arms differ by **2.8x in files/hour at
matched retention** (`bands` 17.3, `cnn-int8` 49.2), so this term reorders them —
see `break_even_c`.

**2. Compute is a *duty cycle*, not a switch.** The core never runs continuously in
a competent always-on design: DMA fills a buffer while the core sits in EM1, an
interrupt wakes it to process the block, and it sleeps again. So an arm can be
"running" and the core "asleep" simultaneously, and **MMAC/s is exactly the fraction
of wall-clock time the core spends in EM0** — 3.2% for `mfcc`, 18.7% for `carihc`.
The arms therefore cost 0.3-1.9 mA in wake bursts, against an arm-independent floor
(EM1 clock tree + the analog chain) of 2.0-4.5 mA. **The floor dominates, and
configuring the peripheral clock is worth more than choosing the front-end.**

**3. ⚠️ The inherited battery capacity is almost certainly wrong.**
`metrics.deployment_life` uses 10,500 mAh at 4.5 V = 47 Wh, which requires each of
three *series* AA cells to hold 10,500 mAh. Series connection sums voltage, not
capacity: the pack's mAh is one cell's mAh, so 3xAA is ~2,400-3,200 mAh (11-14 Wh).
That is a factor of ~3.5, and it inverts the framing the whole project rests on —
at 128 GB the card lasts 23 ungated days, so with 48 battery-days the card binds
and the gate is the product, while with ~12 battery-days the **battery** binds and
a gate that saves no battery extends nothing. `DEFAULT` uses the realistic figure;
`PAPER_BATTERY` reproduces the inherited one so the difference stays visible.

⚠️⚠️ **Every current in `EnergyParams` except the battery capacity is an ESTIMATE,
not a measurement.** They are order-of-magnitude values from datasheet-class
reasoning, and SD cards in particular vary by more than 10x between brands on both
idle draw and per-file cost. Treat absolute outputs as provisional. The *ratios*
and the break-even points are far more robust than the absolute mA, which is why
`break_even_c` is the function to trust and `deployment_days` is the one to
re-derive after `scripts/bench_energy.py` has run on real hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

BYTES_PER_S = 32_000 * 16 / 8
"""64 kB/s: the 32 kHz 16-bit mono stream the gate is deciding about."""


@dataclass(frozen=True)
class EnergyParams:
    """Current budget for one AudioMoth. All currents in mA at `v_supply`.

    Fields are grouped by how much we trust them: the battery and card capacities
    are arithmetic, the MCU currents are datasheet-class, and the SD figures are
    the ones that need a bench.
    """

    # --- capacities: arithmetic, trustworthy -----------------------------------
    v_supply: float = 4.5              # 3xAA in series
    battery_mah: float = 2_600.0       # ONE cell's capacity; series sums volts, not mAh
    card_gb: float = 128.0

    # --- MCU: datasheet-class, moderately trustworthy --------------------------
    # ⚠️ "Does the core sleep?" is NOT a yes/no about whether the arm runs. On an
    # always-on audio front-end the DMA fills a buffer (say 512 samples = 16 ms)
    # while the core sits in EM1; the buffer-full interrupt wakes it, it pushes
    # those samples through the filterbank in a burst, and it returns to EM1. The
    # arm is "running" and the core is "asleep" at the same time, time-sliced at
    # millisecond granularity. So:
    #
    #   **MMAC/s IS the EM0 wake duty cycle**, not an abstract feasibility ratio.
    #   `pct_of_48mhz_m4f` is literally the fraction of wall-clock time the core
    #   spends executing: carihc 18.7%, bands 16.0%, cnn 6.6%, mfcc 3.2%.
    #
    # A core pinned in EM0 is not a design choice, it is a missing sleep loop, and
    # it is the one configuration in which every arm costs the same.
    f_core_mhz: float = 48.0
    em0_ua_per_mhz: float = 211.0     # core executing from flash
    em1_ua_per_mhz: float = 63.0      # core clock gated, peripheral tree still on
    f_peripheral_mhz: float = 48.0
    """⚠️ The EM1 floor scales with the clock tree left running, and this is the
    largest arm-INDEPENDENT lever in the whole budget. At 48 MHz the floor is
    ~3.0 mA and swamps the 0.3-1.9 mA the arms differ by; clocked down to what the
    ADC and DMA actually need (~8 MHz) it is ~0.5 mA and the arm choice starts to
    matter. Configuring this is worth more than choosing the front-end."""

    core_always_em0: bool = False
    """Escape hatch for modelling a firmware with no sleep loop at all. Not a
    regime anyone would choose — kept only because the first version of this model
    assumed it, and the difference (11.6 mA vs 2.3-5.9 mA) should stay visible."""

    i_analog_ma: float = 1.5
    """MEMS mic + preamp + ADC + DMA. **Runs 100% of the time or the pre-roll ring
    buffer does not exist.** This is the floor the gate can never get below, it is
    identical for every arm, and dropping it is the single biggest energy lever
    available — at the cost of pre-roll, which docs/lab-notebook.md §6 measured as the
    best-value recall parameter in the system (0->1 s lifts whole-box capture from
    91.7% to 97.0%). That trade is a real design decision, not an optimisation."""

    # --- SD card: ESTIMATES. This is what the bench is for. --------------------
    i_sd_write_ma: float = 40.0        # active draw while streaming to the card
    sd_throughput_bps: float = 2.0e6   # effective sequential write rate
    i_sd_idle_ma: float = 0.5
    """Draw while powered but not writing. Varies 0.1-5 mA across cards and is a
    pure constant — at 0.5 mA it is already ~5% of the whole budget, and a bad card
    could quietly dominate everything this project optimises."""

    sd_powered_when_idle: bool = True

    file_rewrite_bytes: float = 256_000.0
    """⚠️ **The single most consequential unmeasured number in this project.**

    Parameterised as *bytes of internal card programming triggered by one file
    create + close*, rather than as an energy directly, because that is the
    physical mechanism and it is what a bench measurement actually returns. A close
    dirties the directory entry and both FAT copies; those sectors sit in a
    different allocation unit from the audio stream, so the controller
    read-modify-writes that AU. AU size is typically 4 MB on SDHC/SDXC, so the
    plausible range spans **64 kB (optimistic, controller coalesces) to 4 MB (a
    full AU per close)** — two orders of magnitude.

    256 kB is chosen as the default precisely because it lands *at* the
    `cnn-int8` vs `bands` break-even (~314 kB). It is not a claim that the CNN and
    the filterbank tie; it is a refusal to pre-judge, so that anyone reading a
    ranking out of this model has to look at `break_even_c` first."""

    wake_rewrite_bytes: float = 0.0
    """Card re-initialisation cost, if the firmware power-cycles the card between
    recordings — charged once per file, in the same units. Zero because we have not
    established that it does; if it does, expect it to be comparable to
    `file_rewrite_bytes` and to roughly double the per-file term."""


DEFAULT = EnergyParams()
PAPER_BATTERY = EnergyParams(battery_mah=10_500.0)
"""The inherited assumption, kept so the 3.5x can be shown rather than asserted."""


def per_byte_mas(p: EnergyParams = DEFAULT) -> float:
    """mA-seconds per byte written, from write current and effective throughput."""
    return p.i_sd_write_ma / p.sd_throughput_bps


def audio_second_mas(p: EnergyParams = DEFAULT) -> float:
    """Write energy of one second of gated audio — the unit `break_even_c` uses."""
    return per_byte_mas(p) * BYTES_PER_S


def current_ma(duty: float, files_per_hr: float, mmac_s: float = 0.0,
               p: EnergyParams = DEFAULT) -> dict:
    """Mean current for a configuration, broken out by where it goes.

    `duty` is a fraction, `files_per_hr` a count. Returns a dict of contributions
    in mA plus their total, because the breakdown is the useful part: it says
    immediately whether a decision is worth making at all.
    """
    em0 = p.em0_ua_per_mhz * p.f_core_mhz / 1000.0
    em1 = p.em1_ua_per_mhz * p.f_peripheral_mhz / 1000.0
    util = min(1.0, mmac_s * 1e6 / (p.f_core_mhz * 1e6))
    mcu = em0 if p.core_always_em0 else util * em0 + (1.0 - util) * em1
    write = per_byte_mas(p) * BYTES_PER_S * duty
    files = files_per_hr * per_byte_mas(p) * (
        p.file_rewrite_bytes + p.wake_rewrite_bytes) / 3600.0
    idle = p.i_sd_idle_ma if p.sd_powered_when_idle else 0.0
    total = mcu + p.i_analog_ma + write + files + idle
    return {"mcu_ma": mcu, "analog_ma": p.i_analog_ma, "sd_write_ma": write,
            "sd_files_ma": files, "sd_idle_ma": idle, "total_ma": total}


def deployment_days(duty: float, files_per_hr: float, mmac_s: float = 0.0,
                    p: EnergyParams = DEFAULT) -> dict:
    """Card-days, battery-days, and **which one binds** — the deliverable.

    The binding resource is the whole point. It decides where theta should sit
    (docs/lab-notebook.md §14: "what is actually scarce at the Highland site?"), and it is not
    answerable without both halves of this calculation.
    """
    e = current_ma(duty, files_per_hr, mmac_s, p)
    gb_per_day = BYTES_PER_S * 86400 * duty / 1e9
    card = p.card_gb / gb_per_day if gb_per_day > 0 else float("inf")
    batt = p.battery_mah / (e["total_ma"] * 24)
    return {**e, "gb_per_day": gb_per_day, "card_days": card, "battery_days": batt,
            "deployment_days": min(card, batt),
            "binds": "card" if card < batt else "battery"}


def break_even_c(duty_a: float, files_a: float, duty_b: float, files_b: float) -> float:
    """Per-file cost at which config A stops beating config B, in audio-seconds.

    Units are deliberately *seconds of sequential-write energy per file* rather
    than mA-seconds: it is dimensionless with respect to the card's write current
    and throughput, so it survives not knowing either. Multiply by
    `audio_second_mas()` for mA-s, or by `BYTES_PER_S` for a byte-equivalent.

    Returns `inf` when A writes no more files than B (A wins at any per-file cost),
    and `nan` when A is worse on both axes and there is nothing to trade.
    """
    d_files = files_a - files_b
    d_duty = (duty_b - duty_a) * 3600.0
    if d_files <= 0:
        return float("inf") if d_duty >= 0 else float("nan")
    return float("nan") if d_duty < 0 else d_duty / d_files


def energy_equivalent_seconds(duty: float, files_per_hr: float, c: float) -> float:
    """Write-energy per hour in audio-seconds, given a per-file cost `c`.

    The card-independent objective: rank configurations without pinning
    `i_sd_write_ma` or `sd_throughput_bps` at all. Only `c` has to be guessed, and
    `break_even_c` says how accurately.
    """
    return duty * 3600.0 + files_per_hr * c
