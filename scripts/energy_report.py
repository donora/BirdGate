"""G4c: price every arm and gate setting in current, not bytes.

    ../.venv/bin/python -m scripts.energy_report

Reads the measured operating points (duty and files/hour at MATCHED retention,
which is the only fair comparison — see docs/lab-notebook.md §9b) and pushes them through
`birdgate.energy`. Prints four things, in the order that decides what to do next:

1. Where the current actually goes, per arm.
2. What the gate buys against doing nothing — the number that motivates the project.
3. Sensitivity to `file_rewrite_bytes`, the one unmeasured constant that reorders arms.
4. The bench protocol that would replace the estimates with measurements.

⚠️ Absolute mA here are ESTIMATES (see `energy.EnergyParams`). The comparisons and
break-evens are robust; the deployment-day figures are not, until the bench runs.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from birdgate import energy as E

# duty, files/hr at theta swept on test for 98.0% retention (scripts.compare_arms
# frontier, extended to report file counts), + MMAC/s from the arm cost model.
OPERATING_POINTS = [
    # arm,        duty,   files/hr, MMAC/s
    ("carihc",    0.9561, 11.17,    8.96),
    ("bands",     0.9191, 17.33,    7.68),
    ("mfcc",      0.9148, 33.67,    1.54),
    ("cnn-int8",  0.8747, 49.17,    3.17),
]
UNGATED = ("ungated", 1.0, 1.0, 7.68)
HIGHLAND = [("highland 10% duty", 0.10, 8.0, 1.54), ("highland 5% duty", 0.05, 5.0, 1.54)]


def breakdown(p: E.EnergyParams, points=OPERATING_POINTS) -> pd.DataFrame:
    rows = []
    for arm, duty, files, mmac in points:
        r = E.deployment_days(duty, files, mmac, p)
        rows.append({"arm": arm, "duty_%": 100 * duty, "files_hr": files,
                     "mcu_mA": r["mcu_ma"], "analog_mA": r["analog_ma"],
                     "sd_write_mA": r["sd_write_ma"], "sd_file_mA": r["sd_files_ma"],
                     "sd_idle_mA": r["sd_idle_ma"], "total_mA": r["total_ma"],
                     "card_d": r["card_days"], "batt_d": r["battery_days"],
                     "life_d": r["deployment_days"], "binds": r["binds"]})
    return pd.DataFrame(rows).round(2)


def file_sensitivity(p: E.EnergyParams, kbs=(64, 256, 1024, 4096)) -> pd.DataFrame:
    """Total current per arm as the per-file cost sweeps its plausible range.

    64 kB = one erase block; 4 MB = a full SDHC allocation unit per close. Two
    orders of magnitude of genuine uncertainty, which is why this is a sweep and
    not a number.
    """
    rows = []
    for kb in kbs:
        q = replace(p, file_rewrite_bytes=kb * 1000.0)
        row = {"per_file_kB": kb}
        for arm, duty, files, mmac in OPERATING_POINTS:
            row[arm] = round(E.current_ma(duty, files, mmac, q)["total_ma"], 3)
        row["cheapest"] = min((a for a, *_ in OPERATING_POINTS), key=lambda k: row[k])
        rows.append(row)
    return pd.DataFrame(rows)


def break_even_matrix() -> pd.DataFrame:
    """Per-file cost at which each arm stops beating each other arm."""
    rows = []
    for a, da, fa, _ in OPERATING_POINTS:
        for b, db, fb, _ in OPERATING_POINTS:
            if a == b:
                continue
            c = E.break_even_c(da, fa, db, fb)
            if c == c and c != float("inf"):   # finite: a real trade exists
                rows.append({"arm": a, "beats": b, "break_even_s_per_file": round(c, 2),
                             "kB_equivalent": round(c * E.BYTES_PER_S / 1000)})
    return pd.DataFrame(rows)


BENCH_PROTOCOL = """
Three measurements replace every estimate in EnergyParams. An afternoon with an
AudioMoth (or any EFM32 board + SD socket), a current shunt and a power analyser
or scope. Cruder but still decisive: a card in a USB reader and an inline USB
power meter, which gets the SD constants but not the MCU ones.

  M1  SD idle and per-file cost -> i_sd_idle_ma, file_rewrite_bytes, wake_rewrite_bytes
      Write N files of payload P, sweeping P over {1 kB, 64 kB, 1 MB, 16 MB} at
      fixed N, then N over {1, 10, 100} at fixed P. Total energy is linear in
      (N, N*P); the intercept in N IS the per-file cost, and separating it from the
      per-byte slope is the entire point of sweeping both. Repeat with the card
      powered down between files to get wake_rewrite_bytes. Repeat on at least two
      card brands — the spread between cards is expected to exceed the spread
      between our arms.

  M2  MCU regime -> cpu_sleeps, i_mcu_full_ma, i_mcu_idle_ma
      Run the firmware loop with the classifier stubbed out, then with each arm's
      front-end live. If current does not move between arms, the core is not
      sleeping and MMAC/s is not an energy quantity. This single question is worth
      more than any model comparison in docs/lab-notebook.md §9.

  M3  Analog floor -> i_analog_ma
      Mic + preamp + ADC + DMA running, MCU otherwise idle, no SD. This is the
      floor no gate can get below, and it bounds what the whole project can ever
      save.

Then: re-fit EnergyParams, re-run this report, and re-rank the arms on
deployment_days instead of on saving. Also re-sweep min_on_s and recheck_period_s,
which are revealed as energy parameters rather than ergonomics ones.
"""


def main() -> None:
    pd.set_option("display.width", 200)
    base, slp = E.DEFAULT, replace(E.DEFAULT, f_peripheral_mhz=8.0)

    print("=" * 100)
    print("1. WHERE THE CURRENT GOES  (realistic 3xAA = 2,600 mAh; EM1 clock tree at 48 MHz)")
    print("=" * 100)
    print(breakdown(base).to_string(index=False))
    print("\n   Same, with the peripheral clock reduced to what the ADC actually needs:")
    print(breakdown(slp).to_string(index=False))

    print("\n" + "=" * 100)
    print("2. WHAT THE GATE BUYS, against the alternatives (battery-days)")
    print("=" * 100)
    for lbl, p in (("EM1 @48MHz peri", base), ("EM1 @8MHz peri", slp)):
        _, du, fu, mu = UNGATED
        ung = E.deployment_days(du, fu, mu, p)["battery_days"]
        gat = E.deployment_days(0.9191, 17.33, 7.68, p)["battery_days"]
        print(f"  {lbl:15s} ungated {ung:6.2f} d -> gated {gat:6.2f} d "
              f"({100*(gat/ung-1):+5.1f}%)")
        for hl, d, f, m in HIGHLAND:
            r = E.deployment_days(d, f, m, p)
            print(f"      {hl:20s} {r['battery_days']:6.2f} d battery | "
                  f"{r['card_days']:7.1f} d card | binds: {r['binds']}")

    print("\n" + "=" * 100)
    print("3. SENSITIVITY to the one unmeasured constant that reorders the arms")
    print("=" * 100)
    print(file_sensitivity(base).to_string(index=False))
    print("\n   Break-even per-file cost, pairwise:")
    print(break_even_matrix().to_string(index=False))

    print("\n" + "=" * 100)
    print("4. BENCH PROTOCOL")
    print("=" * 100)
    print(BENCH_PROTOCOL)


if __name__ == "__main__":
    main()
