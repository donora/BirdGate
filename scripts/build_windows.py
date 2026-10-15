"""Build windows and print the oracle-gate report.

    ../.venv/bin/python -m scripts.build_windows

Regenerates every number in docs/lab-notebook.md section 5, so the plan's table is
reproducible rather than folklore. Writes a run manifest for traceability.
"""
from __future__ import annotations

import pandas as pd

from birdgate import data as D, gate as G, metrics as M, provenance as P, windows as W
from birdgate.config import GateConfig, GateParams, WindowParams

SETTINGS = [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2), (3, 5)]


def main() -> None:
    pd.set_option("display.width", 200)
    p = WindowParams(win_s=1.0, train_hop_s=1.0, eval_hop_s=1.0, min_overlap_s=0.15)
    windows, y = W.build_windows(p=p)
    windows = windows.reset_index(drop=True)

    print("== windows ==")
    print(W.summarise(windows, y).to_string(), "\n")

    occ = W.occupancy()
    print("== occupancy per recording ==")
    print(f"min {occ.occupancy_pct.min():.1f}%  median {occ.occupancy_pct.median():.1f}%  "
          f"max {occ.occupancy_pct.max():.1f}%\n")

    rows = []
    for pre, hang in SETTINGS:
        gp = GateParams(theta_on=0.5, theta_off=0.5, hysteresis_margin=0.0,
                        pre_roll_s=pre, hangover_s=hang)
        m = M.evaluate(windows, G.oracle_scores(y), gp)
        ps = M.per_species_retention(windows, m["_regions"])
        rows.append({
            "pre_roll_s": pre, "hangover_s": hang,
            "retention_%": 100 * m["birdsong_retention"],
            "whole_boxes_%": 100 * m["event_recall@1"],
            "duty_%": 100 * m["duty_cycle"], "saving_%": 100 * m["saving"],
            "species_floor_%": 100 * M.retention_floor(ps, min_boxes=20),
        })
    table = pd.DataFrame(rows).round(2)
    print("== oracle gate (ground truth through the real state machine) ==")
    print(table.to_string(index=False))
    print("\nKnee: pre-roll 1 s, hangover 1 s — 100% retention, 100% whole boxes, "
          "100% species floor. Longer hangovers cost saving and buy nothing.")

    out = P.write(GateConfig(name="g1-oracle"),
                  metrics={"oracle": table.to_dict("records")},
                  notes="Oracle gate over all 33 recordings, 1 s decision window.")
    print(f"\nmanifest -> {out}")


if __name__ == "__main__":
    main()
