"""The scoreboard's numbers, computed from the shipped `demo/scores.npz`.

    python scripts/tour/scoreboard_numbers.py      # -> docs/tour/scoreboard.json

Same procedure as `scripts.compare_arms.frontier`: theta swept on the test split to
the lowest setting that keeps 98% of birdsong seconds, the shipping gate, and the
shared `scripts.train.report` for every metric. On top of that it reports what a
fixed "card saved" figure hides: how much a *perfect* detector saves behind the same
gate, recording by recording, and splits each arm's result into what it got right
and what it got wrong:

- **silence skipped**: of the silence a perfect detector behind the same gate skips,
  the share this arm also skips. Only birdsong-free seconds count, so it cannot pass
  100% by throwing birdsong away, which a ratio of card savings can.
- **birdsong missed**: the share of birdsong seconds the gate did not record.

Card saved moves with how much of an hour is birdsong; these two are about the model.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from birdgate import adapt as A, data as D, intervals as I, metrics as M, windows as W  # noqa: E402
from birdgate.config import GateParams, WindowParams  # noqa: E402
from scripts.train import ADAPT, GATE, report  # noqa: E402

ARMS = ("mfcc", "bands", "carihc", "cnn-int8")
TARGET = 0.98


def skipped(regions: np.ndarray, dur: float) -> np.ndarray:
    """The complement of the recorded regions within [0, dur]: what the gate skipped."""
    edges = np.concatenate([[0.0], I.merge(regions).ravel(), [dur]]).reshape(-1, 2)
    return edges[edges[:, 1] > edges[:, 0]]


def main() -> None:
    w, y = W.build_windows(p=WindowParams())
    w = w.reset_index(drop=True)
    tr = w.index[w["split"] == "train"].to_numpy()
    te = w.index[w["split"] == "test"].to_numpy()
    test = w.loc[te].reset_index(drop=True)
    groups = list(w.loc[te].groupby("filename"))
    ann, recs = D.load_annotations(), D.load_recordings()
    gp = lambda th: GateParams(theta_on=float(th), theta_off=None, **GATE)  # noqa: E731

    truth = y.astype(float)
    ceiling = {f: M.evaluate(g.reset_index(drop=True), truth[g.index], gp(0.5), ann, recs)["saving"]
               for f, g in groups}
    dur = dict(zip(recs["filename"], recs["duration_s"]))
    bird = {f: I.merge(ann.loc[ann["filename"] == f, ["start_s", "end_s"]].to_numpy()) for f, _ in groups}
    oracle_regions = M.evaluate(test, truth[te], gp(0.5), ann, recs)["_regions"]
    # what a perfect detector skips; at 100% retention none of it is birdsong
    can_skip = {f: skipped(oracle_regions.get(f, np.zeros((0, 2))), dur[f]) for f, _ in groups}
    out = {"target_retention": TARGET,
           "ceiling": M.evaluate(test, truth[te], gp(0.5), ann, recs)["saving"],
           "ceiling_by_recording": ceiling, "arms": {}}

    with np.load(D.DEMO / "scores.npz", allow_pickle=False) as z0:
        for arm in ARMS:
            raw = z0[f"raw_{arm}"].astype(np.float64)
            z = A.normalise_by_recording(w, raw, replace(ADAPT, global_scale=float(raw[tr].std()) or 1.0))
            pts = sorted((M.evaluate(test, z[te], gp(th), ann, recs)["birdsong_retention"], th)
                         for th in np.quantile(z[te], np.linspace(0.02, 0.995, 80)))
            th = next(t for r, t in pts if r >= TARGET)
            r = report(w, z, th, GATE, {"macs_per_s": 0.0, "pct_of_48mhz_m4f": 0.0})
            per = {f: M.evaluate(g.reset_index(drop=True), z[g.index.to_numpy()], gp(th), ann, recs)["saving"]
                   for f, g in groups}
            share = {f: per[f] / ceiling[f] for f in per}
            regions = M.evaluate(test, z[te], gp(th), ann, recs)["_regions"]
            hit, room, missed, song = {}, {}, {}, {}
            for f, _ in groups:
                reg = regions.get(f, np.zeros((0, 2)))
                hit[f] = I.total(I.intersect(skipped(reg, dur[f]), can_skip[f]))
                room[f] = I.total(can_skip[f])
                song[f] = I.total(bird[f])
                missed[f] = song[f] - I.total(I.intersect(bird[f], reg))
            silence = {f: hit[f] / room[f] for f in hit}
            miss = {f: missed[f] / song[f] for f in missed}
            out["arms"][arm] = {
                "retention": r["retention"], "saving": r["saving"] / 100,
                "share": (r["saving"] / 100) / out["ceiling"],
                "silence_skipped": sum(hit.values()) / sum(room.values()),
                "silence_skipped_by_recording": silence,
                "birdsong_missed": sum(missed.values()) / sum(song.values()),
                "birdsong_missed_by_recording": miss,
                "saving_by_recording": per, "share_by_recording": share,
                "worst_recording": r["worst_rec"], "species_floor": r["species_floor"],
                "floor_species": r["floor_species"],
            }
            n = out["arms"][arm]
            print(f"{arm:9s} silence skipped {n['silence_skipped']:.0%} "
                  f"({min(silence.values()):.0%}-{max(silence.values()):.0%})  birdsong missed "
                  f"{n['birdsong_missed']:.1%} (worst {max(miss.values()):.1%})")
            print(f"{arm:9s} saved {r['saving']:.1f}%  share {out['arms'][arm]['share']:.0%} "
                  f"({min(share.values()):.0%}-{max(share.values()):.0%})  worst rec "
                  f"{r['worst_rec']:.1f}%  worst species {r['species_floor']:.1f}% ({r['floor_species']})")
    dst = ROOT / "docs" / "tour" / "scoreboard.json"
    dst.write_text(json.dumps(out, indent=1))
    print(dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
