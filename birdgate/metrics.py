"""Gate metrics. Classification metrics are the wrong frame — judge the card.

Everything here is computed **after** `gate.run()` and in **continuous time**, via
`intervals.py` — never on raw per-window scores. Two reasons, both load-bearing:

- Pre-roll and hangover change every number, so a model compared on raw AUC can
  lose to a worse model that interacts better with the state machine.
- The thing being judged is a set of recorded *seconds of audio*, not a set of
  correct window decisions. Those differ whenever the hop and the window differ.

**Retention is measured against the union of annotation boxes, not their sum.**
Birds sing over each other constantly; summing box durations double-counts the
overlap and quietly inflates the denominator.

The metric that decides whether this is deployable is `per_species_retention`, and
specifically its **minimum**, not its mean — see that function's docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import gate as G
from . import intervals as I


def _boxes_by_file(ann: pd.DataFrame) -> dict[str, np.ndarray]:
    return {f: g[["start_s", "end_s"]].to_numpy() for f, g in ann.groupby("filename")}


def evaluate(windows: pd.DataFrame, scores: np.ndarray, p, ann=None,
             recordings=None, event_ks=(0.5, 0.9, 1.0)) -> dict:
    """Run the gate and score it. The single entry point for every arm.

    Returns a flat dict of the metrics defined below, plus the recorded regions so
    callers can plot or re-use them without re-running the state machine.
    """
    if ann is None:
        ann = D.load_annotations()
    if recordings is None:
        recordings = D.load_recordings()

    files = set(windows["filename"].unique())
    ann = ann[ann["filename"].isin(files)]
    dur = {r.filename: r.duration_s for r in recordings.itertuples() if r.filename in files}

    recorded = G.run_all(windows, scores, p)
    regions = G.regions(windows, recorded)

    boxes = _boxes_by_file(ann)
    bird_total = kept_total = 0.0
    rec_total = wall_total = 0.0
    for f in files:
        reg = regions.get(f, np.zeros((0, 2)))
        u = I.merge(boxes.get(f, np.zeros((0, 2))))
        bird_total += I.total(u)
        kept_total += I.total(I.intersect(u, reg))
        rec_total += I.total(reg)
        wall_total += dur.get(f, 3600.0)

    # Event recall: per-BOX, not per-second. The 99th-percentile box is 32 s long,
    # so second-level retention is dominated by a handful of long boxes; event
    # recall weights a 0.3 s chip equally. Short quiet chips are precisely what a
    # gate drops, which makes this the sensitive metric of the two.
    ov = np.zeros(len(ann))
    idx = 0
    frac = np.zeros(len(ann))
    for f, g in ann.groupby("filename", sort=False):
        reg = regions.get(f, np.zeros((0, 2)))
        o = I.overlap(g["start_s"].to_numpy(), g["end_s"].to_numpy(), reg)
        d = (g["end_s"] - g["start_s"]).to_numpy()
        sl = slice(idx, idx + len(g))
        ov[sl] = o
        frac[sl] = np.divide(o, d, out=np.ones_like(o), where=d > 0)
        idx += len(g)

    out = {
        "birdsong_retention": kept_total / bird_total if bird_total else float("nan"),
        "duty_cycle": rec_total / wall_total if wall_total else float("nan"),
        "saving": 1 - rec_total / wall_total if wall_total else float("nan"),
        "bird_hours": bird_total / 3600,
        "recorded_hours": rec_total / 3600,
        "wall_hours": wall_total / 3600,
        "n_boxes": len(ann),
    }
    for k in event_ks:
        out[f"event_recall@{k:g}"] = float((frac >= k - 1e-9).mean())
    out["_regions"] = regions
    out["_recorded"] = recorded
    out["_box_frac"] = frac
    return out


def per_species_retention(windows: pd.DataFrame, regions: dict, ann=None,
                          n_boot: int = 0, seed: int = 0, ci: float = 95.0) -> pd.DataFrame:
    """Retention per species, and the floor over species. **The deployment risk.**

    A gate at 97% overall retention that keeps 99% of a loud abundant species and
    45% of a quiet distant one has injected a species bias into every downstream
    analysis those recordings will ever support — and it is *invisible*, because
    the discarded audio no longer exists to audit. Nothing downstream can recover
    from it and no amount of later cleverness can detect it.

    So the acceptance criterion is the MINIMUM over species with enough data to
    score, not the mean. Report `n_boxes` alongside: a species with 3 boxes has a
    retention estimate too noisy to gate a decision on, and must be shown as such
    rather than silently dragging the floor down.

    `n_boot > 0` adds a bootstrap CI by resampling song EVENTS (boxes) — the unit
    that actually varies between deployments, since a species is not observed one
    second at a time. Calibration is why this exists: on val, Brown-headed Cowbird scores
    61% retention from 21 boxes and 41 seconds of song, 95% CI **32-95**. That
    estimate cannot support any decision, yet it set the floor for the whole
    project, because `min_boxes` guards on box COUNT while precision depends on
    total DURATION — 21 boxes clears a threshold of 20. Use `max_ci_width` in
    `retention_floor` to exclude species that cannot be measured rather than
    letting them dictate the operating point.

    ⚠️ The point estimate uses the UNION of boxes (overlapping songs of one species
    counted once); the CI resamples boxes without re-merging, because re-merging
    each resample is too slow to run inside a theta sweep. Where same-species boxes
    overlap the two differ slightly — ~1 point in the cowbird case. The CI is a
    precision guard on the estimate, not a posterior interval for the union figure,
    and is documented as such rather than quietly reconciled.
    """
    if ann is None:
        ann = D.load_annotations()
    ann = ann[ann["filename"].isin(set(windows["filename"].unique()))]
    sp = D.load_species().set_index("code")
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2

    rows = []
    for code, g in ann.groupby("code"):
        bird = kept = 0.0
        dur, ov = [], []
        for f, gg in g.groupby("filename"):
            reg = regions.get(f, np.zeros((0, 2)))
            u = I.merge(gg[["start_s", "end_s"]].to_numpy())
            bird += I.total(u)
            kept += I.total(I.intersect(u, reg))
            dur.append((gg["end_s"] - gg["start_s"]).to_numpy())
            ov.append(I.overlap(gg["start_s"].to_numpy(), gg["end_s"].to_numpy(), reg))
        r = {
            "code": code,
            "common": sp["common"].get(code, code),
            "n_boxes": len(g),
            "bird_s": bird,
            "retention": kept / bird if bird else np.nan,
        }
        if n_boot:
            d, o = np.concatenate(dur), np.concatenate(ov)
            if d.sum() > 0:
                b = rng.integers(0, len(d), size=(n_boot, len(d)))
                boot = o[b].sum(1) / d[b].sum(1)
                r["ci_lo"], r["ci_hi"] = np.percentile(boot, [lo_q, hi_q])
                r["ci_width"] = r["ci_hi"] - r["ci_lo"]
            else:
                r["ci_lo"] = r["ci_hi"] = r["ci_width"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("retention").reset_index(drop=True)


def admissible(per_species: pd.DataFrame, min_boxes: int = 20,
               max_ci_width: float | None = None) -> pd.DataFrame:
    """The species whose retention is measured well enough to constrain a decision.

    Species excluded here are not "ignored" — they are species this dataset cannot
    make a claim about, and the honest move is to say so and report how many were
    dropped, rather than to average them away or let them set the floor.

    ⚠️ `max_ci_width` is evaluated at the CURRENT theta, so the admissible set
    changes along a sweep: a CI is narrow near 100% retention and near 0%, and
    widest in between. In the shipping region (retention >= 90%) almost every
    species qualifies and this is harmless. At aggressive thetas the guard loosens
    precisely where retention is worst, so **a floor read off a low-retention row
    is not a safety guarantee**. Read those rows with `n_admissible`.
    """
    ok = per_species[per_species["n_boxes"] >= min_boxes]
    if max_ci_width is not None and "ci_width" in ok.columns:
        ok = ok[ok["ci_width"].notna() & (ok["ci_width"] <= max_ci_width)]
    return ok


def retention_floor(per_species: pd.DataFrame, min_boxes: int = 20,
                    max_ci_width: float | None = None) -> float:
    """Worst retention among species precise enough to be worth believing."""
    ok = admissible(per_species, min_boxes, max_ci_width)
    return float(ok["retention"].min()) if len(ok) else float("nan")


def oracle_curve(windows: pd.DataFrame, y: np.ndarray, pre_rolls=(0.0, 0.5, 1.0, 2.0, 3.0),
                 hangovers=(0.0, 1.0, 2.0, 3.0, 5.0), ann=None, recordings=None) -> pd.DataFrame:
    """What a PERFECT detector achieves, per (pre-roll, hangover) setting.

    This is the ceiling and the control in one. Every model result is read against
    the row with the same pre-roll/hangover: a model at 96% retention where the
    oracle gets 100% has a 4% model deficit, whereas a model at 96% where the
    oracle gets 96.5% is nearly perfect and the loss is the state machine's.
    Without this table the two are indistinguishable.
    """
    from .config import GateParams
    if ann is None:
        ann = D.load_annotations()
    if recordings is None:
        recordings = D.load_recordings()
    s = G.oracle_scores(y)
    rows = []
    for pre in pre_rolls:
        for hang in hangovers:
            p = GateParams(theta_on=0.5, theta_off=0.5, hysteresis_margin=0.0,
                           pre_roll_s=pre, hangover_s=hang)
            m = evaluate(windows, s, p, ann=ann, recordings=recordings)
            rows.append({
                "pre_roll_s": pre, "hangover_s": hang,
                "retention": m["birdsong_retention"],
                "duty_cycle": m["duty_cycle"], "saving": m["saving"],
                "event_recall@0.9": m["event_recall@0.9"],
                "event_recall@1": m["event_recall@1"],
            })
    return pd.DataFrame(rows)


def deployment_life(duty_cycle: float, sample_rate: int = 32_000, bit_depth: int = 16,
                    card_gb: float = 128.0, battery_mah: float = 10_500.0,
                    current_ma: float = 9.1) -> dict:
    """Card-days and battery-days implied by a duty cycle.

    Battery uses the 9.1 mA always-on figure inherited from the first estimate (see energy.py). Note the gate does
    NOT save battery in the always-on regime — the cochlea runs on every second
    regardless; only the SD writes are avoided. So the honest framing is that the
    gate buys **card life and analysis time**, and battery only via avoided writes.
    Whichever of the two runs out first is the real deployment limit.
    """
    bytes_per_s = sample_rate * bit_depth / 8
    gb_per_day = bytes_per_s * 86400 * duty_cycle / 1e9
    return {
        "gb_per_day": gb_per_day,
        "card_days": card_gb / gb_per_day if gb_per_day > 0 else float("inf"),
        "battery_days": battery_mah / (current_ma * 24),
        "ungated_card_days": card_gb / (bytes_per_s * 86400 / 1e9),
    }
