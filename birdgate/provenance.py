"""Run manifests — so any number can be traced back to what produced it.

This exists because the deliverable is not a model, it is **a device that will be
left in a forest for months and whose recordings become the primary data**. If the
gate on a recovered SD card cannot be identified — which weights, which threshold,
which training data — then the audio it kept is of unknown provenance, and so is
every ecological conclusion drawn from it. Worse, the audio it *discarded* cannot
be re-examined. Traceability is not hygiene here; it is the difference between a
dataset and an anecdote.

So every run writes `runs/<name>/manifest.json` capturing:

- the full resolved config (not the CLI args — the *resolved* values)
- a stable hash of that config, used as the run's identity
- the split seed and the exact list of recordings in each split
- checksums of the label files the run consumed
- versions of every package that can change a number
- the metrics achieved, so the manifest alone answers "was this the good one?"

`export.py` copies the config hash into `model_params.h`, so a `.bin` recovered
from a device six months later resolves back to this file.

⚠️ Callers must pass the config the run actually executed — the gate parameters,
the threshold, the adaptive-threshold parameters, the arm — not a fresh default.
An earlier audit found manifests claiming a minimum-on time of 0 s for tables
produced at 10 s, and the anchor quantile in no manifest at all. `GateConfig` now
carries `adapt`, and every script builds its config from the values it ran with.

⚠️ Commit before any deployment build: a `-dirty` git rev pins nothing.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import data as D

TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "librosa", "soundfile", "torch")


def _to_plain(obj):
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def config_hash(cfg) -> str:
    """Stable 12-char identity for a config. Order-independent, float-exact."""
    blob = json.dumps(_to_plain(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def file_digest(path: Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_rev() -> str | None:
    """The commit that produced this run, plus a `-dirty` marker if it is not clean.

    ⚠️ This used to ask `D.REPO` — **BirdSets/**, the shared parent, which is not a
    repository — and so recorded `None` for every run while `birdgate/` itself was
    perfectly well version-controlled one directory down. docs/lab-notebook.md §11 called runs
    "not yet fully reproducible" on the strength of that `None`. They were; the
    provenance was looking in the wrong place. Ask `D.PROJECT`.

    A dirty tree is reported rather than hidden: a manifest that names a commit the
    code did not actually match is worse than one that admits it does not know.
    """
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=D.PROJECT,
                             capture_output=True, text=True, timeout=5)
        if rev.returncode != 0 or not rev.stdout.strip():
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=D.PROJECT,
                               capture_output=True, text=True, timeout=5)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def manifest(cfg, split_df=None, metrics: dict | None = None, notes: str = "") -> dict:
    """Everything needed to reproduce, or to audit, a run."""
    if split_df is None:
        split_df = D.make_split(seed=getattr(cfg, "seed", 0))
    return {
        "run": getattr(cfg, "name", "unnamed"),
        "config_hash": config_hash(cfg),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": _to_plain(cfg),
        "split": {s: sorted(g["filename"]) for s, g in split_df.groupby("split")},
        "split_seed": getattr(cfg, "seed", 0),
        "data_digests": {
            "annotations.csv": file_digest(D.DATA / "annotations.csv"),
            "species.csv": file_digest(D.DATA / "species.csv"),
        },
        "code": {"git_rev": git_rev(), "python": platform.python_version(),
                 "platform": platform.platform()},
        "packages": package_versions(),
        "metrics": {k: v for k, v in (metrics or {}).items() if not k.startswith("_")},
        "notes": notes,
    }


def write(cfg, split_df=None, metrics=None, notes="", run_dir: Path | None = None) -> Path:
    m = manifest(cfg, split_df, metrics, notes)
    run_dir = Path(run_dir) if run_dir else D.RUNS / m["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(m, indent=2, default=str))
    return out
