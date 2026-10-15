"""The dataset: metadata loading, file resolution, and the recording table.

BirdGate owns no audio. `data/` holds the two label files and an `audio/` link to
wherever the FLACs live (a USB drive, here). Everything downstream reads through
this module, so the layout is defined in exactly one place:

    data/
    ├── annotations.csv      one row per annotation box
    ├── species.csv          eBird code -> names
    ├── recordings.csv       filename, sample rate, duration — so nothing needs the
    │                        audio mounted except feature extraction itself
    └── audio/               -> the FLAC directory (symlink)

The dataset is the Sierra Nevada soundscapes (2018, Lassen and Plumas National
Forests, California): 33 one-hour recordings at 32 kHz with 20,147 strong
time-frequency annotations. One of the evaluation sets of the BirdSet benchmark
(Rauch et al., 2024); https://zenodo.org/records/7050014.

The recording-level split (`splits.py`) is the same algorithm and seed as an
earlier, unpublished species classifier on this dataset, so the two are graded on
the same six held-out recordings. `tests/test_split.py` pins it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent   # birdgate/
DATA = PROJECT / "data"
AUDIO_DIR = DATA / "audio"
CACHE = PROJECT / "cache"
RUNS = PROJECT / "runs"
# Frozen scores, linker maps and short clips, so a fresh clone runs the app without
# the audio or the cache. Built by `scripts.build_demo`.
DEMO = PROJECT / "demo"

# The dataset's native sample rate, and the device's. There is no resample step.
NATIVE_SR = 32_000


def audio_path(filename: str) -> Path:
    """Resolve an annotation's filename to the FLAC on disk."""
    return AUDIO_DIR / filename


def audio_available() -> bool:
    """True when the full recordings are reachable (not just the demo clips)."""
    return AUDIO_DIR.resolve().exists()


def assert_audio_available() -> None:
    """Fail loudly with a clear message if the audio is not reachable."""
    if not AUDIO_DIR.resolve().exists():
        raise FileNotFoundError(
            f"Audio not found: {AUDIO_DIR} does not resolve. Mount the drive or point "
            "data/audio at the directory holding the SNE_*.flac files."
        )


def load_species() -> pd.DataFrame:
    """eBird code -> scientific and common name."""
    df = pd.read_csv(DATA / "species.csv")
    return df.rename(columns={
        "Species eBird Code": "code",
        "Scientific Name": "scientific",
        "Common Name": "common",
    })


def load_annotations() -> pd.DataFrame:
    """Every annotation box, joined to species names, with tidy column names.

    Each row is one box: a call (or run of calls) of one species, localised in
    time (start/end seconds) and frequency (low/high Hz).
    """
    df = pd.read_csv(DATA / "annotations.csv").rename(columns={
        "Filename": "filename",
        "Start Time (s)": "start_s",
        "End Time (s)": "end_s",
        "Low Freq (Hz)": "low_hz",
        "High Freq (Hz)": "high_hz",
        "Species eBird Code": "code",
    })
    df["duration_s"] = df["end_s"] - df["start_s"]
    df = df.merge(load_species(), on="code", how="left")
    df["label"] = df["common"] + " (" + df["code"] + ")"
    return df


def species_codes() -> list[str]:
    """The eBird codes in a fixed, sorted order — the label vocabulary."""
    return sorted(load_species()["code"])


# Recordings were made in two daily blocks: a dawn chorus and an evening session.
# The split stratifies on this, because bird activity differs hugely between them.
FILENAME_RE = re.compile(r"^SNE_(\d+)_(\d{8})_(\d{2})(\d{2})(\d{2})\.flac$")


def _row_from_filename(filename: str, sr: int, duration_s: float) -> dict:
    m = FILENAME_RE.match(filename)
    if m is None:
        raise ValueError(f"Unexpected filename: {filename}")
    rec_id, date, hh, mm, ss = m.groups()
    return {
        "filename": filename,
        "rec_id": int(rec_id),
        "date": date,
        "hour": int(hh),
        "start_local": f"{hh}:{mm}:{ss}",
        "sr": int(sr),
        "duration_s": float(duration_s),
        # 04:00-09:59 is the dawn chorus; 17:00-19:59 the evening session.
        "block": "dawn" if int(hh) < 12 else "evening",
    }


def load_recordings() -> pd.DataFrame:
    """One row per recording: id, date, start hour, duration, block.

    Reads the FLAC headers when the audio is mounted (`sf.info` reads no samples,
    so it is fast and authoritative) and falls back to `data/recordings.csv`
    otherwise, so training, evaluation and the app work entirely off the feature
    cache. `write_recordings_csv` refreshes that file from the headers.
    """
    if AUDIO_DIR.resolve().exists():
        import soundfile as sf
        rows = []
        for path in sorted(AUDIO_DIR.glob("*.flac")):
            info = sf.info(str(path))
            rows.append(_row_from_filename(path.name, info.samplerate,
                                           info.frames / info.samplerate))
        if rows:
            return pd.DataFrame(rows)

    csv = DATA / "recordings.csv"
    if csv.exists():
        t = pd.read_csv(csv)
        return pd.DataFrame([_row_from_filename(r.filename, r.sr, r.duration_s)
                             for r in t.itertuples()])

    assert_audio_available()   # raises with the mount message
    raise FileNotFoundError(f"{AUDIO_DIR} holds no .flac files and {csv} is missing")


def write_recordings_csv() -> Path:
    """Snapshot the recording table from the FLAC headers into `data/recordings.csv`."""
    assert_audio_available()
    out = DATA / "recordings.csv"
    load_recordings()[["filename", "sr", "duration_s"]].to_csv(out, index=False)
    return out


def make_split(*args, **kwargs):
    """`splits.make_split`, re-exported so callers have one module to import."""
    from .splits import make_split as _make_split
    return _make_split(*args, **kwargs)


__all__ = [
    "PROJECT", "DATA", "AUDIO_DIR", "CACHE", "RUNS", "DEMO", "NATIVE_SR",
    "assert_audio_available", "audio_available", "audio_path", "load_annotations", "load_recordings",
    "load_species", "species_codes", "make_split", "write_recordings_csv",
]
