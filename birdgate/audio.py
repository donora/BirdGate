"""Clip loading and a display spectrogram, for the app.

Nothing in training or evaluation uses this; the arms compute their own features
in `frontends.py`. It exists so the app can play a clip and draw a log-mel picture
of it without loading a whole one-hour file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data as D
from .data import NATIVE_SR


@dataclass(frozen=True)
class MelParams:
    """Display spectrogram knobs. 1024 @ 32 kHz is ~32 ms per frame; hop 256 is 8 ms."""
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: int = 150
    fmax: int = 14_000
    sr: int = NATIVE_SR


def load_clip(path: str | Path, start_s: float, end_s: float, pad_s: float = 1.0,
              sr: int = NATIVE_SR) -> tuple[np.ndarray, int, float, float]:
    """Read just `[start_s - pad_s, end_s + pad_s]` from a large FLAC.

    Returns (samples, sr, clip_start_s, clip_end_s), the last two being the absolute
    times in the recording the returned audio covers.
    """
    import librosa
    import soundfile as sf
    path = str(path)
    total = sf.info(path).duration
    lo = max(0.0, start_s - pad_s)
    hi = min(total, end_s + pad_s)
    y, _ = librosa.load(path, sr=sr, offset=lo, duration=hi - lo, mono=True)
    return y, sr, lo, hi


def clip_index():
    """The demo clips (`demo/clips/clips.csv`): one row per 30 s excerpt, with the
    recording and absolute times it covers and why it was chosen. Empty if absent."""
    import pandas as pd
    f = D.DEMO / "clips" / "clips.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(
        columns=["clip", "filename", "start_s", "end_s", "why"])


def load_span(filename: str, start_s: float, end_s: float, pad_s: float = 1.0,
              sr: int = NATIVE_SR) -> tuple[np.ndarray, int, float, float]:
    """`load_clip` by recording name, from the full audio or else a demo clip.

    Times stay absolute (seconds into the hour-long recording) either way, so a
    caller cannot tell the difference. Without the full audio the span must lie
    inside one clip; the padding is trimmed to the clip's edges."""
    if D.audio_available():
        return load_clip(D.audio_path(filename), start_s, end_s, pad_s, sr)
    clips = clip_index()
    hit = clips[(clips["filename"] == filename) & (clips["start_s"] <= start_s)
                & (clips["end_s"] >= end_s)]
    if not len(hit):
        raise FileNotFoundError(f"{filename} {start_s:.0f}-{end_s:.0f} s is in no demo clip")
    c = hit.iloc[0]
    lo = max(float(c["start_s"]), start_s - pad_s)
    hi = min(float(c["end_s"]), end_s + pad_s)
    y, _, _, _ = load_clip(D.DEMO / "clips" / c["clip"], lo - c["start_s"],
                           hi - c["start_s"], pad_s=0.0, sr=sr)
    return y, sr, lo, hi


def mel_spectrogram(y: np.ndarray, p: MelParams = MelParams()) -> np.ndarray:
    """Waveform -> log-mel in dB relative to the clip's peak. Shape (n_mels, n_frames)."""
    import librosa
    power = librosa.feature.melspectrogram(y=y, sr=p.sr, n_fft=p.n_fft,
                                           hop_length=p.hop_length, n_mels=p.n_mels,
                                           fmin=p.fmin, fmax=p.fmax, power=2.0)
    return librosa.power_to_db(power, ref=np.max)
