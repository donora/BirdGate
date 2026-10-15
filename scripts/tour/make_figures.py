"""Draw the tour's three diagrams as hand-written SVG, light and dark.

    python scripts/tour/make_figures.py      # -> docs/tour/{flow,scoreboard,scale}.svg

Every number is in the tables below with where it comes from, so a changed
result is a one-line edit here rather than a redraw.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "tour"

# Descriptions only. Every number on the scoreboard comes from docs/tour/scoreboard.json,
# which scripts/tour/scoreboard_numbers.py computes from the shipped demo scores.
ARMS = [
    dict(name="mfcc", label="MFCC + linear", what="FFT, mel filterbank, 20 MFCCs",
         size="164 B", ships=True, note="best on quiet species"),
    dict(name="bands", label="Bandpass + linear", what="40 independent bandpass filters",
         size="172 B", ships=False, note="5× the compute, erratic"),
    dict(name="carihc", label="Cochlea + linear", what="40 cascaded resonators, after CARFAC",
         size="172 B", ships=False, note="last on every metric"),
    dict(name="cnn-int8", label="Tiny CNN", what="int8 convolutional net over log-mel",
         size="5.9 KB", ships=False, note="best detector, loses quiet species"),
]

# Bytes. The BirdNET and Perch figures are the model files as installed by the
# `birdnet` package (acoustic v2.4 fp32 .tflite; perch-v2-cpu SavedModel).
SCALE = [
    ("MFCC, bandpass, cochlea", 168, True),
    ("Tiny CNN", 5_865, True),
    ("BirdNET 2.4", 49e6, False),
    ("Perch v2", 390e6, False),
]
LIMITS = [("AudioMoth SRAM, 32 KB", 32 * 1024), ("AudioMoth flash, 256 KB", 256 * 1024)]

STYLE = """
<style>
  .bg { fill: #fcfcfb; }
  .ink { fill: #0b0b0b; }
  .ink2 { fill: #52514e; }
  .ink3 { fill: #8a8984; }
  .line { stroke: #c9c8c3; }
  .box { fill: #ffffff; stroke: #c9c8c3; }
  .acc { fill: #1baf7a; }
  .accs { stroke: #1baf7a; }
  .acct { fill: #0e7a54; }
  .accbg { fill: #e6f6ef; stroke: #1baf7a; }
  .mute { fill: #b9b8b2; }
  .limit { stroke: #e34948; }
  .limitt { fill: #c23a39; }
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; }
    .ink { fill: #f0efec; }
    .ink2 { fill: #c3c2b7; }
    .ink3 { fill: #8f8e86; }
    .line { stroke: #4a4945; }
    .box { fill: #232322; stroke: #4a4945; }
    .acc { fill: #199e70; }
    .accs { stroke: #199e70; }
    .acct { fill: #4cc99a; }
    .accbg { fill: #17352a; stroke: #199e70; }
    .mute { fill: #5e5d58; }
    .limit { stroke: #e66767; }
    .limitt { fill: #e66767; }
  }
</style>"""


def svg(w: int, h: int, body: str, title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" role="img" aria-label="{title}"><title>{title}</title>{STYLE}'
            f'<rect class="bg" width="{w}" height="{h}" rx="12"/>{body}</svg>')


def text(x, y, s, cls="ink", size=13, anchor="start", weight=400, mono=False):
    cls = cls + (" mono" if mono else "")
    return (f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def flow() -> str:
    """One second of audio, left to right, with what each stage costs on the chip."""
    stages = [
        ("Microphone", ["32,000 samples/s,", "a second at a time"], "4.1 KB sample buffer"),
        ("Features", ["FFT → mel → MFCCs,", "40 numbers a second"], "2 KB spectrum"),
        ("Tiny model", ["40 weights + 1 bias", "→ one score"], "164 bytes"),
        ("Threshold", ["score minus recent", "background (CFAR)"], "2.4 KB of history"),
        ("Gate", ["pre-roll, hangover,", "≥ 20 s per file"], "a few bytes of state"),
    ]
    w, h, bw, bh, top = 1000, 250, 140, 116, 62
    gap = (w - 40 - bw * len(stages) - 120) / len(stages)
    out = [text(20, 34, "One second of audio, every second", size=17, weight=600),
           text(w - 20, 34, "on a 48 MHz Cortex-M4F, 32 KB RAM", cls="ink2", size=13, anchor="end")]
    x = 20
    for i, (name, lines, cost) in enumerate(stages):
        cls = "accbg" if name == "Tiny model" else "box"
        out.append(f'<rect class="{cls}" x="{x}" y="{top}" width="{bw}" height="{bh}" rx="10" stroke-width="1.5"/>')
        out.append(text(x + 14, top + 28, name, size=15, weight=600))
        for j, ln in enumerate(lines):
            out.append(text(x + 14, top + 54 + j * 19, ln, cls="ink2", size=12.5))
        out.append(text(x + bw / 2, top + bh + 26, cost, cls="acct", size=12.5, anchor="middle", weight=600))
        nx = x + bw + gap
        ax0, ax1, ay = x + bw + 6, nx - 6, top + bh / 2
        out.append(f'<line class="line" x1="{ax0}" y1="{ay}" x2="{ax1 - 6}" y2="{ay}" stroke-width="2"/>')
        out.append(f'<path class="ink3" d="M{ax1} {ay} l-9 -5 v10 z"/>')
        x = nx
    # the decision
    cx = x
    out.append(f'<rect class="accbg" x="{cx}" y="{top}" width="120" height="52" rx="10" stroke-width="1.5"/>')
    out.append(text(cx + 60, top + 22, "Bird?", size=14, anchor="middle", weight=600))
    out.append(text(cx + 60, top + 40, "write to SD card", cls="acct", size=12, anchor="middle", weight=600))
    out.append(f'<rect class="box" x="{cx}" y="{top + 64}" width="120" height="52" rx="10" stroke-width="1.5"/>')
    out.append(text(cx + 60, top + 86, "No bird?", size=14, anchor="middle", weight=600))
    out.append(text(cx + 60, top + 104, "skip, write nothing", cls="ink2", size=12, anchor="middle"))
    out.append(text(20, h - 16, "The model is the smallest part of the system. The threshold and the gate decide "
                    "whether it can be trusted in the field.", cls="ink2", size=12.5))
    return svg(w, h, "".join(out), "BirdGate pipeline: microphone, features, tiny model, adaptive threshold, gate, SD card")


def scoreboard() -> str:
    """Four cards, each split into what the arm got right and what it got wrong: the
    share of skippable silence it skipped (one dot per recording, to show how steady
    it is) and, in red, the birdsong it threw away."""
    nums = json.loads((OUT / "scoreboard.json").read_text())
    ceil = nums["ceiling_by_recording"]
    w, h, pad, cw, ch, top = 1000, 446, 20, 226, 322, 52
    gap = (w - 2 * pad - 4 * cw) / 3
    out = [text(pad, 34, "The four approaches, same data, same gate", size=17, weight=600),
           text(w - pad, 34, "six held-out recordings, 98% of birdsong kept overall", cls="ink2",
                size=13, anchor="end")]
    for i, a in enumerate(ARMS):
        n = nums["arms"][a["name"]]
        x = pad + i * (cw + gap)
        cls = "accbg" if a["ships"] else "box"
        out.append(f'<rect class="{cls}" x="{x}" y="{top}" width="{cw}" height="{ch}" rx="12" '
                   f'stroke-width="{2 if a["ships"] else 1.2}"/>')
        out.append(text(x + 16, top + 30, a["label"], size=16, weight=600))
        out.append(text(x + 16, top + 50, a["what"], cls="ink2", size=11.5))
        out.append(text(x + 16, top + 98, f'{100 * n["silence_skipped"]:.0f}%', size=34, weight=600))
        out.append(text(x + 16, top + 118, "of the skippable silence skipped", cls="ink2", size=12))
        # one dot per recording on a 0-100% strip
        sx0, sx1, sy = x + 16, x + cw - 16, top + 144
        X = lambda v: sx0 + v * (sx1 - sx0)  # noqa: E731
        out.append(f'<line class="line" x1="{sx0}" y1="{sy}" x2="{sx1}" y2="{sy}" stroke-width="1.5"/>')
        for v in (0.0, 0.5, 1.0):
            out.append(f'<line class="line" x1="{X(v)}" y1="{sy - 5}" x2="{X(v)}" y2="{sy + 5}" stroke-width="1"/>')
        for v in n["silence_skipped_by_recording"].values():
            dot = "acc" if a["ships"] else "ink2"
            out.append(f'<circle class="{dot}" cx="{X(v):.1f}" cy="{sy}" r="4.5" opacity="0.85"/>')
        vals = n["silence_skipped_by_recording"].values()
        out.append(text(sx0, sy + 20, f"{100 * min(vals):.0f}–{100 * max(vals):.0f}% by recording",
                        cls="ink3", size=11))
        out.append(text(sx1, sy + 20, "100%", cls="ink3", size=11, anchor="end"))
        # the error, in red
        worst = max(n["birdsong_missed_by_recording"].values())
        out.append(text(x + 16, top + 196, f'{100 * n["birdsong_missed"]:.1f}% of birdsong missed',
                        cls="limitt", size=13, weight=600))
        out.append(text(x + 16, top + 214, f"{100 * worst:.1f}% on the worst recording",
                        cls="limitt", size=11.5))
        rows = [("card saved", f'{100 * n["saving"]:.1f}%'),
                ("worst species kept", f'{n["species_floor"]:.1f}%'),
                ("model size", a["size"])]
        for j, (k, v) in enumerate(rows):
            y = top + 214 + 24 + j * 19
            out.append(text(x + 16, y, k, cls="ink2", size=12))
            out.append(text(x + cw - 16, y, v, size=12, anchor="end", weight=600, mono=True))
        if a["ships"]:
            out.append(f'<rect class="acc" x="{x + 16}" y="{top + ch - 30}" width="62" height="22" rx="11"/>')
            out.append(text(x + 47, top + ch - 15, "SHIPS", cls="bg", size=11.5, anchor="middle", weight=700))
            out.append(text(x + 86, top + ch - 15, a["note"], cls="ink2", size=11.5))
        else:
            out.append(text(x + 16, top + ch - 15, a["note"], cls="ink2", size=11.5))
    out.append(text(pad, h - 50, "Skippable silence: what a perfect detector skips behind the same gate, "
                    f"{100 * min(ceil.values()):.0f}–{100 * max(ceil.values()):.0f}% of each hour "
                    "depending on how much of it is birdsong. Only birdsong-free seconds count.",
                    cls="ink3", size=11.5))
    out.append(text(pad, h - 32, "Birdsong missed: seconds of annotated song not written to the card. "
                    "The threshold holds it near 2% overall; the worst recording is where arms differ.",
                    cls="ink3", size=11.5))
    out.append(text(pad, h - 14, "Worst species: lowest retention among species with enough calls to "
                    "measure. CNN: one training seed.", cls="ink3", size=11.5))
    return svg(w, h, "".join(out), "Scoreboard of the four gate approaches")


def scale() -> str:
    """Model size on a log axis, with the AudioMoth's memory as the wall."""
    w, h, x0, x1, ay = 1000, 250, 40, 960, 150
    lo, hi = 1e2, 1e9
    X = lambda b: x0 + (math.log10(b) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
    out = [text(20, 34, "How big is the model?", size=17, weight=600),
           text(w - 20, 34, "file size, log scale", cls="ink2", size=13, anchor="end")]
    # device limits as a shaded "fits on the chip" region
    out.append(f'<rect class="accbg" x="{x0}" y="{ay - 70}" width="{X(256 * 1024) - x0}" height="70" '
               f'rx="6" stroke-width="0" opacity="0.6"/>')
    out.append(text(x0 + 8, ay - 54, "fits on an AudioMoth", cls="acct", size=12, weight=600))
    for label, b in LIMITS:
        out.append(f'<line class="limit" x1="{X(b)}" y1="{ay - 70}" x2="{X(b)}" y2="{ay + 6}" '
                   f'stroke-width="1.5" stroke-dasharray="4 3"/>')
    out.append(text(X(256 * 1024) + 6, ay - 54, "256 KB flash", cls="limitt", size=11.5))
    out.append(text(X(32 * 1024) - 6, ay - 54, "32 KB RAM", cls="limitt", size=11.5, anchor="end"))
    # axis
    out.append(f'<line class="line" x1="{x0}" y1="{ay}" x2="{x1}" y2="{ay}" stroke-width="1.5"/>')
    for e, lab in [(2, "100 B"), (3, "1 KB"), (4, "10 KB"), (5, "100 KB"), (6, "1 MB"),
                   (7, "10 MB"), (8, "100 MB"), (9, "1 GB")]:
        x = X(10 ** e)
        out.append(f'<line class="line" x1="{x}" y1="{ay}" x2="{x}" y2="{ay + 5}" stroke-width="1"/>')
        out.append(text(x, ay + 22, lab, cls="ink3", size=11.5, anchor="middle"))
    for label, b, ours in SCALE:
        x = X(b)
        cls = "acc" if ours else "mute"
        out.append(f'<circle class="{cls}" cx="{x}" cy="{ay}" r="7"/>')
        size = f"{b:,.0f} B" if b < 1e4 else (f"{b / 1e6:.0f} MB")
        if label.startswith("MFCC"):
            size = "164–172 B"
        anchor, tx = ("start", x - 12) if x < x0 + 80 else ("middle", x)
        out.append(text(tx, ay + 48, label, size=12.5, anchor=anchor, weight=600))
        out.append(text(tx, ay + 65, size, cls="ink2", size=11.5, anchor=anchor, mono=True))
    out.append(text(20, h - 12, "The big models identify hundreds of species on a server. The gate answers one "
                    "question on the recorder, in about a two-millionth of Perch's size.", cls="ink2", size=12.5))
    return svg(w, h, "".join(out), "Model sizes on a log scale: BirdGate models against BirdNET and Perch")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, f in [("flow", flow), ("scoreboard", scoreboard), ("scale", scale)]:
        (OUT / f"{name}.svg").write_text(f())
        print(f"docs/tour/{name}.svg")


if __name__ == "__main__":
    main()
