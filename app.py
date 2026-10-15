"""BirdGate explorer — the problem, the splits, and the oracle gate.

Run:  streamlit run app.py

The point of this app is to make the gate *legible before it exists*. Every tab
shows something you can check by eye against the audio, because the failure mode
of a recording gate is silent: audio it wrongly discards is gone, and no later
analysis can detect that it was ever there.

Tab 1 (Mission):   what is being built and for where — the Highland deployment.
Tab 2 (Splits):    which recordings train, val, test — a recording-level split
                   pinned by a test so every number uses the same holdout — plus the
                   run manifest that makes any number here traceable.
Tab 3 (Occupancy): what silence actually looks like in this data. Per-recording
                   occupancy, the silent-gap distribution, and why the gap
                   distribution (not total silence) decides what a gate can save.
Tab 4 (Oracle):    a gate driven by ground truth. Sets the ceiling and separates
                   state-machine cost from model cost for the rest of the project.
Tab 5 (Inspect):   a recording, second by second — spectrogram, annotation boxes,
                   the gate's decision, and the audio. Listen to what gets kept
                   and what gets thrown away.
Tab 7 (Silicon):   the four resources of the part — SRAM, flash, the sample ring,
                   core time — each drawn as the rectangle it is and partitioned
                   by the function that spends it. Three of the four are read out
                   of the linker map of a real device build, so the tiles are
                   allocations rather than estimates; the compute panel is a
                   model, and the panel says by how much the device disagreed
                   with it.
"""
from __future__ import annotations

import textwrap

import matplotlib
from dataclasses import replace as dc_replace

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from birdgate import adapt as A  # noqa: E402
from birdgate import budget as BG  # noqa: E402
from birdgate import carfac as CF  # noqa: E402
from birdgate import data as D  # noqa: E402
from birdgate import frontends as FE  # noqa: E402
from birdgate import gate as G  # noqa: E402
from birdgate import intervals as I  # noqa: E402
from birdgate import metrics as M  # noqa: E402
from birdgate import provenance as P  # noqa: E402
from birdgate import svm as S  # noqa: E402
from birdgate import windows as W  # noqa: E402
from birdgate.config import (  # noqa: E402
    CarIhcParams, FrontEndParams, GateConfig, GateParams, WindowParams,
)

st.set_page_config(page_title="BirdGate", layout="wide")


def _require_fresh_modules() -> None:
    """Fail with an instruction, not a traceback, when the server holds stale code.

    Streamlit re-executes `app.py` on every rerun but does not reliably re-import
    the `birdgate` package underneath it, so a server started before a code change
    keeps the old class definitions. The symptom is an `unexpected keyword
    argument` or `has no attribute` deep inside a module that is demonstrably
    correct on disk — confusing enough that it has now cost two debugging rounds.

    `watchdog` (a Streamlit dependency for proper file watching) makes this mostly
    go away; this guard covers the rest. Extend `EXPECT` whenever a dataclass
    field is added that the app depends on.
    """
    import dataclasses

    from birdgate.adapt import AdaptParams
    from birdgate.budget import Silicon
    from birdgate.config import CarIhcParams, GateParams

    EXPECT = {
        Silicon: {"core_hz", "flash_bytes", "sram_bytes", "ext_sram_bytes",
                  "n_ring_slots", "heap_bytes"},
        AdaptParams: {"mode", "window_s", "bg_quantile", "scale", "global_scale",
                      "agg", "agg_n"},
        GateParams: {"theta_on", "pre_roll_s", "hangover_s", "min_on_s", "recheck_period_s"},
        CarIhcParams: {"n_channels", "zero_ratio", "min_zeta", "high_f_damping_compression"},
    }
    missing = {
        cls.__name__: sorted(want - {f.name for f in dataclasses.fields(cls)})
        for cls, want in EXPECT.items()
        if want - {f.name for f in dataclasses.fields(cls)}
    }
    if missing:
        st.error(
            "**This Streamlit server is running stale code.** The files on disk are "
            f"newer than the modules loaded in memory (missing: `{missing}`).\n\n"
            "Stop the server (Ctrl-C) and start it again:\n\n"
            "```bash\nstreamlit run app.py\n```\n\n"
            "Streamlit reloads `app.py` on every rerun but not always the `birdgate` "
            "package under it.",
            icon="🔄",
        )
        st.stop()


_require_fresh_modules()

INK = "#2b2b2b"
HUE = "#1f9e89"      # bird / kept
WARN = "#d1495b"     # dropped
MUTE = "#c9c9c9"     # silence / not recorded


# --------------------------------------------------------------------------- #
# Cached loaders                                                               #
# --------------------------------------------------------------------------- #
@st.cache_data
def get_ann():
    return D.load_annotations()


@st.cache_data
def get_recordings():
    return D.load_recordings()


@st.cache_data
def get_split(seed: int):
    return D.make_split(seed=seed)


@st.cache_data
def get_windows(win_s: float, train_hop: float, eval_hop: float, min_overlap: float, seed: int):
    p = WindowParams(win_s=win_s, train_hop_s=train_hop, eval_hop_s=eval_hop,
                     min_overlap_s=min_overlap)
    return W.build_windows(get_split(seed), get_ann(), p)


@st.cache_data
def get_occupancy():
    return W.occupancy(get_ann(), get_recordings())


@st.cache_data
def get_gaps():
    return W.gap_stats(get_ann(), get_recordings())


def get_frontend(kind: str, n_ch: int, f_lo: float, f_hi: float, zero_ratio: float):
    """Deliberately NOT @st.cache_data.

    Designing the bank costs 0.04 ms (`carihc`) to 5 ms (`bands`), so caching buys
    nothing — and it costs real robustness. `st.cache_data` *serialises* what it
    returns, so a cached `FrontEnd`/`FrontEndParams` from a previous version of the
    code comes back as a live object missing whatever field was added since, and
    surfaces as `AttributeError: 'CarIhcParams' object has no attribute ...` deep
    inside `design()` rather than as a clean cache miss. Only cache plain arrays
    and scalars here; rebuild anything that carries a dataclass.
    """
    p = FrontEndParams(kind=kind, carihc=CarIhcParams(
        n_channels=n_ch, f_lo=f_lo, f_hi=f_hi, zero_ratio=zero_ratio))
    return FE.build(p, 32_000), p


@st.cache_data
def channel_responses(kind, n_ch, f_lo, f_hi, zero_ratio, n_probe: int = 90):
    """Measured — not analytic — per-channel response. Probing the built object
    is the only version that cannot drift from what training actually uses."""
    fe, _ = get_frontend(kind, n_ch, f_lo, f_hi, zero_ratio)
    freqs = np.geomspace(100, 15_000, n_probe)
    # 0.3 s probe, first 0.15 s discarded as filter transient. The two numbers
    # must not collide: slicing at >= the probe length leaves an empty array and
    # every response reads NaN.
    t = np.arange(int(0.3 * 32_000)) / 32_000
    out = np.zeros((n_probe, fe.n_channels))
    for i, f in enumerate(freqs):
        x = np.sin(2 * np.pi * f * t) * 0.5
        if kind == "carihc":
            bm = CF.cascade(x, fe._bank)[:, 4800:]
        else:
            from scipy.signal import sosfilt
            bm = np.stack([sosfilt(fe._sos[c:c + 1], x)[4800:] for c in range(fe.n_channels)])
        out[i] = np.sqrt((bm ** 2).mean(axis=1))
    return freqs, out


@st.cache_data
def get_arm_scores(arm: str, seed: int):
    """Train one arm and return its raw score for EVERY window, plus the windows
    table. Cached on (arm, seed) — returns only a DataFrame and arrays, never a
    dataclass, for the reason documented on `get_frontend`.

    The `cnn*` arms are the exception: they are **not** trained here. A network
    cannot be fitted between slider moves, so `scripts.train_cnn --save-scores`
    writes its raw logits once and this loads them. That is enough, because
    everything the compare tab varies happens *after* the model — see
    `frozen_scores`. A linear arm whose feature cache is absent (a fresh clone,
    no audio) is loaded the same way, from the `demo/` bundle."""
    from scripts.train import build_xy
    if arm.startswith("cnn") or not (D.CACHE / arm).exists():
        return frozen_scores(arm)
    wp = WindowParams()
    windows, y, X = build_xy(arm, wp)
    fe = FE.build(FrontEndParams(kind=arm), 32_000)
    tr = windows.index[windows["split"] == "train"].to_numpy()
    model = S.fit(X[tr], y[tr], arm, fe.feature_names, seed=seed)
    return windows, y, model.score(X), np.asarray(model.w), float(model.b)


def scores_file(arm: str):
    """Where an arm's frozen scores live: the CNN's own training run when it is on
    disk, otherwise the `demo/` bundle that ships with the repository."""
    own = D.RUNS / "g3-cnn" / "scores.npz"
    return own if arm.startswith("cnn") and own.exists() else D.DEMO / "scores.npz"


def cnn_params_note() -> dict | None:
    """The training config behind the cached CNN logits, or None if not exported."""
    import json
    f = scores_file("cnn")
    if not f.exists():
        return None
    with np.load(f, allow_pickle=False) as z:
        return json.loads(str(z["params"]))


def using_demo_scores() -> bool:
    """True when the linear arms come frozen from `demo/` rather than trained live."""
    return not (D.CACHE / "mfcc").exists()


@st.cache_data
def frozen_scores(arm: str):
    """Load frozen raw scores and **verify** they line up with our windows.

    The scores were computed against one `build_windows()` result; this session
    builds its own. If the two ever disagree — a changed window length, overlap
    floor or split seed — the rows would silently mean different seconds of audio,
    and the arm would draw a plausible, wrong lane. So it is asserted, not assumed,
    exactly as `cache.assemble` does for the filterbank arms."""
    f = scores_file(arm)
    if not f.exists():
        raise FileNotFoundError(
            f"No frozen scores at {f}. Build them with:\n"
            "    python -m scripts.train_cnn --save-scores\n"
            "    python -m scripts.build_demo")
    windows, y = W.build_windows(p=WindowParams())
    windows = windows.reset_index(drop=True)
    with np.load(f, allow_pickle=False) as z:
        if len(z["filename"]) != len(windows) or \
                not np.array_equal(z["filename"], windows["filename"].to_numpy().astype("U")) or \
                not np.allclose(z["start_s"], windows["start_s"].to_numpy(), atol=1e-6):
            raise ValueError(
                f"{f} was computed against a different window table — rebuild it "
                "(`scripts.train_cnn --save-scores`, `scripts.build_demo`) or change "
                "the window settings back.")
        key = f"raw_{arm}"
        if key not in z:
            raise KeyError(f"{f} has no scores for {arm!r}; it holds "
                           f"{[k[4:] for k in z.files if k.startswith('raw_')]}")
        raw = z[key].astype(np.float64)
        w = z[f"w_{arm}"] if f"w_{arm}" in z else np.zeros(0)
        b = float(z[f"b_{arm}"]) if f"b_{arm}" in z else 0.0
    return windows, y, raw, w, b


@st.cache_data
def get_oracle_curve(win_s, min_overlap, seed):
    w, y = get_windows(win_s, win_s, win_s, min_overlap, seed)
    ev = w[w["split"] != "train"]
    yy = y[ev.index.to_numpy()]
    return M.oracle_curve(ev.reset_index(drop=True), yy,
                          ann=get_ann(), recordings=get_recordings())


def strip(ax, y0, iv, color, label=None, h=0.8):
    """Draw an interval set as a horizontal strip — the timeline primitive."""
    for s, e in iv:
        ax.add_patch(Rectangle((s, y0), max(e - s, 0.5), h, color=color, lw=0))
    if label:
        ax.text(-0.008, y0 + h / 2, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, color=INK)


# --------------------------------------------------------------------------- #
# Area maps — the primitive for the silicon tab                                #
# --------------------------------------------------------------------------- #
# A budget is a fixed rectangle and the things in it are areas, so the honest
# picture is a partition of that rectangle rather than a bar chart of shares: it
# shows what is left as a *place*, and "what is left" is the whole question on a
# 32 KB part. Bars would need a separate mark for free space and would let the
# pieces not add up; a partition cannot — `tests/test_budget.py` asserts the rows
# tile the part, so the rectangle is always full.
def _tint(hex_colour: str, t: float) -> tuple[float, float, float]:
    """Blend toward white by `t`. One hue per owner, lightness per stage, which
    keeps the number of colours *carrying meaning* at five however many stages
    there are — the stage name is written in the tile."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(c + (1.0 - c) * t for c in (r, g, b))


OWNER_HUE = {
    "BirdGate": "#1f9e89",      # the gate — the same teal as "kept" everywhere else
    "Stock": "#4f6d8c",         # inherited AudioMoth firmware
    "Toolchain": "#8f8577",     # libc, startup, whatever the linker adds
    "Padding": "#b0b0b0",       # alignment
    "Free": "#ffffff",          # unspent — drawn as absence, not as a colour
}
COMPUTE_HUE = {"front-end": "#1f9e89", "model": "#4f6d8c",
               "decision": "#b08442", "idle": "#ffffff"}


def _sq_row(sizes, x, y, dx, dy):
    w = sum(sizes) / dy
    out, yy = [], y
    for s in sizes:
        out.append((x, yy, w, s / w)); yy += s / w
    return out


def _sq_col(sizes, x, y, dx, dy):
    h = sum(sizes) / dx
    out, xx = [], x
    for s in sizes:
        out.append((xx, y, s / h, h)); xx += s / h
    return out


def _sq_lay(sizes, x, y, dx, dy):
    return _sq_row(sizes, x, y, dx, dy) if dx >= dy else _sq_col(sizes, x, y, dx, dy)


def _sq_worst(sizes, x, y, dx, dy):
    return max(max(w / h, h / w) for _, _, w, h in _sq_lay(sizes, x, y, dx, dy)
               if w > 0 and h > 0)


def squarify(sizes, x=0.0, y=0.0, dx=1.0, dy=1.0):
    """Bruls-Huizing-van Wijk squarified treemap: rectangles of near-unit aspect.

    Areas are exact; only the arrangement is heuristic. `sizes` must be descending
    — the algorithm's aspect-ratio guarantee depends on it — so callers sort, and
    colour is keyed to the owner rather than to position, which keeps identity
    stable when a different arm reorders the tiles.
    """
    sizes = [s for s in sizes if s > 0]
    if not sizes:
        return []
    k = dx * dy / sum(sizes)
    scaled = [s * k for s in sizes]
    out = []

    def rec(rest, x, y, dx, dy):
        if not rest:
            return
        if len(rest) == 1 or dx <= 0 or dy <= 0:
            out.extend(_sq_lay(rest, x, y, dx, dy))
            return
        i = 1
        while i < len(rest) and _sq_worst(rest[:i], x, y, dx, dy) >= \
                _sq_worst(rest[:i + 1], x, y, dx, dy):
            i += 1
        out.extend(_sq_lay(rest[:i], x, y, dx, dy))
        covered = sum(rest[:i])
        if dx >= dy:
            w = covered / dy
            rec(rest[i:], x + w, y, dx - w, dy)
        else:
            h = covered / dx
            rec(rest[i:], x, y + h, dx, dy - h)

    rec(scaled, x, y, dx, dy)
    return out


def area_map(ax, df, hue, unit="B", group="owner", part="stage",
             value="bytes", aspect=0.52):
    """Draw a resource as a nested partition: `group` blocks, `part` tiles inside.

    Nesting is what makes it legible — one block per owner, so "how much of the
    part is ours" is answered by looking rather than by adding tiles up. Labels are
    drawn only where they fit with padding; everything else is in the table below
    the figure, which is also the non-colour reading of the same data.
    """
    g = (df.groupby([group, part], as_index=False)[value].sum()
           .sort_values(value, ascending=False))
    tot = g[value].sum()
    blocks = g.groupby(group, as_index=False)[value].sum().sort_values(value, ascending=False)
    W, H = 1.0, aspect
    # Text fitting has to be done in axis units, and the conversion depends on the
    # figure size, so measure it rather than guessing: a label that overflows its
    # tile is worse than no label, and a tile big enough for a label that does not
    # get one sends the reader to the table for nothing.
    fs = 7.4
    # The AXES box, not the figure: `subplots` leaves ~22% of the width in margins,
    # and measuring the figure instead silently makes every label 1.3x wider than
    # the fit test believes — which is exactly how a label ends up hanging over the
    # edge of its tile.
    box = ax.get_position()
    w_in, h_in = ax.figure.get_size_inches()
    char_w = fs * 0.62 / (w_in * box.width * 72 / W)
    line_h = fs * 1.45 / (h_in * box.height * 72 / H)

    for (x, y, dx, dy), owner in zip(squarify(blocks[value], 0, 0, W, H), blocks[group]):
        parts = g[g[group] == owner].sort_values(value, ascending=False)
        n = len(parts)
        for i, ((x2, y2, dx2, dy2), row) in enumerate(
                zip(squarify(parts[value], x, y, dx, dy), parts.itertuples())):
            # Lightest tile first. Area already carries magnitude here, so lightness
            # is only separating neighbours — and the big tiles are the ones that
            # have to hold dark text, so they are the ones that get the white.
            t = 0.78 - 0.42 * (i / max(n - 1, 1))
            # A white hue means "this is not spent" — free flash, free SRAM, an
            # idle core. Drawn as absence with a hairline rather than as a fill,
            # so the eye reads it as the space that is left.
            empty = hue[owner].lower() == "#ffffff"
            face = "#ffffff" if empty else _tint(hue[owner], t)
            # A surface gap rather than a stroke: the gap separates the fills
            # without adding a mark of its own. Free space gets a hairline instead,
            # because with no fill there is nothing for a gap to separate.
            pad = 0.004
            ax.add_patch(Rectangle((x2 + pad, y2 + pad), max(dx2 - 2 * pad, 0),
                                   max(dy2 - 2 * pad, 0), facecolor=face,
                                   edgecolor="#d9d9d9" if empty else "none", lw=0.8))
            v = getattr(row, value)
            label = str(getattr(row, part))
            inner_w, inner_h = dx2 - 3 * pad, dy2 - 3 * pad
            # `break_long_words=False` or a narrow tile splits a word across lines
            # and renders "GPS + sunris/e"; better to fail the fit test and fall
            # through to a shorter rendering.
            wrapped = textwrap.wrap(label, width=max(int(inner_w / char_w), 6),
                                    break_long_words=False, break_on_hyphens=False)
            # Most informative rendering that fits, in order. Dropping the share
            # before dropping the name is deliberate: the name is what makes a tile
            # readable at all, and the share is recoverable from the tile's size.
            for lines in (wrapped + [f"{v:,.0f} {unit} · {100 * v / tot:.1f}%"],
                          wrapped + [f"{v:,.0f} {unit}"],
                          wrapped,
                          [f"{v:,.0f}"]):
                if inner_w > char_w * max(len(l) for l in lines) and \
                        inner_h > line_h * len(lines):
                    ax.text(x2 + dx2 / 2, y2 + dy2 / 2, "\n".join(lines), ha="center",
                            va="center", fontsize=fs, color=INK, linespacing=1.45)
                    break
            # Nothing fit: the tile is a sliver, and the table below carries it.
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # Identity is never colour-alone: most tiles are labelled, and the legend
    # covers the ones too small to be.
    ax.legend(handles=[Rectangle((0, 0), 1, 1,
                                 facecolor="#ffffff" if c.lower() == "#ffffff"
                                 else _tint(c, 0.45),
                                 edgecolor="#d9d9d9" if c.lower() == "#ffffff" else "none",
                                 label=o) for o, c in hue.items()],
              frameon=False, fontsize=8, loc="lower center", ncol=len(hue),
              bbox_to_anchor=(0.5, -0.09), handlelength=1.4, handleheight=1.0)
    return g


def tidy(ax, xlabel=None):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x", color="#ececec", lw=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK)


ann = get_ann()
recs = get_recordings()

st.title("🚪 BirdGate")
st.caption(
    "An always-on birdsong gate for the AudioMoth — record when a bird is singing, "
    "skip when it isn't. Proof of concept on 33 h of Sierra Nevada soundscape; "
    "**developed in collaboration with a rewilding project in the Scottish Highlands.**"
)

(tab_mission, tab_splits, tab_occ, tab_oracle, tab_cochlea, tab_compare,
 tab_silicon) = st.tabs(
    ["🎯 Mission", "🧱 Splits", "🔇 Occupancy & silence", "🚪 Oracle gate",
     "🐌 Cochlea", "🔬 Compare arms", "🔩 Silicon budget"]
)


# --------------------------------------------------------------------------- #
# Tab 1 — Mission                                                              #
# --------------------------------------------------------------------------- #
with tab_mission:
    st.subheader("What is being built")
    st.markdown(
        """
A classifier small enough to run **continuously** on AudioMoth hardware
(ARM Cortex-M4 @ 48 MHz, 256 KB flash, 32 KB internal SRAM) that decides, second
by second, *is a bird singing right now?* — and writes to the SD card only when
the answer is yes.

Method: a ~40-feature front-end, a linear model `f(X) = Σ wₚ·xₚ + b` (one binary
score, ~170 bytes), an adaptive background-relative threshold, and a gate state
machine. Four front-ends were compared under controls — a neuromorphic cochlea, a
plain bandpass bank, MFCCs and a tiny CNN — and MFCC is the arm that ships.

**Sierra Nevada is the test rig, not the target.** It is here because it is 33
hours of continuous soundscape with 20,147 strong time-frequency annotations —
the only way to answer *can continuous recordings train a gate this small?*
without waiting for labelled Highland data. Everything is built so the front-end,
the state machine, the metrics and the export path carry over unchanged when the
Scottish recordings arrive; only the training set and the calibrated threshold
are re-fit.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### The three things that must hold")
        st.markdown(
            "1. **Don't miss birdsong.** Discarded audio is unrecoverable — this "
            "is the only truly irreversible failure in the system.\n"
            "2. **Repeatable.** Same config + same data → same gate, byte for byte.\n"
            "3. **Traceable.** Any recording on any card resolves back to the "
            "exact model, threshold and training set that let it through."
        )
    with c2:
        st.markdown("##### Resource budget")
        st.dataframe(pd.DataFrame([
            {"item": "SRAM", "target": "< 8 KB", "why": "what stock firmware leaves free"},
            {"item": "Flash", "target": "< 100 KB", "why": "headroom above stock in a 240 KB part"},
            {"item": "Weights", "target": "~170 B", "why": "one score, P+2 weights and a bias"},
            {"item": "Sample rate", "target": "32 kS/s", "why": "keeps high thin calls (lab notebook §4)"},
        ]), hide_index=True, width="stretch")
    with c3:
        st.markdown("##### What the gate actually buys")
        life = M.deployment_life(1.0)
        st.markdown(
            f"Ungated at 32 kHz/16-bit mono a 128 GB card holds "
            f"**{life['ungated_card_days']:.0f} days**, but three AA cells last only "
            "about 11–14 at the always-on current, so **the battery runs out "
            "first**.\n\n"
            "The front-end runs on every second regardless; the gate saves SD writes, "
            "card and listening time. At this busy site that is only **+1.6%** of "
            "battery life; at a sparse site (~10% birdsong) the same gate buys "
            "about **+39%**. SUMMARY.md §10 has the energy model."
        )

    st.info(
        "**Status:** the Python study is complete, and both the `mfcc-dev` and `cnn` "
        "arms run live on an AudioMoth as real-time detectors (the green LED follows "
        "each one-second decision). Writing gated audio to the SD card is the next "
        "step; see ROADMAP.md. Tab 4 drives the gate with ground truth, which fixes "
        "the ceiling every model is read against.",
        icon="✅",
    )


# --------------------------------------------------------------------------- #
# Tab 2 — Splits                                                               #
# --------------------------------------------------------------------------- #
with tab_splits:
    seed = st.sidebar.number_input("Split seed", 0, 999, 0, key="seed")
    split = get_split(seed)
    occ = get_occupancy().merge(split[["filename", "split"]], on="filename")

    st.subheader("Recording-level split, stratified by date × time-of-day")
    st.caption(
        "Split by **recording**, never by window: two windows from the same hour "
        "share the same individual birds, mic and weather, so a window-level split "
        "measures memorisation. This is `birdgate.splits.make_split`, stratified by "
        "dawn and evening and pinned by `tests/test_split.py`, so every number in this "
        "app is graded on the *same six* held-out recordings."
    )

    c1, c2, c3 = st.columns(3)
    for col, s in zip((c1, c2, c3), ("train", "val", "test")):
        g = occ[occ["split"] == s]
        col.metric(f"{s}", f"{len(g)} recordings",
                   f"{g['occupancy_pct'].mean():.0f}% mean occupancy")

    st.dataframe(
        occ.sort_values(["split", "filename"])[
            ["filename", "split", "date", "hour", "block", "n_boxes", "n_species",
             "occupancy_pct"]
        ].round(1),
        hide_index=True, width="stretch", height=380,
    )

    st.subheader("Traceability — the run manifest")
    st.caption(
        "Written to `runs/<name>/manifest.json` by `birdgate.provenance`. A `.bin` "
        "recovered from a device six months later resolves back through its config "
        "hash to exactly this record: which recordings trained it, which label file "
        "(by checksum), which package versions."
    )
    man = P.manifest(GateConfig(seed=seed), split_df=split)
    rev = man["code"]["git_rev"]
    if rev is None:
        st.warning(
            "⚠️ **`git_rev` is None — this repo is not under version control.** The "
            "config is pinned but the *code* that consumed it is not, so a run is "
            "not yet fully reproducible. `git init` + commit before any deployment "
            "build.", icon="⚠️",
        )
    elif rev.endswith("-dirty"):
        st.warning(
            "⚠️ **The working tree is dirty.** The manifest names a commit, but the "
            "code that produced this run is not that commit — there are uncommitted "
            "changes. Fine while iterating; commit before any deployment build, or "
            "the `.bin` on a recovered card resolves to source that never ran.",
            icon="⚠️",
        )
    st.json(man, expanded=False)


# --------------------------------------------------------------------------- #
# Tab 3 — Occupancy & silence                                                  #
# --------------------------------------------------------------------------- #
with tab_occ:
    occ = get_occupancy().merge(get_split(st.session_state.get("seed", 0))[
        ["filename", "split"]], on="filename")
    gaps = get_gaps()

    st.subheader("How much of this data is actually bird?")
    c1, c2, c3, c4 = st.columns(4)
    tot_bird = occ["bird_s"].sum()
    tot_wall = occ["duration_s"].sum()
    c1.metric("Wall clock", f"{tot_wall/3600:.0f} h")
    c2.metric("Bird-seconds (union)", f"{tot_bird/3600:.1f} h", f"{100*tot_bird/tot_wall:.1f}%")
    c3.metric("Silent gaps", f"{len(gaps):,}")
    c4.metric("Median gap", f"{np.median(gaps):.1f} s")

    st.caption(
        "**Occupancy varies enormously by recording — that is the useful finding.** "
        f"From {occ['occupancy_pct'].min():.1f}% to {occ['occupancy_pct'].max():.1f}%. "
        "The near-empty 04:00 pre-dawn recordings are the closest thing this dataset "
        "has to a realistic deployment prior, and they are where the threshold should "
        "be calibrated — not on the dawn-chorus hours that dominate the average."
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    o = occ.sort_values("occupancy_pct")
    colors = {"train": HUE, "val": "#f4a259", "test": WARN}
    ax.barh(range(len(o)), o["occupancy_pct"], color=[colors[s] for s in o["split"]], height=0.75)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([f"{f[4:7]} · {h:02d}:00" for f, h in zip(o["filename"], o["hour"])],
                       fontsize=7)
    tidy(ax, "bird occupancy (% of the hour)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=s) for s, c in colors.items()],
              frameon=False, loc="lower right")
    st.pyplot(fig); plt.close(fig)

    st.subheader("The silent-gap distribution decides what a gate can save")
    st.caption(
        "Total silence is not the budget — the *shape* of it is. A 2 s hangover "
        "reclaims nothing from a gap shorter than 2 s, however many hours of such "
        "gaps exist. Here the median gap is only "
        f"{np.median(gaps):.1f} s, so most of the silence is unreachable by any "
        "gate that keeps calls intact. Highland recordings, especially at night and "
        "off-season, will have a very different and far more favourable shape."
    )
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(np.clip(gaps, 0, 30), bins=60, color=MUTE, edgecolor="none")
        tidy(ax, "silent gap length (s, clipped at 30)")
        ax.set_ylabel("count", color=INK)
        st.pyplot(fig); plt.close(fig)
    with c2:
        ts = np.arange(0, 20.5, 0.5)
        reach = [gaps[gaps > t].sum() / tot_wall * 100 for t in ts]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(ts, reach, color=HUE, lw=2)
        tidy(ax, "hangover + pre-roll consumed (s)")
        ax.set_ylabel("% of wall clock still reclaimable", color=INK)
        st.pyplot(fig); plt.close(fig)


# --------------------------------------------------------------------------- #
# Tab 4 — Oracle gate                                                          #
# --------------------------------------------------------------------------- #
with tab_oracle:
    st.subheader("A gate driven by ground truth — the ceiling, and the control")
    st.markdown(
        "This runs the **real state machine** (`birdgate.gate`, mirrored in "
        "`firmware/common/gate.c`) on perfect scores. Two uses:\n\n"
        "- **Ceiling.** No model can beat these numbers with these settings.\n"
        "- **Control.** Later, a model at 96% retention where the oracle reaches "
        "99.7% has a 3.7% *model* deficit; at 96% where the oracle reaches 96.5% "
        "the loss is the *state machine's*. Without this table the two are "
        "indistinguishable, and the wrong thing gets optimised."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    win_s = c1.select_slider("Decision window (s)", [0.5, 1.0, 2.0], value=1.0)
    pre_roll = c2.select_slider("Pre-roll (s)", [0.0, 0.5, 1.0, 2.0, 3.0], value=1.0,
                                key="pre_roll")
    hangover = c3.select_slider("Hangover (s)", [0.0, 1.0, 2.0, 3.0, 5.0], value=1.0,
                                key="hangover")
    min_on = c4.select_slider("Min recording (s)", [0.0, 5.0, 10.0, 20.0, 30.0], value=20.0,
                              key="min_on")
    recheck = c5.select_slider("Recheck every (s)", [1.0, 2.0, 5.0, 10.0], value=1.0,
                               key="recheck")
    st.caption(
        "**Pre-roll** is capped by the ring buffer: 8 × 32 KB = 256 KB is 8 s at 16 kHz but "
        "only **4.1 s at our 32 kHz**, so ~3 s is the real ceiling. **Min recording** holds "
        "the gate open regardless of score — its justification is filesystem and listener "
        "ergonomics, not acoustics. It defaults to **20 s to match the model arms**, so the "
        "oracle row is a like-for-like ceiling; docs/lab-notebook.md §6's published oracle numbers were "
        "measured at 0 s, so set it there to reproduce them. **Recheck** is how often the classifier re-runs while "
        "recording; the gate can only close at a recheck instant, so it doubles as a floor "
        "on closing latency. A min-recording shorter than the recheck period is a **no-op**."
    )

    w, y = get_windows(win_s, win_s, win_s, 0.15, st.session_state.get("seed", 0))
    ev = w[w["split"] != "train"].reset_index(drop=True)
    yy = y[w.index[w["split"] != "train"].to_numpy()]

    gp = GateParams(theta_on=0.5, theta_off=0.5, hysteresis_margin=0.0,
                    pre_roll_s=pre_roll, hangover_s=hangover,
                    min_on_s=min_on, recheck_period_s=recheck)
    res = M.evaluate(ev, G.oracle_scores(yy), gp, ann=get_ann(), recordings=get_recordings())

    n_reg = sum(len(r) for r in res["_regions"].values())
    n_files = ev["filename"].nunique()
    per_hr = n_reg / max(n_files, 1)
    mean_len = 3600 * res["duty_cycle"] / max(per_hr, 1e-9)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Birdsong retention", f"{100*res['birdsong_retention']:.2f}%")
    c2.metric("Whole boxes kept", f"{100*res['event_recall@1']:.1f}%",
              help="event_recall@1 — boxes captured 100%, not just touched")
    c3.metric("Duty cycle", f"{100*res['duty_cycle']:.1f}%")
    c4.metric("Saving", f"{100*res['saving']:.1f}%")
    c5.metric("Recordings / hour", f"{per_hr:.0f}",
              help="Separate SD writes. High counts thrash the filesystem and give a "
                   "listener no context — the real reason to set a minimum recording length.")
    c6.metric("Mean recording", f"{mean_len:.0f} s")

    if res["birdsong_retention"] < 0.999:
        st.caption(
            f"⚠️ Retention tops out at {100*res['birdsong_retention']:.2f}%, not 100%, "
            "even with perfect scores. That residue is the `min_overlap_s = 0.15 s` "
            "labelling floor: a box clipping a window by less than 0.15 s never "
            "labels it, so a sliver of that call is dropped. It is a property of the "
            "*labelling rule*, not of the gate — but it is a real ceiling and every "
            "model number must be read against it."
        )

    st.markdown("##### Saving vs (pre-roll, hangover), oracle")
    curve = get_oracle_curve(win_s, 0.15, st.session_state.get("seed", 0))
    piv = curve.pivot(index="pre_roll_s", columns="hangover_s", values="saving")
    st.dataframe((100 * piv).round(1).style.background_gradient(cmap="Greens", axis=None)
                 .format("{:.1f}%"), width="stretch")
    st.caption(
        "Pre-roll quantises to whole decision windows, so at a 1 s window 0.5 s and "
        "1.0 s of pre-roll are the same gate — the table shows identical rows, which "
        "is correct, not a bug. Read it as: **pre-roll is nearly free, hangover is "
        "what you pay.**"
    )

    st.markdown("##### Per-species retention — the deployment risk")
    st.caption(
        "The metric that decides whether a gate is acceptable is the **floor**, not "
        "the mean. A gate that keeps 99% of a loud abundant species and 45% of a "
        "quiet one has injected a species bias into every analysis those recordings "
        "will ever support, and it is undetectable afterwards because the discarded "
        "audio is gone. Species with few boxes are shown but excluded from the floor."
    )
    ps = M.per_species_retention(ev, res["_regions"], get_ann())
    floor = M.retention_floor(ps, min_boxes=20)
    st.metric("Retention floor (species with ≥20 boxes)", f"{100*floor:.1f}%")
    st.dataframe(ps.assign(retention=(100 * ps["retention"]).round(1)).head(20),
                 hide_index=True, width="stretch")

    st.markdown("##### What one hour looks like")
    pick = st.selectbox("Recording", sorted(ev["filename"].unique()), key="oracle_rec")
    g = ev[ev["filename"] == pick]
    reg = res["_regions"][pick]
    boxes = I.merge(get_ann().query("filename == @pick")[["start_s", "end_s"]].to_numpy())
    kept = I.intersect(boxes, reg)

    fig, ax = plt.subplots(figsize=(13, 2.0))
    strip(ax, 2.1, [[0, 3600]], MUTE, "wall clock")
    strip(ax, 1.1, reg, "#9ecae1", "recorded (oracle)")
    strip(ax, 0.1, boxes, MUTE, "birdsong")
    strip(ax, 0.1, kept, HUE)
    ax.set_xlim(0, 3600); ax.set_ylim(0, 3.1)
    ax.set_yticks([])
    tidy(ax, "seconds into the recording")
    st.pyplot(fig); plt.close(fig)
    st.caption(
        f"Grey on the bottom row is birdsong that was **dropped**; teal was kept. "
        f"This hour: {100*I.total(kept)/max(I.total(boxes),1e-9):.1f}% retained, "
        f"{100*(1-I.total(reg)/3600):.1f}% of the hour saved."
    )


# --------------------------------------------------------------------------- #
# Tab 5 — Cochlea                                                              #
# --------------------------------------------------------------------------- #
with tab_cochlea:
    st.subheader("The CAR-IHC front-end — a cascade of asymmetric resonators")
    st.markdown(
        "`P` two-pole-two-zero stages in series, pole frequencies ERB-spaced from "
        "**high (channel 0, cochlear base) to low (channel P-1, apex)**. Each "
        "stage's basilar-membrane output is one channel; the IHC model is a "
        "half-wave rectifier. **No FFT and no logarithm anywhere** — that is the "
        "whole reason it fits on the microcontroller.\n\n"
        "Because it is a *cascade*, the signal reaching channel p has already "
        "passed through every higher-frequency stage, so each channel inherits a "
        "steep high-side roll-off for free. One biquad per channel buys a filter "
        "of effective order p — which is why 40 channels cost 19% of a 48 MHz M4F "
        "rather than all of it."
    )

    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Front-end", ["carihc", "bands"], key="fe_kind",
                        help="`bands` is the control: independent ERB-spaced bandpass "
                             "filters, same channel count, same rectify-and-sum. If it "
                             "ties, the cascade is decoration.")
    n_ch = c2.select_slider("Channels", [16, 24, 30, 40, 56], value=40, key="fe_nch")
    f_lo = c3.select_slider("Lowest pole (Hz)", [80, 150, 300], value=150, key="fe_flo")
    f_hi = c4.select_slider("Highest pole (Hz)", [8000, 10000, 12000, 14000],
                            value=12000, key="fe_fhi")

    fe, fep = get_frontend(kind, n_ch, float(f_lo), float(f_hi), 1.41421356)
    cost = FE.cost_model(fe)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Channels", fe.n_channels)
    m2.metric("Compute", f"{cost['macs_per_s']/1e6:.1f} MMAC/s",
              f"{cost['pct_of_48mhz_m4f']:.0f}% of a 48 MHz M4F")
    m3.metric("Model + state", f"{cost['total_bytes']:,} B",
              help="Filter state + coefficients + SVM weights. The cochlea's whole "
                   "system was 14 KB SRAM / 149 KB flash.")
    m4.metric("Features", fe.n_features,
              help="P channels + log-energy + log-flatness")

    st.markdown("##### Measured channel responses")
    st.caption(
        "Probed with real tones through the real filterbank, not plotted from the "
        "design equations — a response plot drawn from the coefficients would agree "
        "with itself even if the cascade were wired wrongly. Note the **asymmetry**: "
        "each channel falls off far faster above its peak than below. That is the "
        "'A' in CAR, and it is what the zeros above the poles buy."
    )
    freqs, resp = channel_responses(kind, n_ch, float(f_lo), float(f_hi), 1.41421356)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    db = 20 * np.log10(resp + 1e-12)
    for ch in range(fe.n_channels):
        ax.semilogx(freqs, db[:, ch], lw=0.9,
                    color=plt.cm.viridis(ch / max(fe.n_channels - 1, 1)))
    ax.set_ylim(max(db.max() - 60, -80), db.max() + 3)
    ax.set_ylabel("dB", color=INK)
    tidy(ax, "frequency (Hz)")
    ax.grid(which="both", color="#ececec", lw=0.6)
    st.pyplot(fig); plt.close(fig)

    bf = freqs[np.argmax(resp, axis=0)]
    mono = bool(np.all(np.diff(bf) <= 0))
    st.caption(
        f"Best frequency runs {bf[0]:,.0f} Hz (ch 0) → {bf[-1]:,.0f} Hz "
        f"(ch {fe.n_channels-1}), monotonic: **{mono}**. "
        + ("For `carihc` the measured best frequency sits ~7.5% *above* the design "
           "pole — a cascade property, not an error: the stages above have already "
           "attenuated, which shifts the effective peak up." if kind == "carihc" else
           "For `bands` the best frequency is the design centre, since the filters "
           "are independent.")
    )

    st.markdown("##### A cochleagram of real birdsong")
    from birdgate.audio import MelParams, clip_index, load_span, mel_spectrogram
    heard = ann
    if not D.audio_available():
        # Demo mode: only the calls that fall inside a bundled clip can be played.
        clips = clip_index()
        inside = np.zeros(len(ann), dtype=bool)
        for c in clips.itertuples():
            inside |= ((ann["filename"] == c.filename) & (ann["start_s"] >= c.start_s)
                       & (ann["end_s"] <= c.end_s)).to_numpy()
        heard = ann[inside]
        st.caption(f"📦 Playing from the demo clips: {len(heard):,} calls of "
                   f"{heard['label'].nunique()} species. Mount the full audio for all "
                   f"{len(ann):,}.")
    ok = len(heard) > 0
    if not ok:
        st.warning("No audio: neither the full recordings nor the demo clips are present.")

    if ok:
        cc1, cc2 = st.columns([3, 1])
        codes = heard["label"].value_counts().head(25).index.tolist()
        lab = cc1.selectbox("Species", codes, key="coch_sp")
        pool = heard[heard["label"] == lab].reset_index(drop=True)
        i = cc2.number_input("Call #", 0, max(len(pool) - 1, 0), 0, key="coch_i")
        a = pool.iloc[int(i)]

        y_a, sr, lo, hi = load_span(a["filename"], a["start_s"], a["end_s"], pad_s=0.6)
        st.audio(y_a, sample_rate=sr)

        if kind == "carihc":
            coch = CF.cochleagram(y_a, fe._bank)
        else:
            from scipy.signal import sosfilt
            coch = np.maximum(np.stack(
                [sosfilt(fe._sos[c:c + 1], y_a) for c in range(fe.n_channels)]), 0.0)
        # Smooth to a display rate; the raw cochleagram is at the sample rate.
        step = max(1, coch.shape[1] // 900)
        disp = coch[:, :coch.shape[1] // step * step].reshape(
            fe.n_channels, -1, step).mean(axis=2)
        disp = 20 * np.log10(disp + 1e-9)

        fig, (axm, axc) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                       gridspec_kw={"hspace": 0.12})
        import librosa.display
        librosa.display.specshow(mel_spectrogram(y_a, MelParams()), sr=sr,
                                 hop_length=MelParams().hop_length, x_axis="time",
                                 y_axis="mel", fmin=MelParams().fmin,
                                 fmax=MelParams().fmax, ax=axm, cmap="magma")
        axm.set_title("log-mel spectrogram — needs an FFT and a log", fontsize=9, color=INK)
        axm.set_xlabel("")

        extent = [0, len(y_a) / sr, fe.n_channels - 0.5, -0.5]
        axc.imshow(np.clip(disp, disp.max() - 55, None), aspect="auto", cmap="magma",
                   extent=extent, interpolation="nearest")
        ticks = np.linspace(0, fe.n_channels - 1, 6).astype(int)
        axc.set_yticks(ticks)
        axc.set_yticklabels([f"{fe.centre_hz[t]:,.0f}" for t in ticks], fontsize=8)
        axc.set_ylabel("channel centre (Hz)", color=INK)
        axc.set_xlabel("time (s)", color=INK)
        axc.set_title(f"{kind} cochleagram — biquads and max(0, x) only",
                      fontsize=9, color=INK)
        axc.tick_params(colors=INK, labelsize=8)
        st.pyplot(fig); plt.close(fig)
        st.caption(
            "Same audio, two representations. The cochleagram's frequency axis is "
            "**warped and non-uniform** (channels are equally spaced in auditory "
            "filters, not in Hz) and its time resolution is the *sample rate*, not a "
            "frame hop — a cascade has no frames. The classifier never sees this "
            "picture; it sees each channel summed over one second, which is the "
            "42-number vector below."
        )

        st.markdown("##### The template vector Φ — what the SVM actually receives")
        win = int(1.0 * sr)
        if len(y_a) >= win:
            phi = FE.raw_templates(y_a, fe, win)
            feat = FE.normalise(phi, fep)
            fig, ax = plt.subplots(figsize=(12, 2.6))
            for w in range(phi.shape[0]):
                ax.plot(range(fe.n_channels), feat[w, :fe.n_channels], lw=1.6,
                        marker="o", ms=2.5, label=f"second {w}")
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{fe.centre_hz[t]:,.0f}" for t in ticks], fontsize=8)
            ax.set_ylabel("L1-normalised Φ", color=INK)
            ax.legend(frameon=False, fontsize=8)
            tidy(ax, "channel centre frequency (Hz)")
            st.pyplot(fig); plt.close(fig)
            st.caption(
                f"**This is the entire input to the classifier**: {fe.n_channels} "
                "normalised channel sums, plus log-energy and log-flatness. "
                f"log-energy {', '.join(f'{v:.2f}' for v in feat[:, -2])}; log-flatness "
                f"{', '.join(f'{v:.2f}' for v in feat[:, -1])}. ⚠️ L1 normalisation makes the "
                "shape loudness-invariant — right for species ID, **wrong on its own "
                "for presence detection**, since silence normalises to a unit-sum "
                "vector too. The two appended features are what put the level back."
            )


# --------------------------------------------------------------------------- #
# Tab 6 — Compare arms                                                         #
# --------------------------------------------------------------------------- #
with tab_compare:
    st.subheader("What each arm would actually have recorded")
    st.markdown(
        "Every strip below is the **same hour**, the same gate state machine and the "
        "same target retention — only the classifier changes. Grey on an arm's row is "
        "birdsong that arm would have **thrown away**, permanently. The oracle row is "
        "the ceiling: it is what a perfect detector achieves with this state machine, "
        "so the gap between an arm and the oracle is the *model's* deficit, and the "
        "gap between the oracle and the top row is the *state machine's*."
    )

    ARMS = ["carihc", "bands", "mfcc"]
    if using_demo_scores():
        st.info(
            "**Running from the `demo/` bundle.** The linear arms' scores were frozen at "
            "split seed 0, because retraining them needs the 3 GB feature cache. Every "
            "control on this page still applies: they act on the score, not the model.",
            icon="📦")
    cnn_note = cnn_params_note()
    if cnn_note is not None:
        ARMS.append("cnn-int8")
        cp = cnn_note["cnn_params"]
        st.info(
            f"**`cnn-int8` is included, from a frozen training run** "
            f"(`{cnn_note['git_rev'][:8] if cnn_note['git_rev'] else 'no rev'}`): "
            f"channels {tuple(cp['channels'])}, kernel {cp['kernel']}, first-conv time "
            f"stride {cp['first_stride_t']}, {cp['epochs']} epochs, lr {cp['lr']}, "
            f"batch {cp['batch_size']}, seed {cp['seed']}, "
            f"SpecAugment {'on' if cp['spec_augment'] else 'off'}, device `{cp['device']}`. "
            "**Those are NOT driven by the sliders** — a network cannot be refitted between "
            "slider moves, so its raw logits were computed once by "
            "`scripts.train_cnn --save-scores` and cached.\n\n"
            "Everything below the model *is* live: the threshold mode, background window, "
            "anchor, pooling, hangover and minimum-on all apply to this arm exactly as they "
            "do to the others, because they act on the score, not on the model. "
            "So the comparison is fair for every knob on this page, and frozen only for the "
            "ones that are not on it.",
            icon="🧊")
    else:
        st.caption(
            "ℹ️ **`cnn-int8` is not shown** — no cached logits at `runs/g3-cnn/scores.npz`. "
            "The app cannot train a network between slider moves, so the CNN arm has to be "
            "exported once, which takes ~20 min on CPU:\n\n"
            "```\npython -m scripts.train_cnn --save-scores\n```\n"
            "Then rerun this page. Every slider below still applies to it — they act on the "
            "score, not the model.")
    c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1, 1, 1])
    # Held-out recordings first: those are the ones the numbers are about.
    test_recs = set(get_split(st.session_state.get("seed", 0)).query("split == 'test'")["filename"])
    rec = c1.selectbox("Recording", sorted(recs["filename"], key=lambda f: (f not in test_recs, f)),
                       format_func=lambda f: f + ("  (held out)" if f in test_recs else ""),
                       key="cmp_rec")
    mode = c2.selectbox("Threshold", ["quantile", "ema", "fixed"], key="cmp_mode",
                        help="fixed = one absolute threshold everywhere (failed on "
                             "SNE_032). quantile/ema subtract a trailing background "
                             "estimate, so the threshold rises when the noise floor does "
                             "— e.g. when it rains — and one theta means the same thing "
                             "in every recording.")
    target = c3.select_slider("Target retention", [0.90, 0.95, 0.96, 0.98], value=0.96,
                              key="cmp_target")
    hangover = c4.select_slider("Hangover (s)", [0.0, 1.0, 2.0, 5.0], value=1.0,
                                key="cmp_hang")
    min_on = c5.select_slider("Min recording (s)", [0.0, 5.0, 10.0, 20.0, 30.0, 60.0],
                              value=20.0, key="cmp_minon",
                              help="Hold the gate open this long once triggered. Not an "
                                   "acoustic parameter — it buys fewer, longer, more "
                                   "listenable files and spares the filesystem. Default "
                                   "20 s: on `mfcc` it costs 0.16 points of saving against "
                                   "10 s and buys a better worst recording (93.2% → 94.1%), "
                                   "species floor and whole-box capture, with 7% fewer "
                                   "files. At MATCHED retention it is free outright.")

    b1, b2, b3 = st.columns(3)
    bg_win = b1.select_slider("Background window (s)", [60, 120, 300, 600], value=300,
                              key="cmp_bgwin",
                              help="How far back the background estimate looks. Longer is "
                                   "steadier but slower to react to weather.")
    bg_q = b2.select_slider("Background anchor (quantile)", [0.1, 0.2, 0.3, 0.5], value=0.2,
                            key="cmp_bgq",
                            help="Which quantile of recent scores counts as 'background'. "
                                 "MUST stay low. At 0.5 the gate becomes a duty-cycle "
                                 "controller and records ~75% of pure silence; at 0.2 it "
                                 "records ~5%.")
    agg_n = b3.select_slider("Score pooling (max of N)", [1, 2, 3, 5, 8], value=3,
                             key="cmp_aggn",
                             help="Take the max score over the last N seconds before "
                                  "thresholding. Measured best at N=3: recorded silence "
                                  "12.6% -> 4.5% at unchanged saving. It works by allowing "
                                  "a HIGHER theta for the same recall.")
    b4, b5 = st.columns(2)
    bg_scale = b4.selectbox("Scale by", ["global", "mad"], key="cmp_bgscale",
                            help="'global' divides by one constant fitted on train. 'mad' "
                                 "divides by the LOCAL spread — which collapses in quiet "
                                 "stretches and amplifies noise into false triggers. Kept "
                                 "only so the finding stays visible.")
    agg_how = b5.selectbox("Pooling type", ["max", "mean", "median", "none"],
                           key="cmp_agghow",
                           help="mean/median look more principled but measured worse — "
                                "they suppress short calls, forcing theta down.")
    ap = A.AdaptParams(mode=mode, window_s=float(bg_win), bg_quantile=float(bg_q),
                       scale=bg_scale, ema_alpha=0.03, agg=agg_how, agg_n=int(agg_n))
    gp_kw = dict(pre_roll_s=1.0, hangover_s=hangover, hysteresis_margin=0.25,
                 min_on_s=min_on)

    ann_rec = ann[ann["filename"] == rec]
    boxes = I.merge(ann_rec[["start_s", "end_s"]].to_numpy()) if len(ann_rec) else np.zeros((0, 2))
    dur = float(recs.loc[recs["filename"] == rec, "duration_s"].iloc[0])

    lanes, stats = [], []
    for arm in ARMS:
        windows, y, raw, _, _ = get_arm_scores(arm, st.session_state.get("seed", 0))
        tr_idx = windows.index[windows["split"] == "train"].to_numpy()
        ap_a = (dc_replace(ap, global_scale=float(raw[tr_idx].std()) or 1.0)
                if ap.scale == "global" else ap)
        z = A.normalise_by_recording(windows, raw, ap_a)
        # theta on VAL, then applied here — never tuned on the recording being shown
        va = windows.index[windows["split"] == "val"].to_numpy()
        val = windows[windows["split"] == "val"].reset_index(drop=True)
        best = None
        for th in np.quantile(z[va], np.linspace(0.002, 0.998, 70)):
            r = M.evaluate(val, z[va], GateParams(theta_on=float(th), theta_off=None, **gp_kw))
            if r["birdsong_retention"] >= target and (best is None or r["duty_cycle"] < best[0]):
                best = (r["duty_cycle"], float(th))
        th = best[1] if best else float(np.quantile(z[va], 0.02))

        g = windows[windows["filename"] == rec]
        if not len(g):
            continue
        res = M.evaluate(g.reset_index(drop=True), z[g.index.to_numpy()],
                         GateParams(theta_on=th, theta_off=None, **gp_kw))
        reg = res["_regions"][rec]
        lanes.append((arm, reg))
        stats.append({
            "arm": arm, "retention_%": 100 * res["birdsong_retention"],
            "whole_boxes_%": 100 * res["event_recall@1"],
            "saving_%": 100 * res["saving"],
            "recordings": len(reg),
            "mean_len_s": (I.total(reg) / len(reg)) if len(reg) else 0.0,
            "theta": th,
        })

        if arm == ARMS[0]:   # oracle uses the same windows/state machine, once
            ores = M.evaluate(g.reset_index(drop=True),
                              G.oracle_scores(y[g.index.to_numpy()]),
                              GateParams(theta_on=0.5, theta_off=0.5,
                                         hysteresis_margin=0.0, pre_roll_s=1.0,
                                         hangover_s=hangover, min_on_s=min_on))
            oreg = ores["_regions"][rec]
            stats.insert(0, {
                "arm": "ORACLE", "retention_%": 100 * ores["birdsong_retention"],
                "whole_boxes_%": 100 * ores["event_recall@1"],
                "saving_%": 100 * ores["saving"], "recordings": len(oreg),
                "mean_len_s": (I.total(oreg) / len(oreg)) if len(oreg) else 0.0,
                "theta": np.nan})

    lanes.insert(0, ("ORACLE", oreg))

    fig, ax = plt.subplots(figsize=(13, 0.62 * (len(lanes) + 1) + 1.0))
    strip(ax, len(lanes) + 0.1, boxes, "#7a7a7a", "birdsong (truth)", h=0.62)
    for i, (name, reg) in enumerate(lanes):
        y0 = len(lanes) - 1 - i
        kept = I.intersect(boxes, reg) if len(reg) else np.zeros((0, 2))
        strip(ax, y0 + 0.1, boxes, MUTE, name, h=0.62)              # dropped birdsong
        strip(ax, y0 + 0.1, kept, HUE, h=0.62)                      # kept birdsong
        strip(ax, y0 + 0.1, reg, "#9ecae1", h=0.18)                 # recorded extent
    ax.set_xlim(0, dur); ax.set_ylim(0, len(lanes) + 0.85); ax.set_yticks([])
    tidy(ax, "seconds into the recording")
    st.pyplot(fig); plt.close(fig)
    st.caption(
        "Teal = birdsong kept · **grey = birdsong lost** · thin blue = the extent actually "
        "written to the card. Read the grey: that is the irreversible part."
    )

    sdf = pd.DataFrame(stats).round(2)
    st.dataframe(sdf, hide_index=True, width="stretch")
    quiet = np.ones(int(dur), dtype=bool)
    for a_, b_ in boxes:
        quiet[int(a_):int(np.ceil(b_))] = False
    if quiet.sum() > 60:
        qrows = []
        for name, reg in lanes:
            cov = np.zeros(int(dur), dtype=bool)
            for a_, b_ in reg:
                cov[int(a_):int(np.ceil(b_))] = True
            qrows.append({"arm": name,
                          "silence_recorded_%": round(100 * (cov & quiet).sum() / quiet.sum(), 1)})
        st.markdown("##### How much pure silence does each arm record?")
        st.caption(
            f"This hour has {quiet.sum()/60:.0f} minutes with no annotated bird at all. "
            "The oracle records only what pre-roll and hangover spill over. An arm well "
            "above that is chattering through silence — try **Background anchor 0.2** and "
            "**Scale by: global**, then compare against `mad`."
        )
        st.dataframe(pd.DataFrame(qrows), hide_index=True, width="stretch")

    st.caption(
        "`recordings` and `mean_len_s` are the file-count columns — the reason "
        "**Min recording** exists. At min-on 0 s `bands` writes ~158 files/hour "
        "averaging 20 s; at 10 s that becomes 128 files at 24 s for **identical** "
        "saving, and at 30 s, 62 files at 51 s for about 3 points of saving. Fewer, "
        "longer files are easier on FAT32 and far easier to listen to."
    )
    if len(sdf) > 1:
        worst = sdf[sdf["arm"] != "ORACLE"].sort_values("retention_%").iloc[0]
        st.caption(
            f"On this hour the weakest arm is **{worst['arm']}** at "
            f"{worst['retention_%']:.1f}% retention. θ was chosen on the **val** "
            "recordings and applied here unchanged — never tuned on the hour being "
            "shown, which is the whole point of the exercise."
        )

    st.markdown("##### Zoom in and listen to what gets dropped")
    from birdgate.audio import MelParams, clip_index, load_span, mel_spectrogram
    full = D.audio_available()
    clips = clip_index()
    rec_clips = clips[clips["filename"] == rec]
    ok = full or len(rec_clips) > 0
    if not ok:
        st.info("📦 The demo bundle has no clip of this recording. Clips exist for: "
                + ", ".join(f"`{f}`" for f in sorted(clips["filename"].unique()))
                + ". Mount the full audio to listen anywhere.")

    if ok and not full:
        pick = st.selectbox(
            "Demo clip", rec_clips.index,
            format_func=lambda i: f"{rec_clips.loc[i, 'start_s']:.0f} s: {rec_clips.loc[i, 'why']}",
            key="cmp_clip")
        span = 30
        t0 = int(rec_clips.loc[pick, "start_s"])
    elif ok:
        z1, z2 = st.columns([1, 3])
        span = z1.select_slider("Span (s)", [15, 30, 60], value=30, key="cmp_span")
        # Default to the most informative moment: where the arms disagree most.
        drop = np.zeros(int(dur))
        for _, reg in lanes[1:]:
            covered = np.zeros(int(dur), dtype=bool)
            for a_, b_ in reg:
                covered[int(a_):int(np.ceil(b_))] = True
            inbird = np.zeros(int(dur), dtype=bool)
            for a_, b_ in boxes:
                inbird[int(a_):int(np.ceil(b_))] = True
            drop += (inbird & ~covered)
        blk = max(1, int(dur) // 120)
        worst_t = int(np.argmax(np.add.reduceat(drop, np.arange(0, len(drop), blk))) * blk)
        t0 = z2.slider("Start (s)", 0, max(int(dur) - span, 1), min(worst_t, int(dur) - span),
                       step=5, key="cmp_t0")

    if ok:
        y_a, sr, lo, hi = load_span(rec, t0, t0 + span, pad_s=0.0)
        st.audio(y_a, sample_rate=sr)
        near = ann_rec[(ann_rec["end_s"] > t0) & (ann_rec["start_s"] < t0 + span)]

        fig, (axm, axg) = plt.subplots(2, 1, figsize=(13, 4.4 + 0.3 * len(lanes)),
                                       sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1 + 0.35 * len(lanes)],
                                                    "hspace": 0.1})
        import librosa.display
        librosa.display.specshow(mel_spectrogram(y_a, MelParams()), sr=sr,
                                 hop_length=MelParams().hop_length, x_axis="time",
                                 y_axis="mel", fmin=MelParams().fmin,
                                 fmax=MelParams().fmax, ax=axm, cmap="magma")
        for a_ in near.itertuples():
            axm.add_patch(Rectangle((a_.start_s - t0, a_.low_hz), a_.end_s - a_.start_s,
                                    a_.high_hz - a_.low_hz, fill=False, ec=HUE, lw=1.3))
        axm.set_xlabel("")

        bx = I.merge(near[["start_s", "end_s"]].to_numpy() - t0) if len(near) else np.zeros((0, 2))
        strip(axg, len(lanes) + 0.1, bx, "#7a7a7a", "birdsong", h=0.62)
        for i, (name, reg) in enumerate(lanes):
            loc = np.array([[max(a_ - t0, 0), min(b_ - t0, span)] for a_, b_ in reg
                            if b_ > t0 and a_ < t0 + span]) if len(reg) else np.zeros((0, 2))
            y0 = len(lanes) - 1 - i
            strip(axg, y0 + 0.1, bx, MUTE, name, h=0.62)
            if len(loc) and len(bx):
                strip(axg, y0 + 0.1, I.intersect(bx, loc), HUE, h=0.62)
            strip(axg, y0 + 0.1, loc, "#9ecae1", h=0.18)
        axg.set_xlim(0, span); axg.set_ylim(0, len(lanes) + 0.85); axg.set_yticks([])
        tidy(axg, f"seconds from t={t0}")
        st.pyplot(fig); plt.close(fig)
        st.caption(
            f"Opened at t={worst_t} s by default: the window where the arms lose the "
            "most birdsong. Play it, then read the grey." if full else
            f"Demo clip: {rec_clips.loc[pick, 'why']}. Play it, then read the grey."
        )


# --------------------------------------------------------------------------- #
# Tab 7 — Silicon budget                                                       #
# --------------------------------------------------------------------------- #
with tab_silicon:
    st.subheader("Where the part actually goes")
    st.markdown(
        "Four resources, each drawn as the rectangle it physically is, partitioned by "
        "the function that spends it. Every tile is a real allocation: **three of the "
        "four panels are read symbol by symbol out of the linker map of a device "
        "build**, so `2,400 B` of adaptive-threshold history is not an estimate of "
        "what the part might hold, it is what the part does hold.\n\n"
        "Read the panels for two different things. The **white tile** is the answer to "
        "\"can anything else be added\", and on internal SRAM it is the only tile that "
        "matters. The **stage tiles** are the answer to \"what would have to go\", and "
        "they are ordered as `birdgate.c` runs them, so the map can be read against "
        "the block diagram."
    )

    builds = sorted(set(BG.arm_builds()) - {"stock"})
    if not builds:
        st.warning(
            "No linker map on disk, so there is nothing measured to show. "
            "`AM_Software/` is disposable build output — build an arm first:\n\n"
            "```bash\n./firmware/build.sh mfcc-dev\n```", icon="🔨")
        st.stop()

    s1, s2, s3 = st.columns([1.2, 1.4, 2.4])
    arm = s1.selectbox("Arm", builds, index=builds.index("mfcc-dev") if "mfcc-dev" in
                       builds else 0, key="si_arm",
                       help="Only arms that have actually been LINKED appear here. "
                            "`carihc` and `bands` do not: their firmware was deleted "
                            "as dead scaffolding, so there is no map to read and no "
                            "honest footprint to draw.")
    calibrated = s2.toggle(
        "Calibrate compute on the device measurement", value=True, key="si_cal",
        help="Scale the modelled cycle counts by the factor BG_SELFTEST measured on "
             "the CNN arm. On by default because the uncalibrated model is known to "
             "be optimistic; see the compute panel's caption for why applying it to "
             "the linear arm is an extrapolation.")
    fp = BG.footprint(arm)
    ring = BG.ring_model()
    comp = BG.compute_model(arm, calibrated=calibrated)

    def _used(resource):
        g = fp[fp["resource"] == resource]
        return int(g[g["owner"] != "Free"]["bytes"].sum()), int(g[g["owner"] == "Free"]["bytes"].sum())

    sram_used, sram_free = _used("sram")
    flash_used, flash_free = _used("flash")
    ours = fp[fp["owner"] == "BirdGate"].groupby("resource")["bytes"].sum()
    cyc = comp["cycles"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Internal SRAM", f"{sram_used / 1024:.1f} of {BG.SI.sram_bytes // 1024} KB",
              f"{sram_free:,} B left for the stack",
              delta_color="normal" if sram_free > 4096 else "inverse",
              help="The binding resource. 32 KB, and the stack has to live in "
                   "whatever the statics leave.")
    m2.metric("Flash", f"{flash_used / 1024:.0f} of {BG.SI.flash_bytes // 1024} KB",
              f"{flash_free:,} B free", help="240 KB above the bootloader. Not the "
              "constraint on any arm so far.")
    m3.metric("Sample ring", f"{ring['total_s']:.3f} s",
              f"{ring['max_pre_roll_s']:.0f} s reachable by pre-roll",
              delta_color="off",
              help="256 KB of external SRAM as stock divides it. This is the number "
                   "that caps pre-roll.")
    m4.metric("Core time per second", f"{100*cyc/BG.SI.core_hz:.0f}% of 48 MHz",
              f"{(BG.SI.core_hz-cyc)/1e6:.1f} Mcycle/s idle",
              delta_color="normal" if cyc < BG.SI.core_hz else "inverse",
              help="Modelled, not measured. The one panel here that is not read off "
                   "the hardware.")

    # ---- Internal SRAM ---------------------------------------------------- #
    st.markdown(f"##### Internal SRAM — {BG.SI.sram_bytes:,} B, and the one that binds")
    st.caption(
        f"`{arm}` spends **{int(ours.get('sram', 0)):,} B** of the 32 KB; stock spends "
        f"{int(fp[(fp.resource=='sram') & (fp.owner=='Stock')].bytes.sum()):,} B before "
        "we start. Everything here is `.bss` and `.data` from the map, plus the 3,072 B "
        "heap the linker script reserves whether or not anything mallocs."
    )
    fig, ax = plt.subplots(figsize=(13, 6.1))
    area_map(ax, fp[fp["resource"] == "sram"], OWNER_HUE)
    st.pyplot(fig); plt.close(fig)
    st.warning(
        f"⚠️ **The white tile is space left for the stack, not measured free space.** "
        f"The linker knows the statics; it does not know how deep the call stack goes "
        f"at run time, and **no stack high-water measurement exists for this project**. "
        f"So {sram_free:,} B is a ceiling on what the stack may use, and the honest "
        f"reading of this panel is *\"the stack has {sram_free:,} B and nobody has "
        f"checked what it needs.\"* On the CNN arm that figure has been as low as "
        f"3,840 B. Filling the gap means an `-fstack-usage` pass or a fill-and-watch "
        f"probe on the device, and it belongs in ROADMAP before any deployment build.",
        icon="⚠️")
    with st.expander("Every SRAM symbol, by stage"):
        st.dataframe(
            fp[fp["resource"] == "sram"].sort_values("bytes", ascending=False)
              .assign(**{"% of 32 KB": lambda d: (100 * d["bytes"] / BG.SI.sram_bytes).round(2)}),
            hide_index=True, width="stretch", height=340)

    # ---- Flash ------------------------------------------------------------ #
    st.markdown(f"##### Flash — {BG.SI.flash_bytes:,} B above the bootloader")
    hook = BG.hook_cost(arm)
    st.caption(
        f"`{arm}` adds **{int(ours.get('flash', 0)):,} B**, against docs/lab-notebook.md §4's "
        f"< 100 KB target — comfortable, and the interesting part is *what* the bytes "
        "are. They are mostly **constant tables, not code**: the tile sizes say so "
        "directly, and it is why a bigger model is cheap here and a bigger front-end "
        f"is not.\n\n✅ The spent tiles sum to **{flash_used:,} B**, which is the exact "
        f"size of `firmware/out/birdgate-{arm}.bin` — the file that gets flashed. "
        "Nothing in the parser knows about that file, so agreement to the byte is the "
        "check that this panel has neither dropped a symbol nor counted one twice."
    )
    fig, ax = plt.subplots(figsize=(13, 6.1))
    area_map(ax, fp[fp["resource"] == "flash"], OWNER_HUE)
    st.pyplot(fig); plt.close(fig)
    roles = (fp[(fp["resource"] == "flash") & (fp["owner"] == "BirdGate")]
             .groupby("role")["bytes"].sum())
    st.caption(
        "Split by what the bytes *are*, for `" + arm + "`: " +
        " · ".join(f"**{k}** {int(v):,} B" for k, v in roles.sort_values(ascending=False).items()) +
        ". Tables dominate, which is the direction that makes flash the easy resource: "
        "a table is bought once and costs nothing at run time except the wait states "
        "to read it." +
        (f" ℹ️ Differencing `main.o` against the stock build puts our hook into the "
         f"recording loop at **{hook:+,} B** — ⚠️ inconclusive, since that object also "
         f"absorbs every inlining decision the compiler made differently between the "
         f"two builds, and here it comes out negative."
         if hook is not None else "")
    )

    # ---- The ring --------------------------------------------------------- #
    st.markdown(f"##### External SRAM — {BG.SI.ext_sram_bytes:,} B as a sample ring")
    st.caption(
        "Not a treemap, because this resource is not divided by function: stock cuts "
        f"it into **{BG.SI.n_ring_slots} equal slots** and cycles through them, so it "
        "is one structure with a moving head. It matters because the ring is the *only "
        "past audio that exists anywhere* — pre-roll can reach back into it and no "
        "further, which is what turns a tuning knob into a hard ceiling."
    )
    pre_roll_now = float(st.session_state.get("pre_roll", 1.0))
    fig, ax = plt.subplots(figsize=(13, 2.5))
    for i in range(BG.SI.n_ring_slots):
        x0 = i * ring["slot_s"]
        # Slot 0 is where the DMA is writing: it cannot be read, which is why the
        # usable history is one slot shorter than the ring.
        writing = i == 0
        ax.add_patch(Rectangle((x0 + 0.012, 0.42), ring["slot_s"] - 0.024, 0.5,
                               facecolor="#f0f0f0" if writing else _tint(OWNER_HUE["Stock"], 0.72),
                               edgecolor="#cfcfcf" if writing else "none", lw=0.9,
                               hatch="///" if writing else None))
        ax.text(x0 + ring["slot_s"] / 2, 0.67,
                f"slot {i}\n{ring['slot_bytes']//1024} KB · {ring['slot_s']:.3f} s",
                ha="center", va="center", fontsize=7.4, color=INK, linespacing=1.4)
    # Where the past lives, under the slots that hold it: the pale bar is what
    # pre-roll could claim, the solid bar what it is set to claim. Labels go
    # outside their bars — inside, the two collide at every pre-roll below ~2 s.
    ax.plot([ring["readable_s"]] * 2, [0.06, 0.40], color="#9a9a9a", lw=1.0, ls=(0, (2, 2)))
    ax.text(ring["readable_s"] - 0.02, 0.04, f"readable: {ring['readable_s']:.3f} s",
            ha="right", va="bottom", fontsize=7.4, color="#7a7a7a")
    ax.add_patch(Rectangle((0.012, 0.10), ring["max_pre_roll_s"] - 0.024, 0.22,
                           facecolor=_tint(OWNER_HUE["BirdGate"], 0.55), lw=0))
    ax.text(ring["max_pre_roll_s"] + 0.07, 0.21,
            f"ceiling on pre-roll: {ring['max_pre_roll_s']:.0f} s",
            ha="left", va="center", fontsize=8, color=INK)
    ax.add_patch(Rectangle((0.012, 0.10), pre_roll_now - 0.024, 0.22,
                           facecolor=OWNER_HUE["BirdGate"], lw=0))
    if pre_roll_now >= 0.7:
        ax.text(pre_roll_now / 2, 0.21, f"pre-roll {pre_roll_now:.1f} s", ha="center",
                va="center", fontsize=8, color="white", fontweight="bold")
    ax.set_xlim(0, ring["total_s"]); ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, ring["total_s"] + 0.01, ring["slot_s"]))
    ax.set_xticklabels([f"{t:.2f}" for t in np.arange(0, ring["total_s"] + 0.01, ring["slot_s"])],
                       fontsize=7.5)
    tidy(ax, "seconds of audio held (32 kHz, 16-bit mono — 64,000 B/s)")
    st.pyplot(fig); plt.close(fig)
    st.caption(
        f"**{BG.SI.n_ring_slots} × {ring['slot_bytes']:,} B = "
        f"{ring['total_s']:.3f} s at 32 kHz.** The hatched slot is the one the DMA is "
        f"filling, so readable history is {ring['readable_s']:.3f} s, and the firmware "
        f"backfills in whole 1 s windows — hence the **{ring['max_pre_roll_s']:.0f} s** "
        "ceiling the oracle tab's pre-roll slider is drawn against (move it there and "
        "the dark bar here moves). The same 256 KB is 8.192 s at 16 kHz, which is why "
        "inherited AudioMoth advice about pre-roll does not transfer to this "
        "configuration. The ring is also what the overrun bug of 2026-09-22 was "
        "about: a detector slower than real time keeps reading a position the DMA has "
        "already lapped, and every internal counter still agrees with itself."
    )

    # ---- Compute ---------------------------------------------------------- #
    st.markdown(f"##### Core time — {BG.SI.core_hz/1e6:.0f} Mcycle per second of audio")
    st.error(
        "⚠️ **This is the modelled panel, and the model has been measured wrong.** "
        "`BG_SELFTEST` timed the CNN arm's front-end on the device at **450 ms per "
        "window against 110 ms counted by hand — 4.1×** (ROADMAP §C), on code whose "
        "inner loops had been read. Operation *counts* below are exact; they come "
        "from the same exported constants the firmware compiled against. "
        "**Cycles-per-operation are assumptions** (`budget.CYCLES`), and on a "
        "Cortex-M4 running from flash with no cache, cycles are not instructions. "
        "Read the stage *shares*; treat the total as an order of magnitude.",
        icon="⚠️")
    cal = BG.calibration()
    idle = max(BG.SI.core_hz - cyc, 0)
    cdf = comp.groupby(["cls", "stage"], as_index=False)["cycles"].sum()
    cdf = pd.concat([cdf, pd.DataFrame([{"cls": "idle", "stage": "Idle — core asleep"
                                        if idle else "OVER BUDGET", "cycles": idle}])],
                    ignore_index=True)
    fig, ax = plt.subplots(figsize=(13, 5.4))
    area_map(ax, cdf, COMPUTE_HUE, unit="cyc", group="cls", part="stage", value="cycles")
    st.pyplot(fig); plt.close(fig)
    st.caption(
        ("**Calibrated**: front-end stages ×"
         f"{cal['front-end']:.2f} and model stages ×{cal['model']:.2f}, the factors by "
         "which `BG_SELFTEST` on the **CNN** arm exceeded this model. "
         + ("For the CNN that is a fit to its own measurement, so the panel reproduces "
            "790 ms/window by construction. " if arm == "cnn" else
            f"⚠️ For `{arm}` it is an **extrapolation**: the front-end factor is "
            "defensible — both arms run the same `fft.c` over the same 97 frames — and "
            "the model factor is not, because a 40-weight dot product and a "
            "three-layer convolution stack have no reason to miss by the same amount. ")
         if calibrated else
         "**Uncalibrated**: `budget.CYCLES` as written, which the one device "
         "measurement says is optimistic by about 2× on this chain. ") +
        "The core does not run continuously either way — the DMA fills a buffer while "
        "it sits in EM1 — so the idle tile is real sleep, and `energy.py` reads exactly "
        "this fraction as the EM0 wake duty cycle."
    )

    cc1, cc2 = st.columns([3, 2])
    with cc1:
        st.markdown("###### Every stage, with the operation count it comes from")
        st.dataframe(
            comp.assign(ms=comp["ms"].round(2),
                        **{"% of 1 s": comp["pct_of_budget"].round(2)})
                [["stage", "what", "ops", "kind", "cycles_per_op", "ms", "% of 1 s"]],
            hide_index=True, width="stretch", height=380)
    with cc2:
        st.markdown("###### Model against the device")
        meas = BG.MEASURED_MS.get(arm, {})
        if "front-end" in meas:
            raw = BG.compute_model(arm, calibrated=False)
            rows = []
            for cls in ("front-end", "model"):
                rows.append({"phase": cls,
                             "modelled_ms": round(raw.loc[raw["cls"] == cls, "ms"].sum(), 1),
                             "measured_ms": meas[cls],
                             "wrong_by": f"{meas[cls]/raw.loc[raw['cls']==cls,'ms'].sum():.2f}x"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(meas["note"])
        else:
            st.info(
                f"**`{arm}` has never been timed on the device.** " +
                meas.get("note", "") +
                "\n\nThe measurement is cheap — build with `BG_SELFTEST 1` and read "
                "the two LED on-times — and it is the single number that would most "
                "improve this panel. Until then the calibrated total above is an "
                "extrapolation from a different arm.", icon="🔬")

    # ---- Arms side by side ------------------------------------------------ #
    st.markdown("##### Both arms, on all four resources")
    rows = []
    for a in builds:
        f, c = BG.footprint(a), BG.compute_model(a, calibrated=calibrated)
        o = f[f["owner"] == "BirdGate"].groupby("resource")["bytes"].sum()
        rows.append({
            "arm": a,
            "our SRAM (B)": int(o.get("sram", 0)),
            "stack left (B)": int(f[(f.resource == "sram") & (f.owner == "Free")].bytes.sum()),
            "our flash (B)": int(o.get("flash", 0)),
            "flash free (B)": int(f[(f.resource == "flash") & (f.owner == "Free")].bytes.sum()),
            "MACs / window": int(c.loc[c["kind"] == "mac", "ops"].sum()),
            "core time (%)": round(100 * c["cycles"].sum() / BG.SI.core_hz, 1),
            "timed on device": "yes" if "front-end" in BG.MEASURED_MS.get(a, {}) else "no",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "The trade the two arms make is visible in one line: the CNN is **cheaper in "
        "flash tables** than the linear arm — it has no 10,240 B DCT matrix — and dearer "
        "in working memory and in core time. Flash is not scarce, so it pays that trade "
        "at a price in the two resources that are. Both rows come from the live linker "
        "maps, so they move whenever `firmware/build.sh` runs and will disagree with any "
        "figure written down by hand; when they do, the map is right."
    )
