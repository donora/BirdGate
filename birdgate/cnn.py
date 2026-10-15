"""The tiny-CNN arm — a direct test of the premise behind a tiny linear model.

The usual argument against deep learning on this class of device is **data
scarcity**: for a rare or near-extinct species you may have 50 recordings, and an
SVM handles that where a CNN does not.

That premise does not hold for us. We have 33 hours of strongly-annotated
soundscape and a binary target — roughly 119,000 labelled seconds. So the honest
question is: **with abundant data, does the SVM's advantage survive?** If a 8 KB
int8 CNN clearly beats the template-SVM at comparable cost, the tiny linear model is
the right answer to a question we don't have.

Fair-comparison rules, or the result means nothing:
- same windows, same recording-level split, same metrics
- **count the front-end cost, not just the model.** log-mel needs an FFT; CAR-IHC
  does not. On a 48 MHz M4 that gap can exceed the whole classifier. Report
  MACs/s and flash for front-end + model together (`frontends.cost_model`).
- int8 post-training quantisation, CMSIS-NN-compatible ops only. A float CNN that
  can't be deployed is not a comparison arm.

Target: <= 10k params, <= 32 KB flash including the mel filterbank tables.

## What this arm actually changes

Every other arm collapses a 1 s window to one vector *before* the classifier sees
it: the template sum throws away all time structure inside the second. This arm
keeps it — the input is the full 32 x 100 log-mel patch — so the comparison is
"linear model on a summed spectrum" versus "small conv net on a spectrogram" at
the same decision rate, same windows, same split, same gate.

## Architecture, and why each choice is forced

    (1, 32, 100)  log-mel dB, globally standardised
      conv 3x3 x8,  stride (1, 2) -> BN -> ReLU -> maxpool 2x2   (8, 16, 25)
      conv 3x3 x16            -> BN -> ReLU -> maxpool 2x2       (16, 8, 12)
      conv 3x3 x32            -> BN -> ReLU -> maxpool 2x2       (32, 4, 6)
      global average pool -> 32 -> linear -> 1 logit

- **stride 2 in time in the first conv.** 100 frames is a lot of redundancy at
  10 ms hop, and the first layer is where a full-resolution conv is most
  expensive. Striding there costs nothing measurable and halves the layer.
- **BatchNorm folds away.** At inference BN is an affine per channel, which
  multiplies into the preceding conv's weights and bias (`fold_bn`). It exists
  during training and is not on the device.
- **Global average pool, not flatten.** Flattening (32*4*6 = 768) into a dense
  layer would be 768 weights *and* would tie the model to the exact patch size.
  GAP makes the head 33 weights and the window length a free parameter.
- **No dropout, no residuals, no attention.** They would not survive int8 export
  to CMSIS-NN, and at 6k parameters there is nothing to regularise that weight
  decay cannot reach.

## The input normaliser is a deliberate repeat of docs/lab-notebook.md §4

Per-window mean subtraction (cepstral-mean style) is the reflex here and is
**wrong for a presence detector** for exactly the reason L1 normalisation is
wrong in `frontends.normalise`: it makes silence and birdsong look alike. The
standardisation is one global mean and one global std fitted on **train**, so the
level information survives, and on the device it folds into the input
quantisation scale — zero runtime cost.

Recording-to-recording offset is *not* handled here. It is handled where it is
handled for every other arm: after the score, by `adapt.py`. Keeping that one
mechanism shared is what makes the arms comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import cache as C, frontends as FE
from .config import SAMPLE_RATE, WindowParams

N_MELS = 32
"""Bands in the cached log-mel. Set in `frontends.logmel`; the filterbank table is
flash, so this is a budget decision, not an accuracy one."""

FRAMES_PER_S = 100
"""`frontends.MEL_HOP` = 320 at 32 kHz. A 1 s window is exactly 100 frames, so no
patch ever straddles a fractional frame."""


@dataclass(frozen=True)
class CnnParams:
    """Everything that defines the arm. An experiment is a diff of this."""
    channels: tuple[int, ...] = (8, 16, 32)
    kernel: int = 3
    first_stride_t: int = 2       # stride the first conv in TIME only
    epochs: int = 12
    batch_size: int = 256
    # 1.5e-3, not 3e-3. At 3e-3 the run bifurcates: some runs reach val AP 0.937
    # and others sit at 0.83 with the loss stuck near 0.61, on the same seed. A
    # 6k-parameter net with 8 channels in its first conv has little redundancy —
    # a handful of dead ReLUs early is not recoverable — and this is the cheapest
    # fix. `grad_clip` is the belt to that braces.
    lr: float = 1.5e-3
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    seed: int = 0
    # ⚠️ **'cpu', deliberately, on a machine with a working GPU.** Apple's MPS
    # backend is ~6x faster per epoch here (16 s vs 100 s) and measurably worse at
    # the same seed, config and schedule. One epoch, seed 0:
    #
    #     cpu           loss 0.482   val AP 0.929   (0.483 / 0.929 on a repeat)
    #     mps           loss 0.644   val AP 0.833
    #     mps, repeat   loss 0.670   val AP 0.777
    #     mps, repeat   loss 0.619   val AP 0.916
    #
    # Two failures at once: it does not replicate (Metal fixes no reduction order,
    # so the same seed in the same process diverges), and it converges to a worse
    # place. Gradient clipping is not the cause — the spread is the same with it
    # off. Not chased further; the whole training run is 20 minutes on CPU, which
    # is cheaper than being unsure what a number means. docs/lab-notebook.md §11 asks for runs
    # that replicate, and this is a real difference from every other arm:
    # LinearSVC is convex, so its seed variance is exactly zero.
    device: str = "cpu"
    # Time/frequency masking on the training patches only. Cheap, and the one
    # regulariser that matches the deployment failure modes: a masked frequency
    # band is a wind gust, a masked time slice is a dropout mid-call.
    spec_augment: bool = True
    freq_mask: int = 4
    time_mask: int = 10
    # Epoch selection on VAL, never on test. `val_metric` is the number that picks
    # the checkpoint; average precision, because the gate operates well away from
    # the 0.5 threshold and accuracy would be measuring the wrong region.
    val_metric: str = "ap"
    # Which cached log-mel the arm reads. "logmel-dev" is the causal, device-faithful
    # one: no centred padding, no whole-array dB clamp, and therefore 97 frames per
    # window rather than 100. Global average pooling means that width change costs no
    # parameters, but the weights still have to be refit — it is a different feature
    # distribution. Port work uses "logmel-dev"; "logmel" reproduces docs/lab-notebook.md §9b.
    frontend: str = "logmel"
    # How the log-mel is centred before the network sees it.
    #
    # "fixed" subtracts one global mean fitted on train, which makes the arm depend on
    # the ABSOLUTE recording level. Measured on hardware: the device's acquisition is
    # quieter than the corpus, the input quantiser only spans about 46 dB, and detection
    # falls from 91.5% to 19.0% at -20 dB and to nothing at -30 dB. The linear arm is
    # unaffected, because its features are spectral shape plus a log-energy term the
    # model barely weights.
    #
    # "ema" subtracts a causal exponential mean of the log-mel level instead, held
    # constant across each window. It is level-invariant by construction, and on the
    # busiest recording it also halves the false positives (75 -> 31) for 3.6 points of
    # hit rate. This is what the firmware runs.
    input_centre: str = "ema"
    centre_tau_s: float = 60.0


# --------------------------------------------------------------------------- #
# Data — patches straight out of the log-mel cache, mmapped.
# --------------------------------------------------------------------------- #
def frames_per_window(wp: WindowParams = WindowParams(), frontend: str = "logmel") -> int:
    """Whole log-mel frames inside one decision window: 100 centred, 97 causal."""
    if frontend.endswith("-dev"):
        win_samples = int(round(wp.win_s * SAMPLE_RATE))
        return 1 + (win_samples - FE.MEL_NFFT) // FE.MEL_HOP
    return int(round(wp.win_s * FRAMES_PER_S))


def level_centres(lm, tau_s: float, win_frames: int):
    """Causal per-window estimate of the acquisition level, indexed by frame.

    One update per DEPLOYMENT window (1 s, `FRAMES_PER_S` frames apart), from the mean
    of that window's own log-mel, and the value a window is normalised by is the state
    from BEFORE it — so nothing depends on audio not yet processed. The value is held
    across the frames it covers, so a training window at any offset looks up the centre
    of the deployment window it falls in, and train and device share one cadence.

    Per *frame* would be the obvious choice and cannot be mirrored: the device computes
    97 frames of every 100, so a per-frame average here would drift from the device's by
    construction. Per window uses exactly the frames the firmware has, which is what
    lets `firmware/arms/cnn/cnn.c` hold the same state.

    At a 60 s time constant this tracks the microphone and the scene, not the birds: one
    loud window moves it by under 2%.
    """
    lm = np.asarray(lm, dtype=np.float64)
    n_frames = lm.shape[1]
    out = np.zeros(n_frames, dtype=np.float32)
    alpha = 1.0 / max(tau_s, 1e-9)                 # one update per window, 1 per second
    level = None
    f0 = 0
    while f0 + win_frames <= n_frames:
        patch = lm[:, f0:f0 + win_frames]
        if level is None:
            level = float(patch[:, 0].mean())      # seed on the first frame ever seen
        out[f0:f0 + FRAMES_PER_S] = level
        level += alpha * (float(patch.mean()) - level)
        f0 += FRAMES_PER_S
    if level is not None:
        out[f0:] = level
    return out


def patch_source(windows, wp: WindowParams = WindowParams(), frontend: str = "logmel"):
    """Return `(arrays, file_idx, frame0)` for row-by-row patch extraction.

    The cache holds one `(n_mels, n_frames)` array per recording — the whole hour,
    not per-window slices — so a patch is a view, not a copy, and 33 hours of
    log-mel never has to fit in RAM. `mmap_mode='r'` leaves paging to the OS.

    ⚠️ Every other arm reaches the cache through `cache.assemble`, which assumes
    the cached array is already one row per window. `logmel` is the exception: it
    is stored as a continuous spectrogram precisely so this arm can keep the time
    axis. That is why this function exists instead.
    """
    files = list(dict.fromkeys(windows["filename"]))
    arrays = [np.load(C.feat_path(frontend, f), mmap_mode="r") for f in files]
    fmap = {f: i for i, f in enumerate(files)}
    file_idx = windows["filename"].map(fmap).to_numpy()
    f0 = np.rint(windows["start_s"].to_numpy() * FRAMES_PER_S).astype(np.int64)
    n = frames_per_window(wp, frontend)
    for a, f in zip(arrays, files):
        need = f0[file_idx == fmap[f]].max() + n
        assert a.shape[0] == N_MELS, f"{f}: expected {N_MELS} mels, got {a.shape[0]}"
        assert a.shape[1] >= need, f"{f}: log-mel cache is stale ({a.shape[1]} < {need})"
    return arrays, file_idx, f0, n


def centre_source(arrays, cfg: CnnParams, wp: WindowParams = WindowParams()):
    """Per-array centring series, or None when the arm uses the fixed global mean."""
    if cfg.input_centre != "ema":
        return None
    win = frames_per_window(wp, cfg.frontend)
    return [level_centres(a, cfg.centre_tau_s, win) for a in arrays]


def gather(arrays, file_idx, f0, n, rows, centres=None, in_mean: float = 0.0) -> np.ndarray:
    """Materialise `(len(rows), 1, n_mels, n)` float32 for a batch of window rows."""
    out = np.empty((len(rows), 1, N_MELS, n), dtype=np.float32)
    # Re-centring is applied here rather than inside the model so that everything
    # downstream -- InputNorm, the quantisers, the firmware -- stays unchanged: the net
    # effect is (log-mel - tracked level) / std, exactly as if the fixed mean had been
    # the right one for this recording.
    for k, r in enumerate(rows):
        fi = file_idx[r]
        patch = arrays[fi][:, f0[r]:f0[r] + n]
        if centres is None:
            out[k, 0] = patch
        else:
            out[k, 0] = patch - centres[fi][f0[r]] + in_mean
    return out


@dataclass(frozen=True)
class InputNorm:
    """One global mean and std, fitted on train. See the module docstring."""
    mean: float
    std: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def fit_input_norm(arrays, file_idx, f0, n, train_rows, seed=0, n_sample=20_000,
                   centres=None):
    """Global statistics from a random sample of TRAIN patches only.

    Fitted on ALREADY-CENTRED patches when the arm tracks level, so the mean it
    returns is the residual offset of centred data (near zero) rather than the
    corpus's absolute loudness. Every later `gather` adds that same mean back before
    `InputNorm` removes it, which keeps the net operation
    `(log-mel - tracked level) / std` identical everywhere, firmware included.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(train_rows, size=min(n_sample, len(train_rows)), replace=False)
    x = gather(arrays, file_idx, f0, n, rows, centres, 0.0)
    return InputNorm(float(x.mean()), float(x.std()) or 1.0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build(cfg: CnnParams = CnnParams(), n_mels: int = N_MELS):
    """The torch module. Import is local so the rest of birdgate stays torch-free."""
    import torch.nn as nn

    layers, c_in = [], 1
    for i, c_out in enumerate(cfg.channels):
        stride = (1, cfg.first_stride_t) if i == 0 else 1
        layers += [
            nn.Conv2d(c_in, c_out, cfg.kernel, stride=stride, padding=cfg.kernel // 2,
                      bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        c_in = c_out
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, 1)]
    return nn.Sequential(*layers)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def deployable_params(model) -> int:
    """Parameters that ship: BatchNorm folds into the preceding conv, so it is free."""
    import torch.nn as nn
    return sum(p.numel() for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
               for p in m.parameters())


def macs_per_window(cfg: CnnParams, n_mels: int = N_MELS,
                    n_frames: int = FRAMES_PER_S) -> int:
    """Multiply-accumulates for one forward pass, counted layer by layer.

    Convs only — pooling, ReLU and the GAP are adds and comparisons, and the head
    is 32 MACs. Deliberately the same accounting as `frontends.cost_model`, which
    also counts MACs and also undercounts the cheap ops, so the arms compare.
    """
    h, w, c_in, total = n_mels, n_frames, 1, 0
    for i, c_out in enumerate(cfg.channels):
        if i == 0:
            w = w // cfg.first_stride_t
        total += h * w * c_out * c_in * cfg.kernel ** 2
        h, w, c_in = h // 2, w // 2, c_out
    return total + c_in


def activation_bytes(cfg: CnnParams, n_mels: int = N_MELS,
                     n_frames: int = FRAMES_PER_S) -> dict:
    """Peak int8 working SRAM. **The number that decides whether this arm ships.**

    MACs are the cost everyone quotes and RAM is the cost that actually stops you:
    the AudioMoth budget is <8 KB (docs/lab-notebook.md §12), and a
    convolution needs its whole input feature map and its whole output feature map
    live at once. Counted as:

    - a **persistent** ring of the last `n_frames` log-mel frames (int8), which
      cannot be freed between decisions because the next window overlaps it;
    - the largest `input + output` pair over the layers, since a layer can free
      everything but its own two tensors;
    - CMSIS-NN's im2col scratch, `2 * in_ch * k^2` int16, which is small but real.

    The filterbank arms need none of this — `carihc` and `bands` carry two floats
    per channel and nothing else — so this is where the honest comparison gets
    uncomfortable for the CNN.
    """
    h, w, c_in = n_mels, n_frames, 1
    ring = h * w                                     # int8, persistent
    peak, scratch = 0, 0
    for i, c_out in enumerate(cfg.channels):
        w_out = w // cfg.first_stride_t if i == 0 else w
        peak = max(peak, h * w * c_in + h * w_out * c_out)
        scratch = max(scratch, 2 * c_in * cfg.kernel ** 2 * 2)
        h, w, c_in = h // 2, w_out // 2, c_out
    return {"ring_bytes": ring, "peak_layer_bytes": peak, "im2col_bytes": scratch,
            "total_bytes": ring + peak + scratch}


def cost(cfg: CnnParams = CnnParams(), model=None, n_mels: int = N_MELS) -> dict:
    """Front-end **plus** model, which is the only honest way to quote this arm.

    The log-mel FFT is a cost `carihc` and `bands` simply do not pay, and it is the
    entire usual argument against this family. Quoting the CNN's MACs alone would
    reproduce exactly the sleight of hand docs/lab-notebook.md §8 caught in the other direction.
    """
    from . import frontends as FE
    from .config import FrontEndParams

    fe = FE.build(FrontEndParams(kind="logmel"), 32_000)
    front = FE.cost_model(fe)
    w = deployable_params(model) if model is not None else None
    macs = macs_per_window(cfg, n_mels)          # one window per second
    act = activation_bytes(cfg, n_mels)
    stream = streaming_bytes(cfg, n_mels)
    return {
        "frontend_macs_per_s": front["macs_per_s"],
        "model_macs_per_s": float(macs),
        "macs_per_s": front["macs_per_s"] + macs,
        "pct_of_48mhz_m4f": 100 * (front["macs_per_s"] + macs) / 48e6,
        "n_weights": w,
        "model_bytes_int8": w,                   # int8 weights, per-channel scales aside
        "model_bytes_f32": None if w is None else 4 * w,
        # The FFT working buffer the filterbank arms never allocate. `mfcc` pays
        # it too, so it is a front-end cost rather than a CNN one.
        "fft_buffer_bytes": 4 * FE.MEL_NFFT,
        # Two execution orders, two very different budgets. `sram_bytes` is the
        # one that decides whether this ships, and it is the streaming figure —
        # the whole-tensor number is kept beside it because it is what a training
        # framework does and therefore what an unwary cost estimate reports.
        "sram_bytes": stream["total_bytes"] + 4 * FE.MEL_NFFT,
        "sram_bytes_whole_tensor": act["total_bytes"] + 4 * FE.MEL_NFFT,
        **{f"stream_{k}": v for k, v in stream.items()},
        **{f"act_{k}": v for k, v in act.items()},
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class CnnGate:
    """The trained arm, in the shape the rest of the pipeline expects.

    `score()` returns the raw logit, which plays the role the SVM's margin plays:
    `adapt.py` normalises it per recording and `gate.py` thresholds it. Nothing
    downstream knows or cares that this arm is a CNN — which is the point, since
    that is what keeps the metrics comparable.
    """
    model: object
    norm: InputNorm
    cfg: CnnParams
    arm: str = "cnn"
    history: list = field(default_factory=list)

    @property
    def n_bytes(self) -> int:
        """int8 weights. BN is folded, so it costs nothing here."""
        return deployable_params(self.model)

    def score_all(self, windows, wp: WindowParams = WindowParams()) -> np.ndarray:
        """Same call as `QuantGate.score_all`, so callers need no branch."""
        return score_all(self, windows, wp)


def _device(kind: str = "auto"):
    import torch
    if kind != "auto":
        return torch.device(kind)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _augment(x, cfg: CnnParams, rng):
    """SpecAugment on a numpy batch, in place. Train split only."""
    if not cfg.spec_augment:
        return x
    n, _, n_mels, n_t = x.shape
    if cfg.freq_mask:
        w = rng.integers(0, cfg.freq_mask + 1, size=n)
        s = rng.integers(0, np.maximum(n_mels - w, 1))
        for i in range(n):
            x[i, 0, s[i]:s[i] + w[i], :] = 0.0
    if cfg.time_mask:
        w = rng.integers(0, cfg.time_mask + 1, size=n)
        s = rng.integers(0, np.maximum(n_t - w, 1))
        for i in range(n):
            x[i, 0, :, s[i]:s[i] + w[i]] = 0.0
    return x


def _ap(y, s) -> float:
    """Average precision, without dragging sklearn into the training loop."""
    o = np.argsort(-s)
    y = np.asarray(y, dtype=np.float64)[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1.0))


def fit(windows, y, cfg: CnnParams = CnnParams(), wp: WindowParams = WindowParams(),
        verbose: bool = True) -> CnnGate:
    """Train on the train recordings; select the epoch on val. Test is never read.

    Unweighted BCE, exactly as `svm.py` fits unweighted: the deployment prior is
    handled by moving theta afterwards (`calibrate.py`), not by baking an assumed
    class balance into the weights. Same argument, same decision, so the arms stay
    comparable.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = _device(cfg.device)

    arrays, file_idx, f0, n = patch_source(windows, wp, cfg.frontend)
    centres = centre_source(arrays, cfg)
    tr = windows.index[windows["split"] == "train"].to_numpy()
    va = windows.index[windows["split"] == "val"].to_numpy()
    norm = fit_input_norm(arrays, file_idx, f0, n, tr, cfg.seed, centres=centres)

    model = build(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * (len(tr) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()

    yt = np.asarray(y, dtype=np.float32)
    best, best_state, history = -np.inf, None, []
    step = 0
    for ep in range(cfg.epochs):
        model.train()
        order = rng.permutation(tr)
        tot, nb = 0.0, 0
        for i in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
            rows = order[i:i + cfg.batch_size]
            xb = norm(gather(arrays, file_idx, f0, n, rows, centres, norm.mean))
            xb = torch.from_numpy(_augment(xb, cfg, rng)).to(dev)
            tb = torch.from_numpy(yt[rows]).to(dev)
            loss = lossf(model(xb).squeeze(1), tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if step < steps:
                sched.step()
            step += 1
            tot += float(loss.detach())
            nb += 1
        zv = predict(model, norm, arrays, file_idx, f0, n, va, cfg.batch_size, dev,
                     centres, norm.mean)
        ap = _ap(yt[va], zv)
        history.append({"epoch": ep, "loss": tot / max(nb, 1), "val_ap": ap})
        if verbose:
            print(f"  epoch {ep:2d}  loss {tot/max(nb,1):.4f}  val AP {ap:.4f}",
                  flush=True)
        if ap > best:
            best, best_state = ap, {k: v.detach().clone()
                                    for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return CnnGate(model=model, norm=norm, cfg=cfg, history=history)


def predict(model, norm, arrays, file_idx, f0, n, rows, batch=512, dev=None,
            centres=None, in_mean=None):
    """Logits for `rows`, in `rows` order. No gradient, no shuffling."""
    import torch
    dev = dev or _device()
    was = model.training
    model.eval()
    out = np.empty(len(rows), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            xb = torch.from_numpy(norm(gather(
                arrays, file_idx, f0, n, r,
                centres, norm.mean if in_mean is None else in_mean))).to(dev)
            out[i:i + len(r)] = model(xb).squeeze(1).float().cpu().numpy()
    model.train(was)
    return out


def score_all(gate: CnnGate, windows, wp: WindowParams = WindowParams()) -> np.ndarray:
    """Logit for every row of `windows`, aligned row-for-row. The arm's raw score."""
    arrays, file_idx, f0, n = patch_source(windows, wp, gate.cfg.frontend)
    rows = windows.index.to_numpy()
    return predict(gate.model, gate.norm, arrays, file_idx, f0, n, rows,
                   dev=_device(gate.cfg.device),
                   centres=centre_source(arrays, gate.cfg), in_mean=gate.norm.mean)


# --------------------------------------------------------------------------- #
# int8 — because a float CNN is not a comparison arm
# --------------------------------------------------------------------------- #
def fold_bn(model):
    """Return an equivalent network with every BatchNorm folded into its conv.

        y = gamma * (Wx - mu)/sqrt(var + eps) + beta
          = (gamma/sqrt(var+eps) * W) x + (beta - gamma*mu/sqrt(var+eps))

    so BN is a per-output-channel scale and shift on weights that already exist.
    Nothing is approximated here — `fold_bn(m)` and `m` agree to float error — and
    it is what makes the BN-during-training choice free at deployment.
    """
    import copy
    import torch
    import torch.nn as nn

    src = list(model)
    out = []
    i = 0
    while i < len(src):
        m = src[i]
        if isinstance(m, nn.Conv2d) and i + 1 < len(src) and isinstance(src[i + 1], nn.BatchNorm2d):
            bn = src[i + 1]
            conv = nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size,
                             stride=m.stride, padding=m.padding, bias=True)
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            with torch.no_grad():
                conv.weight.copy_(m.weight * s.reshape(-1, 1, 1, 1))
                b = torch.zeros(m.out_channels) if m.bias is None else m.bias.detach().cpu()
                conv.bias.copy_(bn.bias.detach().cpu()
                                + s.detach().cpu() * (b - bn.running_mean.detach().cpu()))
            out.append(conv)
            i += 2
        else:
            out.append(copy.deepcopy(m))
            i += 1
    return nn.Sequential(*out).eval()


class ActQuant:
    """Fake-quantise one activation tensor to the int8 grid CMSIS-NN would use.

    Non-negative tensors (everything downstream of a ReLU) get the full 256 levels
    above zero — CMSIS-NN's s8 with zero-point -128. Signed tensors get a
    symmetric grid, which costs a level and buys auditability.
    """

    def __init__(self, amax: float, signed: bool, bypass: bool = False):
        self.signed, self.amax, self.bypass = signed, max(float(amax), 1e-12), bypass
        # A bypassed quantiser is a pass-through and exports a scale of 0, which is
        # the firmware's signal to skip it.
        self.scale = 0.0 if bypass else self.amax / (127.0 if signed else 255.0)

    def __call__(self, x):
        import torch
        if self.bypass:
            return x
        q = torch.round(x / self.scale)
        lo, hi = (-128, 127) if self.signed else (0, 255)
        return torch.clamp(q, lo, hi) * self.scale


def _quantise_weights_(module):
    """Per-output-channel symmetric int8 weights, in place. Biases stay int32-wide.

    Per-channel scales are what `arm_convolve_s8` already expects, and they matter:
    with one scale for the whole tensor a single wide channel costs every other
    channel most of its resolution.
    """
    import torch
    with torch.no_grad():
        w = module.weight
        flat = w.reshape(w.shape[0], -1)
        s = flat.abs().max(dim=1).values.clamp_min(1e-12) / 127.0
        shape = (-1,) + (1,) * (w.dim() - 1)
        w.copy_(torch.round(w / s.reshape(shape)).clamp_(-127, 127) * s.reshape(shape))
    return module


def quantise(gate: CnnGate, calib_x: np.ndarray) -> "QuantGate":
    """Post-training int8: fold BN, quantise weights per-channel, activations per-tensor.

    ⚠️ **This simulates int8 arithmetic; it is not a bit-exact CMSIS-NN run.** It
    rounds the weights and every activation onto the grid the device would use and
    computes in float, which captures the resolution loss but not the accumulator
    and requantisation details. Bit-exact parity against the C implementation is
    the firmware port's job, with the same treatment `firmware/test/test_parity.c`
    gives the gate. The number that decides whether the export was acceptable is
    retention through the gate, not weight SNR (budget ≤ 0.2 %, ROADMAP F10).

    Activation ranges come from a **train** calibration batch at the 99.9th
    percentile of |x|, not the max: one outlier patch setting the scale would cost
    every ordinary patch a bit of resolution.
    """
    import torch
    import torch.nn as nn

    def _p999(t):
        """99.9th percentile of |t|, subsampled — torch.quantile caps its input."""
        f = t.detach().abs().flatten().float()
        if len(f) > 1_000_000:
            f = f[torch.randperm(len(f))[:1_000_000]]
        return float(torch.quantile(f, 0.999))

    folded = fold_bn(gate.model.to("cpu"))
    mods = list(folded)
    # Weights first, so the activation ranges are calibrated on the network that
    # will actually run — not on the float one it was derived from.
    for m in mods:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            _quantise_weights_(m)

    with torch.no_grad():
        x = torch.from_numpy(gate.norm(calib_x))
        acts = [ActQuant(_p999(x), signed=True)]           # the input
        h = acts[0](x)
        for i, m in enumerate(mods):
            h = m(h)
            # Signedness is read off the tensor, not guessed from the layer type:
            # everything downstream of a ReLU is non-negative, including the pools.
            #
            # Two quantisers are bypassed, both artefacts of the "after every module"
            # rule rather than anything a device would do. `Flatten` changes no values,
            # so quantising its output only adds noise. And the final `Linear` produces
            # ONE number that is immediately compared to a threshold: a device reads it
            # out of the int32 accumulator, where CMSIS-NN leaves it, rather than
            # storing it as int8. Quantising it spreads 8 bits over the whole logit
            # range — a step of ~0.07 against a threshold near 0.54 — and it was the
            # dominant reason the C and this reference disagreed on gate decisions.
            bypass = (i == len(mods) - 1) or isinstance(m, nn.Flatten)
            q = ActQuant(_p999(h), signed=bool((h < 0).any()), bypass=bypass)
            acts.append(q)
            h = q(h)
    return QuantGate(mods=mods, acts=acts, norm=gate.norm, cfg=gate.cfg)


@dataclass
class QuantGate:
    """The int8-simulated network. Same `score` contract as `CnnGate`."""
    mods: list
    acts: list
    norm: InputNorm
    cfg: CnnParams
    arm: str = "cnn-int8"

    def forward_np(self, x: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            h = self.acts[0](torch.from_numpy(self.norm(x)))
            for m, q in zip(self.mods, self.acts[1:]):
                h = q(m(h))
            return h.squeeze(1).double().numpy()

    def score_all(self, windows, wp: WindowParams = WindowParams(),
                  batch: int = 512) -> np.ndarray:
        arrays, file_idx, f0, n = patch_source(windows, wp, self.cfg.frontend)
        centres = centre_source(arrays, self.cfg)
        rows = windows.index.to_numpy()
        out = np.empty(len(rows), dtype=np.float64)
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            out[i:i + len(r)] = self.forward_np(
                gather(arrays, file_idx, f0, n, r, centres, self.norm.mean))
        return out

    @property
    def n_bytes(self) -> int:
        import torch.nn as nn
        return sum(m.weight.numel() + (0 if m.bias is None else 4 * m.bias.numel())
                   for m in self.mods if isinstance(m, (nn.Conv2d, nn.Linear)))


# --------------------------------------------------------------------------- #
# Streaming execution — the same arithmetic, a fifteenth of the memory
# --------------------------------------------------------------------------- #
"""
`activation_bytes` costs the network the way a training framework runs it: build
each feature map whole, then the next. That is what made this arm look unshippable
at 23.9 KB against an 8 KB budget, with 16 KB of it in the first conv's input and
output held live together.

**That cost is an artefact of the execution order, not a property of the network.**
The log-mel front-end produces one frame every 10 ms, and evaluation windows do
not overlap (`WindowParams.eval_hop_s == win_s`), so a frame can be consumed the
moment it exists and never stored. Each layer keeps a ring of the last `k` input
columns, emits an output column whenever its stride phase comes round, and the
global average pool accumulates into 32 int32s. Nothing else is live.

- **the 3.2 KB log-mel patch buffer disappears** — there is no patch, only a stream
- **the 16 KB feature-map pair disappears** — one column per layer, not one map
- **MACs are unchanged** — every position is computed exactly once, so unlike
  tiled-with-recomputation (which costs 1.57x MACs to reach 13.4 KB) this is free
- **the result is bit-identical**, which `tests/test_cnn.py` asserts rather than
  assumes

What remains is the FFT working buffer, which is a front-end cost `mfcc` pays too.
"""


def layer_spec(cfg: CnnParams) -> list[tuple]:
    """`(kind, k, stride, pad)` on the TIME axis, in execution order.

    Derived from `build()` rather than written down twice — a spec that drifts from
    the model it describes would make every memory number here fiction.
    """
    spec = []
    for i, _ in enumerate(cfg.channels):
        s = cfg.first_stride_t if i == 0 else 1
        spec.append(("conv", cfg.kernel, s, cfg.kernel // 2))
        spec.append(("pool", 2, 2, 0))
    return spec


def geometry(cfg: CnnParams, n_mels: int = N_MELS, n_frames: int = FRAMES_PER_S):
    """Channels and frequency extent at each layer boundary, plus time extents."""
    ch, fq, ext = [1], [n_mels], [n_frames]
    for kind, k, s, p in layer_spec(cfg):
        n = (ext[-1] + 2 * p - k) // s + 1
        ext.append(n)
        if kind == "conv":
            ch.append(cfg.channels[len(ch) // 2])
            fq.append(fq[-1])
        else:
            ch.append(ch[-1])
            fq.append(fq[-1] // 2)
    return ch, fq, ext


def streaming_bytes(cfg: CnnParams, n_mels: int = N_MELS,
                    n_frames: int = FRAMES_PER_S) -> dict:
    """Peak int8 working set under streaming execution. Compare `activation_bytes`.

    Counted as the worst moment of the cascade: every stage's column ring, plus the
    `k`-column window the active stage is convolving and the column it emits, plus
    the int32 accumulator the global average pool sums into.
    """
    spec = layer_spec(cfg)
    ch, fq, _ = geometry(cfg, n_mels, n_frames)
    rings = [k * ch[i] * fq[i] for i, (_, k, _, _) in enumerate(spec)]
    peak = 0
    for i, (_, k, _, _) in enumerate(spec):
        transient = k * ch[i] * fq[i] + ch[i + 1] * fq[i + 1]
        peak = max(peak, sum(rings) + transient)
    gap_acc = 4 * cfg.channels[-1]
    return {"stage_rings": sum(rings), "gap_accumulator": gap_acc,
            "total_bytes": peak + gap_acc}


class _Stage:
    """One conv-or-pool stage of the streaming cascade.

    Owns its own time padding. That is the subtlety that makes a naive tiled
    implementation wrong: `padding=1` is a boundary condition on **every layer's**
    input, not just the network's, so a tile that reaches past the patch edge must
    see zeros there — not `conv(zeros) + bias`, which is what it gets if only the
    input is padded. Costing that mistake: it moved the logits by 4e-3 on the two
    edge columns while every interior column stayed bit-exact.
    """

    def __init__(self, kind, k, s, p, n_in, zero, conv=None):
        import torch  # noqa: F401
        self.kind, self.k, self.s, self.p, self.n_in, self.conv = kind, k, s, p, n_in, conv
        self.n_out = (n_in + 2 * p - k) // s + 1
        self.zero, self.buf, self.seen, self.next_out = zero, {}, 0, 0
        self.filled = p
        for j in range(p):
            self.buf[j] = zero

    def _drain(self):
        import torch
        import torch.nn.functional as F
        outs = []
        while self.next_out < self.n_out and self.next_out * self.s + self.k <= self.filled:
            a = self.next_out * self.s
            win = torch.cat([self.buf[j] for j in range(a, a + self.k)], dim=3)
            if self.kind == "conv":
                outs.append(F.relu(F.conv2d(win, self.conv.weight, self.conv.bias,
                                            stride=(1, self.s), padding=(1, 0))))
            else:
                outs.append(F.max_pool2d(win, 2))
            self.next_out += 1
            for j in [j for j in self.buf if j < self.next_out * self.s]:
                self.buf.pop(j)
        return outs

    def push(self, col):
        self.buf[self.p + self.seen] = col
        self.seen += 1
        self.filled = self.p + self.seen
        outs = self._drain()
        if self.seen == self.n_in:
            for j in range(self.p):
                self.buf[self.p + self.n_in + j] = self.zero
                self.filled += 1
            outs += self._drain()
        return outs


def stream_forward(folded, x, cfg: CnnParams = CnnParams()):
    """Reference streaming implementation: one frame in at a time, no patch buffer.

    Takes a **BN-folded** `Sequential` (see `fold_bn`) and must agree with calling
    it directly to float rounding. That equality is the whole claim, so it is a
    test, not a comment.
    """
    import torch
    import torch.nn as nn

    convs = [m for m in folded if isinstance(m, nn.Conv2d)]
    lin = folded[-1]
    B, _, n_mels, n_t = x.shape
    ch, fq, _ = geometry(cfg, n_mels, n_t)
    stages, n, ci = [], n_t, 0
    for i, (kind, k, s, p) in enumerate(layer_spec(cfg)):
        conv = None
        if kind == "conv":
            conv = convs[ci]
            ci += 1
        st = _Stage(kind, k, s, p, n, torch.zeros(B, ch[i], fq[i], 1), conv)
        stages.append(st)
        n = st.n_out

    acc, cnt = torch.zeros(B, cfg.channels[-1]), 0
    for t in range(n_t):
        outs = [x[:, :, :, t:t + 1]]
        for st in stages:
            nxt = []
            for c in outs:
                nxt += st.push(c)
            outs = nxt
            if not outs:
                break
        for c in outs:
            acc = acc + c.sum(dim=(2, 3))
            cnt += c.shape[2] * c.shape[3]
    return lin(acc / cnt)


def stream_forward_quant(qg, x):
    """Streaming schedule **and** int8-grid arithmetic: what `arms/cnn/cnn.c` mirrors.

    `stream_forward` has the schedule and `QuantGate.forward_np` has the arithmetic,
    and neither has both. The firmware needs both at once, so this is the reference it
    is checked against — and it in turn must agree with `forward_np`, which is what
    produced every `cnn-int8` number on record.

    Written out rather than assembled from `_Stage` so it mirrors the C line for line:
    the same per-stage rings, the same per-stage time padding, and the quantisers in
    the same places (conv, quantise, ReLU, quantise, pool, quantise).
    """
    import torch
    with torch.no_grad():
        return _stream_forward_quant(qg, x)


def _stream_forward_quant(qg, x):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = qg.cfg
    convs = [m for m in qg.mods if isinstance(m, nn.Conv2d)]
    lin = qg.mods[-1]
    q = qg.acts                       # [input, then one after every module]
    n_conv = len(convs)
    q_in, q_conv = 0, lambda i: 1 + 3 * i
    q_relu, q_pool = lambda i: 2 + 3 * i, lambda i: 3 + 3 * i
    q_avg = 1 + 3 * n_conv

    x = torch.from_numpy(qg.norm(np.asarray(x, dtype=np.float32)))
    B, _, n_mels, n_t = x.shape
    ch, fq, _ = geometry(cfg, n_mels, n_t)
    spec = layer_spec(cfg)

    # Per-stage: k columns of its own input, the count seen, and the next output due.
    st = []
    n = n_t
    for i, (kind, k, s, p) in enumerate(spec):
        n_out = (n + 2 * p - k) // s + 1
        st.append(dict(kind=kind, k=k, s=s, p=p, n_in=n, n_out=n_out,
                       buf={j: torch.zeros(B, ch[i], fq[i], 1) for j in range(p)},
                       filled=p, seen=0, next_out=0, idx=i // 2))
        n = n_out

    acc = torch.zeros(B, cfg.channels[-1], dtype=torch.float64)
    cnt = 0

    def column(s_i, a):
        """One output column of stage `s_i`, windows starting at absolute index `a`."""
        S = st[s_i]
        win = torch.cat([S["buf"][j] for j in range(a, a + S["k"])], dim=3)
        i = S["idx"]
        if S["kind"] == "conv":
            h = F.conv2d(win, convs[i].weight, convs[i].bias,
                         stride=(1, S["s"]), padding=(1, 0))
            h = q[q_conv(i)](h)            # conv -> quantise
            h = q[q_relu(i)](F.relu(h))    # ReLU -> quantise
        else:
            h = q[q_pool(i)](F.max_pool2d(win, 2))
        return h

    def push(s_i, col):
        nonlocal acc, cnt
        S = st[s_i]
        S["buf"][S["filled"]] = col
        S["filled"] += 1
        S["seen"] += 1
        drain(s_i)
        if S["seen"] == S["n_in"]:
            for _ in range(S["p"]):
                S["buf"][S["filled"]] = torch.zeros_like(col)
                S["filled"] += 1
            drain(s_i)

    def drain(s_i):
        nonlocal acc, cnt
        S = st[s_i]
        while S["next_out"] < S["n_out"] and S["next_out"] * S["s"] + S["k"] <= S["filled"]:
            a = S["next_out"] * S["s"]
            out = column(s_i, a)
            S["next_out"] += 1
            for j in [j for j in S["buf"] if j < S["next_out"] * S["s"]]:
                S["buf"].pop(j)
            if s_i + 1 < len(st):
                push(s_i + 1, out)
            else:
                acc = acc + out.sum(dim=(2, 3)).double()
                cnt += out.shape[2] * out.shape[3]

    for t in range(n_t):
        push(0, q[q_in](x[:, :, :, t:t + 1]))

    h = (acc / max(cnt, 1)).float()
    h = q[q_avg](h)                        # AdaptiveAvgPool2d -> quantise
    h = q[q_avg + 1](h)                    # Flatten -> quantise (values unchanged)
    return q[q_avg + 2](lin(h)).squeeze(1).double().numpy()

