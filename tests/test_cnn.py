"""The CNN arm's invariants — the ones whose failure would be silent.

Three things can go wrong here without raising anything: the patch handed to the
network can be misaligned with the label (off by a frame or a window), BN folding
can quietly change the function it is supposed to preserve exactly, and the int8
export can degrade the score while every shape still checks out. Each gets a test.

The budget is a test too. "<= 10k parameters, <= 32 KB flash" is a *constraint*,
not an aspiration, and an architecture change that breaches it should fail CI
rather than be discovered at export time.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from birdgate import cnn as CN                    # noqa: E402
from birdgate import frontends as FE              # noqa: E402
from birdgate.config import WindowParams          # noqa: E402


def test_budget():
    """<= 10k parameters and <= 32 KB flash, front-end tables included."""
    cfg = CN.CnnParams()
    m = CN.build(cfg)
    assert CN.deployable_params(m) <= 10_000
    # Flash = int8 weights + the tables the front-end needs but the other arms do
    # not: FFT twiddles (n_fft/2 complex int16) and the mel filterbank (a sparse
    # triangular bank is ~2 int16 per FFT bin, generously counted).
    n_fft = FE.MEL_NFFT
    tables = 2 * 2 * (n_fft // 2) + 2 * 2 * (n_fft // 2)
    assert CN.deployable_params(m) + tables <= 32 * 1024


def test_bn_folds_exactly():
    """`fold_bn` must be an identity on the function, not an approximation."""
    torch.manual_seed(0)
    m = CN.build(CN.CnnParams()).eval()
    # Give BN non-trivial running stats, or folding an untouched BN proves nothing.
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.normal_(0, 1)
                mod.running_var.uniform_(0.5, 2.0)
                mod.weight.normal_(1, 0.2)
                mod.bias.normal_(0, 0.2)
    x = torch.randn(4, 1, CN.N_MELS, CN.FRAMES_PER_S)
    with torch.no_grad():
        assert torch.allclose(m(x), CN.fold_bn(m)(x), atol=1e-5)


def test_macs_match_a_forward_pass():
    """The quoted cost must be the cost of the network that actually runs."""
    cfg = CN.CnnParams()
    counted = CN.macs_per_window(cfg)
    m = CN.build(cfg).eval()
    total = 0

    def hook(mod, inp, out):
        nonlocal total
        if isinstance(mod, torch.nn.Conv2d):
            total += out.shape[2] * out.shape[3] * mod.out_channels * \
                mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, torch.nn.Linear):
            total += mod.in_features * mod.out_features

    hs = [mod.register_forward_hook(hook) for mod in m.modules()]
    with torch.no_grad():
        m(torch.zeros(1, 1, CN.N_MELS, CN.FRAMES_PER_S))
    for h in hs:
        h.remove()
    assert total == counted


def test_int8_weights_are_faithful():
    """Per-channel int8 weights alone must be all but lossless.

    Separated from the activation grid on purpose: if the two are quantised
    together and the result drifts, this test says which half to look at.
    """
    torch.manual_seed(0)
    m = CN.fold_bn(CN.build(CN.CnnParams()).eval())
    x = torch.from_numpy(np.random.default_rng(0).standard_normal(
        (64, 1, CN.N_MELS, CN.FRAMES_PER_S)).astype(np.float32))
    with torch.no_grad():
        f = m(x).squeeze(1).numpy()
        for mod in m:
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                CN._quantise_weights_(mod)
        q = m(x).squeeze(1).numpy()
    assert _spearman(f, q) > 0.999


def test_int8_tracks_float():
    """Quantisation may cost accuracy; it may not change the model's mind.

    ⚠️ The network has to be **trained** for this to mean anything. An untrained
    net's logits have a standard deviation around 0.005 while its intermediate
    activations span ~1, so the int8 grid is coarse relative to a signal that is
    almost entirely noise, and the rank correlation lands near 0.98 no matter how
    good the export path is. Twenty-five seconds of synthetic training gives the
    network a real dynamic range to quantise, and the answer moves to 0.998 —
    which is the number that actually says the export is sound.
    """
    m, x = _trained_toy()
    with torch.no_grad():
        f = m(torch.from_numpy(x)).squeeze(1).numpy()
    gate = CN.CnnGate(model=m, norm=CN.InputNorm(0.0, 1.0), cfg=CN.CnnParams())
    q = CN.quantise(gate, x).forward_np(x)
    assert f.std() > 1.0, "the toy model did not train; the test would be vacuous"
    assert _spearman(f, q) > 0.99


def _spearman(a, b) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def _trained_toy(n: int = 1024, epochs: int = 12):
    """A CNN that has actually learned something, and the patches it learned on.

    The task is the crudest possible caricature of the real one: a horizontal
    energy band in noise, which is what a tonal call looks like on a log-mel patch.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((n, 1, CN.N_MELS, CN.FRAMES_PER_S)) * 0.3).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(np.float32)
    for i in np.flatnonzero(y):
        b = rng.integers(4, CN.N_MELS - 4)
        x[i, 0, b:b + 2, rng.integers(0, 60):] += 1.5
    m = CN.build(CN.CnnParams())
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(epochs):
        for i in range(0, n, 128):
            opt.zero_grad()
            lossf(m(xt[i:i + 128]).squeeze(1), yt[i:i + 128]).backward()
            opt.step()
    return m.eval(), x


@pytest.mark.skipif(not (CN.C.arm_dir("logmel")).exists(), reason="no log-mel cache")
def test_patches_align_with_the_cache():
    """Window `t` must be log-mel frames [100t, 100t+100) of *its own* recording.

    This is the off-by-one that would poison every number downstream without ever
    raising: a one-window shift still trains, still converges, and reports metrics
    against labels for the wrong second of audio.
    """
    from birdgate import windows as W

    wp = WindowParams()
    w, _ = W.build_windows(p=wp)
    w = w.reset_index(drop=True).groupby("filename", group_keys=False).head(3)
    w = w.reset_index(drop=True)
    arrays, file_idx, f0, n = CN.patch_source(w, wp)
    got = CN.gather(arrays, file_idx, f0, n, np.arange(len(w)))
    for i, row in w.iterrows():
        ref = np.load(CN.C.feat_path("logmel", row["filename"]), mmap_mode="r")
        s = int(round(row["start_s"] * CN.FRAMES_PER_S))
        assert np.array_equal(got[i, 0], ref[:, s:s + n]), row["filename"]


def test_streaming_is_the_same_arithmetic():
    """Streaming must be an execution order, not an approximation.

    The whole SRAM argument rests on this: if streaming changed the answer even
    slightly, the 5.9 KB figure would be for a different network than the one the
    accuracy tables describe, and the two could not be quoted together.
    """
    torch.manual_seed(0)
    cfg = CN.CnnParams()
    m = CN.fold_bn(CN.build(cfg).eval())
    x = torch.randn(3, 1, CN.N_MELS, CN.FRAMES_PER_S)
    with torch.no_grad():
        ref = m(x)
        got = CN.stream_forward(m, x, cfg)
    assert torch.allclose(got, ref, atol=1e-6), float((got - ref).abs().max())


def test_streaming_beats_whole_tensor_on_memory():
    """The claim that makes the arm shippable, asserted so a refactor cannot lose it."""
    cfg = CN.CnnParams()
    stream = CN.streaming_bytes(cfg)["total_bytes"]
    whole = CN.activation_bytes(cfg)["total_bytes"]
    assert stream < whole / 5
    # Total working SRAM, FFT buffer included, against docs/lab-notebook.md §12's target.
    assert CN.cost(cfg)["sram_bytes"] <= 8 * 1024


def test_geometry_matches_the_real_model():
    """`geometry` and `layer_spec` describe the model `build` actually returns."""
    cfg = CN.CnnParams()
    ch, fq, ext = CN.geometry(cfg)
    m = CN.fold_bn(CN.build(cfg).eval())
    h = torch.zeros(1, 1, CN.N_MELS, CN.FRAMES_PER_S)
    seen = [(1, CN.N_MELS, CN.FRAMES_PER_S)]
    with torch.no_grad():
        for mod in m:
            if isinstance(mod, torch.nn.ReLU):
                continue
            h = mod(h)
            if h.dim() == 4 and not isinstance(mod, torch.nn.AdaptiveAvgPool2d):
                seen.append((h.shape[1], h.shape[2], h.shape[3]))
    assert seen == list(zip(ch, fq, ext)), (seen, list(zip(ch, fq, ext)))


def test_streaming_quantised_matches_whole_tensor():
    """`stream_forward_quant` is what the firmware mirrors, so it has to be the same
    network as the one every `cnn-int8` number came from — including the edge columns.

    PLAN §9c records why that last clause is not rhetorical: the first streaming
    attempt was bit-exact on every interior column and adrift on the two edges,
    because `padding=1` is a boundary condition on every layer's input rather than
    the network's. Only an equality test catches it.
    """
    import torch
    from birdgate import cnn as CN

    cfg = CN.CnnParams(frontend="logmel-dev")
    torch.manual_seed(0)
    gate = CN.CnnGate(model=CN.build(cfg), norm=CN.InputNorm(-30.0, 12.0), cfg=cfg)
    rng = np.random.default_rng(0)
    n_t = CN.frames_per_window(frontend="logmel-dev")
    calib = (rng.standard_normal((64, 1, CN.N_MELS, n_t)).astype(np.float32) * 12 - 30)
    qg = CN.quantise(gate, calib)

    x = (rng.standard_normal((6, 1, CN.N_MELS, n_t)).astype(np.float32) * 12 - 30)
    assert np.array_equal(qg.forward_np(x), CN.stream_forward_quant(qg, x))

