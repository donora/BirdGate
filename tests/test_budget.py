"""The resource panel must tile its resource exactly, or every share it shows is a lie.

Two classes of assertion here, and the distinction is the point of the file:

- **Bookkeeping.** The per-symbol attribution has to sum to the linker's own
  section headers and then to the size of the part. A treemap whose pieces do not
  fill their rectangle is not a measurement of anything, and the failure is silent:
  bytes that fall out of the attribution reappear as free space, which is the one
  direction that would let the panel say "this fits" when it does not.
- **Against the record.** The CNN's MAC count and the ring's pre-roll ceiling were
  derived independently and written down elsewhere — ROADMAP.md §C and the oracle
  tab's caption. Re-deriving them from the exported constants and hitting the same
  numbers is the only external check this module has.

Footprint tests skip when `AM_Software/AudioMoth-<arm>/build/` is absent: those
trees are disposable build output, so a fresh clone has no maps until
`firmware/build.sh` has run.
"""
import pytest

from birdgate import budget as B

BUILT = sorted(set(B.arm_builds()) - {"stock"})
needs_build = pytest.mark.skipif(not BUILT, reason="no linker map; run firmware/build.sh")


@needs_build
@pytest.mark.parametrize("arm", BUILT)
def test_footprint_tiles_the_part(arm):
    f = B.footprint(arm)
    assert f["bytes"].sum() > 0
    for resource, cap in (("flash", B.SI.flash_bytes), ("sram", B.SI.sram_bytes)):
        g = f[f["resource"] == resource]
        assert int(g["bytes"].sum()) == cap, f"{arm}/{resource} does not tile the part"
        assert (g["bytes"] >= 0).all(), f"{arm}/{resource} has a negative row — overflowed"


@needs_build
@pytest.mark.parametrize("arm", sorted(B.arm_builds()))
def test_symbols_account_for_every_section_byte(arm):
    """Symbols plus padding plus the residual row must equal the section headers.

    The residual is allowed but bounded: it is meant to be the handful of bytes the
    linker script places without naming an object. If it grows into the kilobytes,
    the row parser has stopped matching a whole shape of line and the attribution
    is quietly losing symbols.
    """
    path = B.arm_builds()[arm]
    tot, df = B.section_totals(path), B.parse_map(path)
    for section in (".text", ".data", ".bss"):
        got = int(df.loc[df["section"] == section, "bytes"].sum())
        assert 0 <= tot[section] - got < 256, f"{arm}/{section}: {got} vs {tot[section]}"


@needs_build
@pytest.mark.parametrize("arm", BUILT)
def test_attributed_flash_equals_the_binary_on_disk(arm):
    """The strongest check available: the flash rows must sum to the exact size of
    the `.bin` that gets flashed.

    Nothing in the parser knows about that file, so agreement to the byte says the
    attribution has neither dropped a symbol nor counted one twice. It is skipped
    rather than failed when the `.bin` is older than the map, because the two are
    produced by the same command and a stale pair means the tree is mid-build, not
    that the parser is wrong.
    """
    out = B.FIRMWARE / "out" / f"birdgate-{arm}.bin"
    if not out.exists() or out.stat().st_mtime < B.arm_builds()[arm].stat().st_mtime - 60:
        pytest.skip(f"{out.name} missing or older than the map")
    f = B.footprint(arm)
    used = int(f[(f["resource"] == "flash") & (f["owner"] != "Free")]["bytes"].sum())
    assert used == out.stat().st_size


@needs_build
def test_known_symbols_land_in_the_right_stage():
    """The stage labels are hand-written rules, so spot-check the ones that carry
    the panel's headline: the DCT table and the adaptive threshold's history."""
    f = B.footprint("mfcc-dev")
    by = f.groupby(["owner", "stage"])["bytes"].sum()
    assert by[("BirdGate", "5. DCT / cepstrum")] == 10_240      # 20 x 128 floats
    assert by[("BirdGate", "10. Adaptive threshold")] > 2_400   # 2 x 300 floats + state
    assert by[("Free", "Left for the stack")] > 0, "the linked build does not fit"


def test_ring_caps_pre_roll_at_three_seconds():
    """The number the oracle tab's pre-roll slider is justified by. 8 x 32 KB is
    8 s at 16 kHz and 4.096 s at ours; one slot is always being filled by the DMA,
    and backfill happens in whole 1 s windows, which is what makes it 3."""
    r = B.ring_model()
    assert r["slot_bytes"] == 32_768 and r["total_s"] == pytest.approx(4.096)
    assert r["max_pre_roll_s"] == 3.0


def test_cnn_mac_count_matches_the_device_reckoning():
    """995,360 MACs per window, the figure ROADMAP.md §C reports against 939,984
    counted on the device. It is reached only with the pooling schedule right —
    frequency AND time halve after every conv — so this pins the schedule, not
    just the arithmetic."""
    d = B.compute_model("cnn")
    macs = d[(d["kind"] == "mac") & (d["stage"] == "9. Model arithmetic")]["ops"].sum()
    assert int(macs) == 995_360


@pytest.mark.parametrize("arm", ["mfcc-dev", "cnn"])
def test_compute_model_is_exact_in_ops_and_ordered_in_stages(arm):
    d = B.compute_model(arm)
    assert (d["ops"] > 0).all() and d["cycles_per_op"].notna().all()
    assert set(d["cls"]) <= {"front-end", "model", "decision"}
    # 97 frames x 1024 samples: the external-SRAM read count ROADMAP.md quotes.
    assert int(d.loc[d["stage"] == "1. Frame the sample ring", "ops"].iloc[0]) == 99_328


def test_calibration_reproduces_the_measured_total():
    """Calibrating on the CNN must land on the CNN's measured milliseconds — that
    is what calibration means. It is a tautology by construction, and worth
    asserting because it is the only thing keeping `stage_class` honest: mislabel
    one stage and the factors silently absorb the error."""
    m = B.MEASURED_MS["cnn"]
    d = B.compute_model("cnn", calibrated=True)
    got = d.groupby("cls")["ms"].sum()
    assert got["front-end"] == pytest.approx(m["front-end"], rel=1e-9)
    assert got["model"] == pytest.approx(m["model"], rel=1e-9)
