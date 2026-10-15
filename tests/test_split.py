"""The recording-level split is frozen: the same six val and six test recordings
every run, and the same ones the sibling project holds out. If this fails, every
number in every report is being compared across a different test set.
"""
from birdgate import splits as SP

VAL = [
    "SNE_005_20180509_090004.flac", "SNE_008_20180509_190001.flac",
    "SNE_011_20180510_070003.flac", "SNE_018_20180523_050002.flac",
    "SNE_025_20180523_190001.flac", "SNE_031_20180529_090004.flac",
]
TEST = [
    "SNE_004_20180509_080004.flac", "SNE_012_20180510_080004.flac",
    "SNE_016_20180510_190001.flac", "SNE_021_20180523_080004.flac",
    "SNE_030_20180529_080004.flac", "SNE_032_20180529_170000.flac",
]


def test_split_is_frozen():
    s = SP.make_split(seed=0)
    by = {k: sorted(g["filename"]) for k, g in s.groupby("split")}
    assert by["val"] == VAL
    assert by["test"] == TEST
    assert len(by["train"]) == 21
    assert set(by["train"]).isdisjoint(VAL + TEST)


def test_apportion_sums_exactly():
    for n in range(1, 40):
        parts = SP._apportion(SP.DEFAULT_FRACTIONS, n)
        assert sum(parts.values()) == n
