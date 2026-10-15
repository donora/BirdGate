"""Precompute front-end features for all 33 recordings into cache/.

    ../.venv/bin/python -m scripts.build_cache [--force] [--arms carihc,bands]
"""
import argparse

from birdgate import cache as C
from birdgate.config import WindowParams

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(C.ARMS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    meta = C.build(arms=tuple(a.arms.split(",")), wp=WindowParams(), force=a.force)
    print("\ncache ready:", {k: v for k, v in meta["n_features"].items()})
