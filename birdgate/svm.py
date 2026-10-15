"""Template-SVM — a linear model over the front-end's template outputs.

    f(X) = sum_p Q_p * Phi_p(X) + b

That is the entire model. The "in-filter computing" idea is that Phi_p *is* the
p-th front-end channel, so feature extraction and the SVM's kernel evaluation are
the same computation. We need one score rather than one-vs-all over several
classes, so the deployable model is **P+2 weights and a bias** — about 170 bytes.

## Standardisation is free at deployment

Training standardises features (zero mean, unit variance) because a linear SVM on
raw inputs is dominated by whichever feature has the largest scale — here
`log_energy`, which spans ~10 while the L1-normalised channels span ~0.05. But
standardisation does **not** cost anything on the device, because it folds into
the weights analytically:

    w·(x - mu)/sigma + b  ==  (w/sigma)·x + (b - sum(w*mu/sigma))

`fold()` does that, so what ships is still one dot product and one add. Verified
against the sklearn pipeline in `tests/test_svm.py`.

## Which prior to fit against

Negatives are ~48% of windows in this dawn-chorus data and will be 90-98% in a
real Highland deployment. Two ways to handle it:

1. **Reweight the loss** toward the deployment prior, so the boundary is placed
   for the world the device will see.
2. **Fit unweighted and move the threshold afterwards** (`calibrate.py`).

We do (2) by default. It is cheaper, it is the same knob the deployer already has,
and — the deciding argument — a threshold can be recalibrated on Highland audio
*without retraining*, whereas a reweighted fit bakes an assumed prior into the
weights. `class_weight` is exposed so the choice can be measured, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearGate:
    """The deployable model: weights, bias, and where they came from."""
    w: np.ndarray
    b: float
    feature_names: list[str]
    arm: str
    class_weight: str | None = None

    def score(self, X: np.ndarray) -> np.ndarray:
        """f(X) for every window. Raw margin — thresholding is `gate.py`'s job."""
        return np.asarray(X, dtype=np.float64) @ self.w + self.b

    @property
    def n_bytes(self) -> int:
        """float32 on the device."""
        return (len(self.w) + 1) * 4


def fit(X: np.ndarray, y: np.ndarray, arm: str, feature_names: list[str],
        class_weight: str | None = None, C: float = 1.0, seed: int = 0) -> LinearGate:
    """Train the linear SVM (hinge loss) and fold the scaler into the weights."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    X = np.nan_to_num(np.asarray(X, dtype=np.float64), posinf=0.0, neginf=0.0)
    pipe = make_pipeline(
        StandardScaler(),
        LinearSVC(C=C, class_weight=class_weight, dual="auto", max_iter=5000,
                  random_state=seed),
    )
    pipe.fit(X, np.asarray(y).ravel())
    return fold(pipe, arm, feature_names, class_weight)


def fold(pipe, arm: str, feature_names: list[str], class_weight=None) -> LinearGate:
    """Collapse StandardScaler + LinearSVC into a single (w, b)."""
    sc, svc = pipe.steps[0][1], pipe.steps[-1][1]
    w = svc.coef_.ravel() / sc.scale_
    b = float(svc.intercept_[0] - (svc.coef_.ravel() * sc.mean_ / sc.scale_).sum())
    return LinearGate(w=w, b=b, feature_names=list(feature_names), arm=arm,
                      class_weight=class_weight)


def top_weights(model: LinearGate, k: int = 8):
    """The k features the model leans on hardest. Sanity, and interpretability —
    a 42-weight model is small enough that you can just look at it."""
    order = np.argsort(-np.abs(model.w))[:k]
    return [(model.feature_names[i], float(model.w[i])) for i in order]
