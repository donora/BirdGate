"""The scaler-folding must be exact, or the device computes a different model."""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from birdgate import svm as S


def test_folded_weights_reproduce_the_pipeline_exactly():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 12)) * rng.uniform(0.01, 50, 12)   # wildly varied scales
    y = (X @ rng.standard_normal(12) + rng.standard_normal(500) * 0.1 > 0).astype(int)

    pipe = make_pipeline(StandardScaler(), LinearSVC(dual="auto", max_iter=5000, random_state=0))
    pipe.fit(X, y)
    model = S.fold(pipe, "test", [f"f{i}" for i in range(12)])

    expected = pipe.decision_function(X)
    assert np.abs(model.score(X) - expected).max() < 1e-8
    assert np.array_equal(model.score(X) > 0, pipe.predict(X) == 1)


def test_model_is_tiny():
    m = S.LinearGate(w=np.zeros(42), b=0.0, feature_names=["x"] * 42, arm="carihc")
    assert m.n_bytes == 172, "the whole deployable model must stay in the low hundreds of bytes"
