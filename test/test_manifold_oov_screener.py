"""Tests for 3D manifold OOV score model selection and scoring."""

from __future__ import annotations

import numpy as np

from pelinker.config import ManifoldOovScreenerConfig
from pelinker.manifold_oov_screener import (
    ManifoldOovScoreModel,
    evaluate_manifold_oov_cv,
    fit_manifold_oov_score_model,
)


def test_evaluate_manifold_oov_cv_returns_winner_and_payload() -> None:
    rng = np.random.default_rng(42)
    n = 80
    X0 = rng.normal(size=(n, 3)).astype(np.float64) * 0.3
    X1 = rng.normal(loc=2.0, size=(n // 4, 3)).astype(np.float64)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n // 4, dtype=np.int64)])
    cfg = ManifoldOovScreenerConfig(
        cv_n_splits=5,
        cv_test_size=0.25,
        cv_random_state=0,
    )
    raw = evaluate_manifold_oov_cv(X, y, cfg)
    assert raw is not None
    cv_payload, winner = raw
    assert winner in ("lda", "svm", "rbf")
    assert "winner_f1_mean" in cv_payload
    model, pl = fit_manifold_oov_score_model(X, y, cfg, cv_payload_and_winner=raw)
    assert model is not None
    assert pl is cv_payload
    assert isinstance(model, ManifoldOovScoreModel)
    scores = model.score(X)
    assert scores.shape == (X.shape[0],)
    oov = model.is_oov(X)
    assert oov.shape == (X.shape[0],) and oov.dtype == bool


def test_fit_manifold_oov_returns_none_when_cv_infeasible() -> None:
    X = np.zeros((4, 3), dtype=np.float64)
    y = np.array([0, 0, 0, 1], dtype=np.int64)
    cfg = ManifoldOovScreenerConfig(cv_n_splits=20)
    assert evaluate_manifold_oov_cv(X, y, cfg) is None
    assert fit_manifold_oov_score_model(X, y, cfg, cv_payload_and_winner=None) == (
        None,
        None,
    )
