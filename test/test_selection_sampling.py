"""Stratified selection draws for model-selection evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pelinker.config import ClusteringOptimizationConfig
from pelinker.onto import NEGATIVE_LABEL
from pelinker.sampling import (
    draw_selection_sample,
    selection_sample_target_size,
    stratified_mention_sample,
)


def _synthetic_frame(n_kb: int, n_neg: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for i in range(n_kb):
        rows.append(
            {
                "pmid": i,
                "entity": "kb_a",
                "mention": f"k{i}",
                "embed": rng.standard_normal(4).astype(np.float32),
            }
        )
    for j in range(n_neg):
        rows.append(
            {
                "pmid": 10_000 + j,
                "entity": NEGATIVE_LABEL,
                "mention": f"n{j}",
                "embed": rng.standard_normal(4).astype(np.float32),
            }
        )
    return pd.DataFrame(rows)


def test_selection_sample_target_size_frac_and_cap() -> None:
    assert (
        selection_sample_target_size(1_000_000, frac=0.1, eval_max_rows=100_000)
        == 100_000
    )
    assert (
        selection_sample_target_size(500_000, frac=0.1, eval_max_rows=100_000) == 50_000
    )
    assert selection_sample_target_size(10_000, frac=0.5, eval_max_rows=None) == 5_000


def test_stratified_mention_sample_preserves_both_classes() -> None:
    frame = _synthetic_frame(n_kb=500, n_neg=200)
    sub = stratified_mention_sample(
        frame,
        n_target=100,
        negative_label=NEGATIVE_LABEL,
        random_state=42,
    )
    assert len(sub) == 100
    assert (sub["entity"] == NEGATIVE_LABEL).any()
    assert (sub["entity"] != NEGATIVE_LABEL).any()


def test_draw_selection_sample_deterministic() -> None:
    frame = _synthetic_frame(n_kb=800, n_neg=300)
    cfg = ClusteringOptimizationConfig(
        frac=0.2,
        eval_max_rows=200,
        base_seed=7,
    )
    a = draw_selection_sample(frame, cfg, sample_index=0)
    b = draw_selection_sample(frame, cfg, sample_index=0)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_draw_selection_sample_distinct_bootstrap_indices() -> None:
    frame = _synthetic_frame(n_kb=2000, n_neg=800)
    cfg = ClusteringOptimizationConfig(
        frac=0.5,
        eval_max_rows=500,
        base_seed=99,
    )
    s0 = draw_selection_sample(frame, cfg, sample_index=0)
    s1 = draw_selection_sample(frame, cfg, sample_index=1)
    assert len(s0) == len(s1) == 500
    assert set(s0["pmid"].tolist()) != set(s1["pmid"].tolist())
