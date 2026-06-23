"""Stratified selection draws for model-selection evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pelinker.config import ClusteringOptimizationConfig
from pelinker.onto import NEGATIVE_LABEL
from pelinker.sampling import (
    cap_mentions_per_entity,
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


def test_selection_sample_target_size_clustering_sample_rows() -> None:
    assert (
        selection_sample_target_size(1_000_000, clustering_sample_rows=100_000)
        == 100_000
    )
    assert (
        selection_sample_target_size(500_000, clustering_sample_rows=100_000) == 100_000
    )
    assert selection_sample_target_size(10_000, clustering_sample_rows=None) == 10_000


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
        clustering_sample_rows=200,
        base_seed=7,
    )
    a = draw_selection_sample(frame, cfg, sample_index=0)
    b = draw_selection_sample(frame, cfg, sample_index=0)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_draw_selection_sample_distinct_bootstrap_indices() -> None:
    frame = _synthetic_frame(n_kb=2000, n_neg=800)
    cfg = ClusteringOptimizationConfig(
        clustering_sample_rows=500,
        base_seed=99,
    )
    s0 = draw_selection_sample(frame, cfg, sample_index=0)
    s1 = draw_selection_sample(frame, cfg, sample_index=1)
    assert len(s0) == len(s1) == 500
    assert set(s0["pmid"].tolist()) != set(s1["pmid"].tolist())


def _multi_entity_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows: list[dict[str, object]] = []
    for ent, n in [("a", 30), ("b", 5), (NEGATIVE_LABEL, 50)]:
        for i in range(n):
            rows.append(
                {
                    "pmid": f"{ent}_{i}",
                    "entity": ent,
                    "mention": f"m{i}",
                    "embed": rng.standard_normal(4).astype(np.float32),
                }
            )
    return pd.DataFrame(rows)


def test_cap_mentions_per_entity_truncates_heavy_entity() -> None:
    frame = _multi_entity_frame()
    capped = cap_mentions_per_entity(
        frame,
        max_mentions=10,
        negative_label=NEGATIVE_LABEL,
        max_mentions_negative=None,
        random_state=42,
    )
    assert capped.groupby("entity").size()["a"] == 10
    assert capped.groupby("entity").size()["b"] == 5
    assert capped.groupby("entity").size()[NEGATIVE_LABEL] == 50


def test_cap_mentions_per_entity_negative_cap() -> None:
    frame = _multi_entity_frame()
    capped = cap_mentions_per_entity(
        frame,
        max_mentions=10,
        negative_label=NEGATIVE_LABEL,
        max_mentions_negative=8,
        random_state=42,
    )
    assert capped.groupby("entity").size()[NEGATIVE_LABEL] == 8


def test_cap_mentions_per_entity_reproducible() -> None:
    frame = _multi_entity_frame()
    a = cap_mentions_per_entity(
        frame,
        max_mentions=10,
        negative_label=NEGATIVE_LABEL,
        max_mentions_negative=None,
        random_state=7,
    )
    b = cap_mentions_per_entity(
        frame,
        max_mentions=10,
        negative_label=NEGATIVE_LABEL,
        max_mentions_negative=None,
        random_state=7,
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
