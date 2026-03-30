"""Tests for fusion candidate ranking helpers."""

import pathlib

import pytest

from pelinker.clustering_fusion_ranking import (
    singleton_items_by_dbcv_score,
    top_k_fusion_candidates_by_dbcv_proxy,
)


def test_singleton_items_by_dbcv_score_sorts_desc() -> None:
    files = [
        (pathlib.Path("/a"), "m1", "L0"),
        (pathlib.Path("/b"), "m2", "L1"),
    ]
    scores = {("m1", "L0"): 0.5, ("m2", "L1"): 0.9}
    rows = singleton_items_by_dbcv_score(files, scores)
    assert [t[3] for t in rows] == [0.9, 0.5]


def test_top_k_fusion_candidates_deduplicates_component_sets() -> None:
    items = [
        (pathlib.Path("/a"), "m1", "L0", 0.8),
        (pathlib.Path("/b"), "m2", "L1", 0.7),
        (pathlib.Path("/c"), "m3", "L2", 0.1),
    ]
    # Best pair sum: m1/L0 + m2/L1 = 1.5
    out = top_k_fusion_candidates_by_dbcv_proxy(items, order=2, k=2)
    assert len(out) >= 1
    assert out[0][3] == pytest.approx(1.5)


def test_top_k_fusion_candidates_short_circuit() -> None:
    assert top_k_fusion_candidates_by_dbcv_proxy([], order=2, k=1) == []
