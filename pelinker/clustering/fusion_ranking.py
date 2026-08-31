"""Rank single-embedding runs for fusion experiments (DBCV proxy before fused clustering)."""

from __future__ import annotations

import pathlib
from itertools import combinations


def singleton_items_by_dbcv_score(
    valid_files: list[tuple[pathlib.Path, str, str]],
    score_by_model_layer: dict[tuple[str, str], float],
) -> list[tuple[pathlib.Path, str, str, float]]:
    """
    One tuple per (model, layer) that has a score, with path and mean DBCV, best-first.
    """
    rows: list[tuple[pathlib.Path, str, str, float]] = []
    for fp, model, layer in valid_files:
        key = (model, layer)
        if key in score_by_model_layer:
            rows.append((fp, model, layer, float(score_by_model_layer[key])))
    rows.sort(key=lambda t: -t[3])
    return rows


def top_k_fusion_candidates_by_dbcv_proxy(
    items: list[tuple[pathlib.Path, str, str, float]],
    order: int,
    k: int,
) -> list[tuple[list[pathlib.Path], list[str], list[str], float]]:
    """
    Up to ``k`` distinct ``order``-tuples of distinct embeddings with highest sum of
    per-embedding DBCV scores (cheap proxy before running fused clustering).

    Each element is (
        paths in combination order,
        models,
        layers,
        sum_singleton_scores,
    ). Component identity is sorted lexicographically by (model, layer).
    """
    if order < 2 or k < 1 or len(items) < order:
        return []

    best: list[tuple[float, tuple[int, ...]]] = []
    indices = range(len(items))
    for idxs in combinations(indices, order):
        s = sum(items[i][3] for i in idxs)
        best.append((s, idxs))
    best.sort(key=lambda t: (-t[0], t[1]))

    seen: set[tuple[tuple[str, str], ...]] = set()
    out: list[tuple[list[pathlib.Path], list[str], list[str], float]] = []
    for sum_s, idxs in best:
        components = sorted((items[i][1], items[i][2]) for i in idxs)
        key = tuple(components)
        if key in seen:
            continue
        seen.add(key)
        idxs_list = list(idxs)
        paths = [items[i][0] for i in idxs_list]
        models = [items[i][1] for i in idxs_list]
        layers = [items[i][2] for i in idxs_list]
        out.append((paths, models, layers, sum_s))
        if len(out) >= k:
            break
    return out
