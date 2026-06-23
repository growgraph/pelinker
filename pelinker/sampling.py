"""Stratified mention-frame subsampling for model-selection evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pelinker.config import ClusteringOptimizationConfig


def selection_sample_target_size(
    n_rows: int,
    *,
    clustering_sample_rows: int | None,
) -> int:
    """Target row count for one selection draw: ``min(n_rows, clustering_sample_rows)`` or all rows."""
    if clustering_sample_rows is None:
        return n_rows
    return min(n_rows, clustering_sample_rows)


def _rng_from_state(random_state: int | np.random.Generator) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(int(random_state))


def _allocate_stratified_counts(
    n_target: int,
    n_neg: int,
    n_pos: int,
    n_total: int,
) -> tuple[int, int]:
    """Neg/pos draw sizes summing to ``n_target`` (each stratum >=1 when available).

    Proportional rounding, then min/max clamping, then while-loops to hit ``n_target``.
    """
    k_neg = min(max(1, round(n_target * n_neg / n_total)), n_neg)
    k_pos = min(max(1, n_target - k_neg), n_pos)
    k_neg = n_target - k_pos
    k_neg = max(1, min(k_neg, n_neg))
    k_pos = max(1, min(k_pos, n_pos))
    while k_neg + k_pos < n_target and (k_neg < n_neg or k_pos < n_pos):
        if k_neg < n_neg:
            k_neg += 1
        elif k_pos < n_pos:
            k_pos += 1
        else:
            break
    while k_neg + k_pos > n_target:
        if k_neg > 1 and k_neg >= k_pos:
            k_neg -= 1
        elif k_pos > 1:
            k_pos -= 1
        else:
            break
    return k_neg, k_pos


def stratified_mention_sample(
    frame: pd.DataFrame,
    *,
    n_target: int,
    negative_label: str,
    random_state: int | np.random.Generator,
) -> pd.DataFrame:
    """
    Stratified subsample by negative vs KB ``entity`` (preserve class proportions).

    Returns ``frame`` unchanged when ``len(frame) <= n_target``.
    """
    if n_target < 1:
        raise ValueError("n_target must be >= 1")
    if "entity" not in frame.columns:
        raise ValueError("frame must contain an 'entity' column")

    n = len(frame)
    if n <= n_target:
        return frame

    rng = _rng_from_state(random_state)
    entities = frame["entity"].astype(str)
    is_neg = entities == negative_label
    neg_idx = np.flatnonzero(is_neg.to_numpy())
    pos_idx = np.flatnonzero((~is_neg).to_numpy())

    if len(neg_idx) == 0 or len(pos_idx) == 0:
        chosen = rng.choice(n, size=n_target, replace=False)
        return frame.iloc[np.sort(chosen)].reset_index(drop=True)

    n_neg = len(neg_idx)
    n_pos = len(pos_idx)
    k_neg, k_pos = _allocate_stratified_counts(n_target, n_neg, n_pos, n)

    pick_neg = rng.choice(neg_idx, size=k_neg, replace=False)
    pick_pos = rng.choice(pos_idx, size=k_pos, replace=False)
    chosen = np.sort(np.concatenate([pick_neg, pick_pos]))
    return frame.iloc[chosen].reset_index(drop=True)


def cap_mentions_per_entity(
    frame: pd.DataFrame,
    *,
    max_mentions: int,
    negative_label: str,
    max_mentions_negative: int | None,
    random_state: int | np.random.Generator,
) -> pd.DataFrame:
    """
    Keep at most ``max_mentions`` rows per KB entity (seeded random subset).

    When ``max_mentions_negative`` is ``None``, rows with ``entity == negative_label`` are
    never capped. Otherwise negatives use that cap.
    """
    if max_mentions < 1:
        raise ValueError("max_mentions must be >= 1")
    if max_mentions_negative is not None and max_mentions_negative < 1:
        raise ValueError("max_mentions_negative must be >= 1 when provided")
    if "entity" not in frame.columns:
        raise ValueError("frame must contain an 'entity' column")
    if len(frame) == 0:
        return frame

    rng = _rng_from_state(random_state)
    work = frame.copy()
    work["_cap_key"] = rng.random(len(work))
    work["_entity_str"] = work["entity"].astype(str)
    neg_mask = work["_entity_str"] == negative_label

    capped_parts: list[pd.DataFrame] = []
    kb = work.loc[~neg_mask]
    if len(kb) > 0:
        capped_parts.append(
            kb.sort_values(["_entity_str", "_cap_key"], kind="mergesort")
            .groupby("_entity_str", as_index=False, sort=False)
            .head(max_mentions)
        )
    neg = work.loc[neg_mask]
    if len(neg) > 0:
        if max_mentions_negative is None:
            capped_parts.append(neg)
        else:
            capped_parts.append(
                neg.sort_values(["_entity_str", "_cap_key"], kind="mergesort")
                .groupby("_entity_str", as_index=False, sort=False)
                .head(max_mentions_negative)
            )

    if not capped_parts:
        return frame.iloc[0:0].copy()

    out = pd.concat(capped_parts, ignore_index=True)
    return out.drop(columns=["_cap_key", "_entity_str"]).reset_index(drop=True)


def draw_selection_sample(
    frame: pd.DataFrame,
    config: ClusteringOptimizationConfig,
    *,
    sample_index: int,
) -> pd.DataFrame:
    """Draw one stratified evaluation subsample for bootstrap ``sample_index``."""
    n_target = selection_sample_target_size(
        len(frame),
        clustering_sample_rows=config.clustering_sample_rows,
    )
    if n_target >= len(frame):
        return frame
    seed = config.base_seed + int(sample_index)
    return stratified_mention_sample(
        frame,
        n_target=n_target,
        negative_label=config.ambient_screener.negative_label,
        random_state=seed,
    )
