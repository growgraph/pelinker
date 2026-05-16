"""Stratified mention-frame subsampling for model-selection evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pelinker.config import ClusteringOptimizationConfig


def selection_sample_target_size(
    n_rows: int,
    *,
    frac: float,
    eval_max_rows: int | None,
) -> int:
    """Target row count for one selection draw: ``min(int(n * frac), eval_max_rows)``."""
    by_frac = int(n_rows * frac)
    if eval_max_rows is None:
        return by_frac
    return min(by_frac, eval_max_rows)


def _rng_from_state(random_state: int | np.random.Generator) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(int(random_state))


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
    k_neg = max(1, round(n_target * n_neg / n))
    k_pos = max(1, n_target - k_neg)
    if k_neg + k_pos > n_target:
        if k_neg >= k_pos:
            k_neg = max(1, n_target - k_pos)
        else:
            k_pos = max(1, n_target - k_neg)
    k_neg = min(k_neg, n_neg)
    k_pos = min(k_pos, n_pos)
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

    pick_neg = rng.choice(neg_idx, size=k_neg, replace=False)
    pick_pos = rng.choice(pos_idx, size=k_pos, replace=False)
    chosen = np.sort(np.concatenate([pick_neg, pick_pos]))
    return frame.iloc[chosen].reset_index(drop=True)


def draw_selection_sample(
    frame: pd.DataFrame,
    config: ClusteringOptimizationConfig,
    *,
    sample_index: int,
) -> pd.DataFrame:
    """Draw one stratified evaluation subsample for bootstrap ``sample_index``."""
    n_target = selection_sample_target_size(
        len(frame),
        frac=config.frac,
        eval_max_rows=config.eval_max_rows,
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
