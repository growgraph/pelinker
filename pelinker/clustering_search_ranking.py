"""Shared outer ranking for clustering search (model selection / dim selection).

**MCS** (``min_cluster_size``): HDBSCAN hyperparameter on the *inner* grid.

**Two-level criterion** (same in model selection and dim selection):

1. **Inner (choose MCS)** — ``grid_objective=dbcv_ari_mean_minmax`` on the MCS
   curve (min–max normalize mean DBCV and mean ARI, average, smooth, left plateau).
2. **Outer (rank candidates)** — at each candidate's pooled MCS, take mean DBCV
   and mean ARI, then combine with the same DBCV+ARI pooling used on the grid
   (``dbcv_ari_mean_minmax`` across the leaderboard, or a resume-safe raw average).

``best_score`` in flat summary rows remains **mean DBCV** (for DBCV heatmaps).
``outer_score`` is the DBCV+ARI combo used to pick winners.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from pelinker.clustering_grid import combine_two_metric_series
from pelinker.reporting import ClusteringSearchSummaryRow

OUTER_SCORE_COL = "outer_score"
OUTER_SCORE_STD_COL = "outer_score_std"

MCS_DOC = (
    "MCS (min_cluster_size): HDBSCAN hyperparameter searched on an inner grid; "
    "the smallest cluster size HDBSCAN will form."
)

METRICS_TWO_LEVEL_DOC = {
    "mcs": MCS_DOC,
    "inner_min_cluster_size": (
        "grid_objective=dbcv_ari_mean_minmax: min-max normalize mean DBCV and mean ARI "
        "on the MCS curve, average them, smooth, and pick the left plateau."
    ),
    "outer_candidate_ranking": (
        "At the pooled consensus MCS, combine mean DBCV and mean ARI with the same "
        "DBCV+ARI pooling (dbcv_ari_mean_minmax across candidates, or raw 0.5·(DBCV+ARI) "
        "per row for resume-safe proxies). best_score stays mean DBCV for heatmaps; "
        "outer_score ranks winners."
    ),
}


def raw_outer_dbcv_ari_score(
    dbcv: float,
    ari: float | None,
    *,
    dbcv_std: float = 0.0,
    ari_std: float = 0.0,
) -> tuple[float, float]:
    """
    Resume-safe per-candidate combine: ``0.5 * (dbcv + ari)`` when both finite.

    Falls back to whichever metric is finite. Used for fusion proxies and mid-run
    progress when cross-leaderboard min-max is not appropriate.
    """
    dbcv_f = float(dbcv)
    dbcv_ok = np.isfinite(dbcv_f)
    ari_ok = ari is not None and np.isfinite(float(ari))
    if dbcv_ok and ari_ok:
        ari_f = float(ari)
        score = 0.5 * (dbcv_f + ari_f)
        std = float(np.sqrt((float(dbcv_std) ** 2 + float(ari_std) ** 2) / 4.0))
        return score, std
    if dbcv_ok:
        return dbcv_f, float(dbcv_std)
    if ari_ok:
        return float(ari), float(ari_std)
    return float("nan"), float("nan")


def outer_score_from_summary_row(row: ClusteringSearchSummaryRow) -> float:
    """Raw DBCV+ARI outer score from a search summary row."""
    ari = None if row.ari is None else float(row.ari.mean)
    ari_std = 0.0 if row.ari is None else float(row.ari.std)
    score, _ = raw_outer_dbcv_ari_score(
        float(row.dbcv.mean),
        ari,
        dbcv_std=float(row.dbcv.std),
        ari_std=ari_std,
    )
    return float(score)


def attach_outer_scores(
    df: pd.DataFrame,
    *,
    use_minmax: bool = True,
    dbcv_col: str = "best_score",
    dbcv_std_col: str = "best_score_std",
    ari_col: str = "ari",
    ari_std_col: str = "ari_std",
) -> pd.DataFrame:
    """
    Add ``outer_score`` / ``outer_score_std`` columns.

    When ``use_minmax`` is true (default), min–max normalize DBCV and ARI across
    rows then average — same pooling as ``dbcv_ari_mean_minmax`` on an MCS grid.
    When false, use the resume-safe raw average per row.
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        out[OUTER_SCORE_COL] = pd.Series(dtype=np.float64)
        out[OUTER_SCORE_STD_COL] = pd.Series(dtype=np.float64)
        return out

    if dbcv_col not in out.columns:
        raise ValueError(f"missing DBCV column {dbcv_col!r}")

    md = pd.to_numeric(out[dbcv_col], errors="coerce").to_numpy(dtype=np.float64)
    sd = (
        pd.to_numeric(out[dbcv_std_col], errors="coerce").to_numpy(dtype=np.float64)
        if dbcv_std_col in out.columns
        else np.zeros(n, dtype=np.float64)
    )
    sd = np.nan_to_num(sd, nan=0.0)

    if ari_col in out.columns:
        ma = pd.to_numeric(out[ari_col], errors="coerce").to_numpy(dtype=np.float64)
        sa = (
            pd.to_numeric(out[ari_std_col], errors="coerce").to_numpy(dtype=np.float64)
            if ari_std_col in out.columns
            else np.zeros(n, dtype=np.float64)
        )
        sa = np.nan_to_num(sa, nan=0.0)
    else:
        ma = np.full(n, np.nan, dtype=np.float64)
        sa = np.zeros(n, dtype=np.float64)

    cd = np.ones(n, dtype=np.float64)
    ca = np.ones(n, dtype=np.float64)
    # Treat missing ARI counts as 0 so combine falls back to DBCV-only.
    ca = np.where(np.isfinite(ma), 1.0, 0.0)
    cd = np.where(np.isfinite(md), 1.0, 0.0)

    means, stds, _counts = combine_two_metric_series(
        md, sd, cd, ma, sa, ca, use_minmax=use_minmax
    )
    out[OUTER_SCORE_COL] = means
    out[OUTER_SCORE_STD_COL] = stds
    return out


def pick_best_row(
    df: pd.DataFrame,
    *,
    score_col: str = OUTER_SCORE_COL,
    std_col: str = OUTER_SCORE_STD_COL,
    tie_break_cols: Sequence[str] = (),
    use_minmax: bool = True,
) -> dict[str, object]:
    """
    Rank by outer DBCV+ARI score (descending), then lower std, then tie-break columns ascending.
    """
    if df.empty:
        raise ValueError("df must be a non-empty DataFrame")
    ranked_base = attach_outer_scores(df, use_minmax=use_minmax)
    if score_col not in ranked_base.columns:
        raise ValueError(f"missing score column {score_col!r}")

    by = [score_col]
    ascending = [False]
    if std_col in ranked_base.columns:
        by.append(std_col)
        ascending.append(True)
    for col in tie_break_cols:
        if col in ranked_base.columns:
            by.append(col)
            ascending.append(True)

    ranked = ranked_base.sort_values(by=by, ascending=ascending, kind="mergesort")
    row = ranked.iloc[0]
    return {str(k): row[k] for k in ranked.columns}
