"""Shared outer ranking for clustering search (model selection / dim selection).

**MCS** (``min_cluster_size``): HDBSCAN hyperparameter on the *inner* grid.

**Two-level criterion** (same in model selection and dim selection):

1. **Inner (choose MCS)** — ``grid_objective=dbcv_ari_geomean`` on the MCS curve
   (clip DBCV and ARI at 0, take ``sqrt(dbcv * ari)``, smooth, then the paired
   one-standard-error rule).
2. **Outer (rank candidates)** — at each candidate's pooled MCS, combine mean DBCV
   and mean ARI with **the same formula**.

Using one scale-free formula at both levels is deliberate. The geometric mean's
ranking is invariant to rescaling either metric, so nothing has to be normalized
against the curve or against the leaderboard. That in turn makes ``outer_score``
independent of which *other* candidates happened to be in the run, and comparable
across runs — neither of which held while the outer level min–max normalized across
the leaderboard.

``best_score`` in flat summary rows remains **mean DBCV** (for DBCV heatmaps).
``outer_score`` is the DBCV+ARI combo used to pick winners.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from pelinker.clustering.grid import combine_clipped_geomean, geomean_std
from pelinker.reports.schema import ClusteringSearchSummaryRow

OUTER_SCORE_COL = "outer_score"
OUTER_SCORE_STD_COL = "outer_score_std"

MCS_DOC = (
    "MCS (min_cluster_size): HDBSCAN hyperparameter searched on an inner grid; "
    "the smallest cluster size HDBSCAN will form."
)

METRICS_TWO_LEVEL_DOC = {
    "mcs": MCS_DOC,
    "inner_min_cluster_size": (
        "grid_objective=dbcv_ari_geomean: clip mean DBCV and mean ARI at 0, take "
        "sqrt(dbcv*ari) per bootstrap sample, smooth, then pick the largest MCS within "
        "one paired standard error of the best."
    ),
    "outer_candidate_ranking": (
        "At the pooled consensus MCS, combine mean DBCV and mean ARI with the same "
        "clipped geometric mean used on the grid. Scale-free, so it does not depend on "
        "which other candidates were in the run. best_score stays mean DBCV for heatmaps; "
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
    Per-candidate combine: ``sqrt(max(dbcv, 0) * max(ari, 0))`` when both are finite.

    Falls back to whichever metric is finite. Depends only on this candidate's own
    numbers, so it is safe for mid-run leaderboards and resumed runs.
    """
    dbcv_f = float(dbcv)
    dbcv_ok = np.isfinite(dbcv_f)
    ari_ok = ari is not None and np.isfinite(float(ari))
    if dbcv_ok and ari_ok:
        score = float(
            combine_clipped_geomean(np.array([dbcv_f]), np.array([float(ari)]))[0]
        )
        std = float(
            geomean_std(
                np.array([dbcv_f]),
                np.array([float(dbcv_std)]),
                np.array([float(ari)]),
                np.array([float(ari_std)]),
            )[0]
        )
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
    dbcv_col: str = "best_score",
    dbcv_std_col: str = "best_score_std",
    ari_col: str = "ari",
    ari_std_col: str = "ari_std",
) -> pd.DataFrame:
    """
    Add ``outer_score`` / ``outer_score_std`` columns.

    Each row is scored from its own DBCV and ARI alone with the clipped geometric mean —
    the same formula the MCS grid uses. Nothing is normalized across rows, so adding or
    removing an unrelated candidate cannot reorder the others.
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

    both = np.isfinite(md) & np.isfinite(ma)
    means = np.where(both, combine_clipped_geomean(md, ma), np.nan)
    stds = np.where(both, geomean_std(md, sd, ma, sa), np.nan)
    # Fall back to whichever metric a row actually has, matching raw_outer_dbcv_ari_score.
    only_dbcv = np.isfinite(md) & ~np.isfinite(ma)
    only_ari = np.isfinite(ma) & ~np.isfinite(md)
    means = np.where(only_dbcv, md, np.where(only_ari, ma, means))
    stds = np.where(only_dbcv, sd, np.where(only_ari, sa, stds))

    out[OUTER_SCORE_COL] = means
    out[OUTER_SCORE_STD_COL] = stds
    return out


def pick_best_row(
    df: pd.DataFrame,
    *,
    score_col: str = OUTER_SCORE_COL,
    std_col: str = OUTER_SCORE_STD_COL,
    tie_break_cols: Sequence[str] = (),
) -> dict[str, object]:
    """
    Rank by outer DBCV+ARI score (descending), then lower std, then tie-break columns ascending.
    """
    if df.empty:
        raise ValueError("df must be a non-empty DataFrame")
    ranked_base = attach_outer_scores(df)
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
