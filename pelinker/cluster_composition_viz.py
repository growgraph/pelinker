"""Entity-weighted cluster composition tables and emergent-cluster reporting."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

ENTITY_WEIGHTING_INV_SQRT = "inv_sqrt_mention_count"
HDBSCAN_NOISE_CLUSTER_ID = -1
NOISE_CLUSTER_LABEL = "noise"
DEFAULT_MAX_CLUSTERS_FOR_PLOTS = 48
DEFAULT_MAX_ENTITIES_FOR_FLOW_PLOTS = 24
_CLUSTER_SCORE_PERCENTILES = (10, 25, 50, 75, 90)


def entity_mention_weights(entities: pd.Series) -> pd.Series:
    """Per-row weight ``1 / sqrt(n_mentions(entity))`` in the sample."""
    counts = entities.astype(str).value_counts()
    return entities.astype(str).map(lambda e: 1.0 / math.sqrt(float(counts[e])))


def is_emergent_cluster(cluster: object) -> bool:
    """True for HDBSCAN cluster ids other than noise (-1)."""
    try:
        return int(cluster) != HDBSCAN_NOISE_CLUSTER_ID
    except (TypeError, ValueError):
        return True


def filter_emergent_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    """Drop HDBSCAN noise rows (cluster ``-1``)."""
    if "cluster" not in assignments.columns:
        return assignments.copy()
    clusters = assignments["cluster"].astype(int)
    return assignments.loc[clusters != HDBSCAN_NOISE_CLUSTER_ID].copy()


def count_emergent_clusters(assignments: pd.DataFrame) -> int:
    """Number of distinct emergent cluster labels (excludes ``-1``)."""
    if "cluster" not in assignments.columns or len(assignments) == 0:
        return 0
    labels = assignments["cluster"].astype(int)
    emergent = labels[labels != HDBSCAN_NOISE_CLUSTER_ID]
    if len(emergent) == 0:
        return 0
    return int(emergent.nunique())


def aggregate_cluster_entity_mass(
    assignments: pd.DataFrame,
    *,
    weight_by_entity: bool = True,
    exclude_noise: bool = True,
) -> pd.DataFrame:
    """
    Long table of weighted mass per (cluster, entity).

    Columns: ``cluster`` (int), ``entity`` (str), ``count`` (float).
    """
    if "entity" not in assignments.columns or "cluster" not in assignments.columns:
        raise ValueError("assignments must contain 'entity' and 'cluster' columns")
    work = assignments[["entity", "cluster"]].copy()
    work["cluster"] = work["cluster"].astype(int)
    work["entity"] = work["entity"].astype(str)
    if exclude_noise:
        work = work.loc[work["cluster"] != HDBSCAN_NOISE_CLUSTER_ID]
    if len(work) == 0:
        return pd.DataFrame(columns=["cluster", "entity", "count"])
    if weight_by_entity:
        work["weight"] = entity_mention_weights(work["entity"])
    else:
        work["weight"] = 1.0
    return (
        work.groupby(["cluster", "entity"], sort=False)["weight"]
        .sum()
        .rename("count")
        .reset_index()
    )


def top_cluster_ids_by_mass(
    mass: pd.DataFrame,
    *,
    max_clusters: int | None,
    always_include: Sequence[int] | None = None,
) -> list[int]:
    """Cluster ids ordered by descending total mass (optionally truncated).

    ``always_include`` ids present in ``mass`` are appended after the truncated
    ranking (so noise ``-1`` is not dropped when capping emergent clusters).
    """
    if mass.empty:
        return []
    totals = (
        mass.groupby("cluster", sort=False)["count"].sum().sort_values(ascending=False)
    )
    present = {int(c) for c in totals.index.tolist()}
    force = [int(c) for c in (always_include or ()) if int(c) in present]
    force_set = set(force)
    ranked = [int(c) for c in totals.index.tolist() if int(c) not in force_set]
    if max_clusters is not None and max_clusters > 0:
        ranked = ranked[: int(max_clusters)]
    return ranked + force


def with_noise_cluster_label(cluster_labels: dict[int, str] | None) -> dict[int, str]:
    """Copy labels and ensure HDBSCAN noise maps to :data:`NOISE_CLUSTER_LABEL`."""
    out = dict(cluster_labels) if cluster_labels else {}
    out.setdefault(HDBSCAN_NOISE_CLUSTER_ID, NOISE_CLUSTER_LABEL)
    return out


def cluster_score_percentile_summary(
    assignments: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Percentiles of ``cluster_score`` overall, emergent-only, and noise rows."""
    empty = {f"p{p}": float("nan") for p in _CLUSTER_SCORE_PERCENTILES}
    if "cluster_score" not in assignments.columns or len(assignments) == 0:
        return {"overall": dict(empty), "emergent": dict(empty), "noise": dict(empty)}

    scores = pd.to_numeric(assignments["cluster_score"], errors="coerce")
    clusters = (
        assignments["cluster"].astype(int)
        if "cluster" in assignments.columns
        else pd.Series(dtype=int)
    )

    def _pcts(vals: pd.Series) -> dict[str, float]:
        finite = vals.to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return dict(empty)
        qs = np.percentile(finite, list(_CLUSTER_SCORE_PERCENTILES))
        return {
            f"p{p}": float(q)
            for p, q in zip(_CLUSTER_SCORE_PERCENTILES, qs, strict=True)
        }

    emergent_mask = clusters != HDBSCAN_NOISE_CLUSTER_ID
    noise_mask = clusters == HDBSCAN_NOISE_CLUSTER_ID
    return {
        "overall": _pcts(scores),
        "emergent": _pcts(scores.loc[emergent_mask]) if len(clusters) else dict(empty),
        "noise": _pcts(scores.loc[noise_mask]) if len(clusters) else dict(empty),
    }


def limit_entity_flow_for_plots(
    flow_df: pd.DataFrame,
    *,
    max_clusters: int | None = DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    max_entities: int | None = DEFAULT_MAX_ENTITIES_FOR_FLOW_PLOTS,
    min_within_cluster_fraction: float = 0.0,
) -> pd.DataFrame:
    """
    Subset a KB-in entity→cluster flow table for Sankey charts.

    Keeps top clusters by mass and top entities by total mass; drops the rest
    (no synthetic ``Other`` labels). Thin within-cluster edges below
    ``min_within_cluster_fraction`` are dropped.
    """
    if flow_df.empty:
        return flow_df
    work = flow_df.copy()
    work["cluster"] = work["cluster"].astype(int)
    work["entity"] = work["entity"].astype(str)
    work["count"] = work["count"].astype(float)

    always = (
        [HDBSCAN_NOISE_CLUSTER_ID]
        if HDBSCAN_NOISE_CLUSTER_ID in set(work["cluster"].astype(int))
        else None
    )
    keep_clusters = top_cluster_ids_by_mass(
        work, max_clusters=max_clusters, always_include=always
    )
    if keep_clusters:
        work = work.loc[work["cluster"].isin(keep_clusters)]
    if work.empty:
        return work

    if max_entities is not None and max_entities > 0:
        entity_mass = work.groupby("entity", sort=False)["count"].sum()
        top_entities = set(
            entity_mass.sort_values(ascending=False)
            .head(int(max_entities))
            .index.astype(str)
        )
        work = work.loc[work["entity"].isin(top_entities)]

    if min_within_cluster_fraction > 0.0 and not work.empty:
        cluster_totals = work.groupby("cluster", sort=False)["count"].transform("sum")
        work = work.loc[
            work["count"] / cluster_totals >= min_within_cluster_fraction
        ].reset_index(drop=True)

    return work.reset_index(drop=True)


def _bundle_top_n_and_other(
    cluster_id: int,
    group: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    group = group.sort_values(by="count", ascending=False)
    if len(group) <= top_n:
        return group
    top = group.head(top_n)
    other_count = float(group.iloc[top_n:]["count"].sum())
    n_other = len(group) - top_n
    other_row = pd.DataFrame(
        [
            {
                "cluster": cluster_id,
                "entity": f"Other ({n_other} terms)",
                "count": other_count,
            }
        ]
    )
    return pd.concat([top, other_row], ignore_index=True)


def build_cluster_composition_df(
    assignments: pd.DataFrame,
    *,
    top_n: int = 3,
    weight_by_entity: bool = True,
    exclude_noise: bool = True,
    max_clusters: int | None = None,
) -> pd.DataFrame:
    """
    Aggregate per-(cluster, entity) mass and keep top-N entities plus Other per cluster.

    When ``weight_by_entity`` is true, each mention row contributes
    ``1 / sqrt(n_mentions(entity))`` instead of unit weight.

    ``max_clusters`` keeps only the largest emergent clusters by total mass (for plots).
    When ``exclude_noise`` is false, noise (``-1``) is always retained if present.
    """
    counts = aggregate_cluster_entity_mass(
        assignments,
        weight_by_entity=weight_by_entity,
        exclude_noise=exclude_noise,
    )
    if counts.empty:
        return counts
    always = (
        [HDBSCAN_NOISE_CLUSTER_ID]
        if not exclude_noise
        and HDBSCAN_NOISE_CLUSTER_ID in set(counts["cluster"].astype(int))
        else None
    )
    keep_ids = top_cluster_ids_by_mass(
        counts, max_clusters=max_clusters, always_include=always
    )
    keep_set = set(keep_ids)
    counts = counts.loc[counts["cluster"].isin(keep_ids)]
    return pd.concat(
        [
            _bundle_top_n_and_other(int(cid), grp, top_n=top_n)
            for cid, grp in counts.groupby("cluster", sort=True)
            if int(cid) in keep_set
        ],
        ignore_index=True,
    )


def limit_composition_for_flow_plots(
    composition_df: pd.DataFrame,
    *,
    max_clusters: int | None = DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    max_entities: int | None = DEFAULT_MAX_ENTITIES_FOR_FLOW_PLOTS,
    min_within_cluster_fraction: float = 0.0,
) -> pd.DataFrame:
    """
      Subset a long composition table for Sankey/bump charts.

      Keeps top clusters by mass and top entities by total mass (drops ``Other (...)`` rows
    from entity ranking, then re-adds per-cluster Other slices when needed).

    When ``min_within_cluster_fraction`` > 0, entity slices below that share of their
    cluster's total mass are rolled into ``Other (...)``.
    """
    if composition_df.empty:
        return composition_df
    work = composition_df.copy()
    work["cluster"] = work["cluster"].astype(int)
    work["entity"] = work["entity"].astype(str)

    if min_within_cluster_fraction > 0.0:
        work = _apply_within_cluster_fraction_floor(
            work,
            min_within_cluster_fraction=min_within_cluster_fraction,
        )

    cluster_totals = work.groupby("cluster", sort=False)["count"].sum()
    mass_for_rank = cluster_totals.reset_index()
    if "cluster" not in mass_for_rank.columns:
        mass_for_rank = mass_for_rank.rename(
            columns={mass_for_rank.columns[0]: "cluster"}
        )
    always = (
        [HDBSCAN_NOISE_CLUSTER_ID]
        if HDBSCAN_NOISE_CLUSTER_ID in set(work["cluster"].astype(int))
        else None
    )
    keep_clusters = top_cluster_ids_by_mass(
        mass_for_rank,
        max_clusters=max_clusters,
        always_include=always,
    )
    if not keep_clusters and max_clusters is not None:
        keep_clusters = top_cluster_ids_by_mass(
            mass_for_rank,
            max_clusters=None,
            always_include=always,
        )
    work = work.loc[work["cluster"].isin(keep_clusters)]

    entity_rows = work[~work["entity"].str.startswith("Other (")]
    if max_entities is not None and max_entities > 0 and len(entity_rows) > 0:
        ent_mass = entity_rows.groupby("entity", sort=False)["count"].sum()
        top_entities = set(
            ent_mass.sort_values(ascending=False)
            .head(int(max_entities))
            .index.astype(str)
        )
        other_slices = work[work["entity"].str.startswith("Other (")]
        work = pd.concat(
            [
                entity_rows.loc[entity_rows["entity"].isin(top_entities)],
                other_slices,
            ],
            ignore_index=True,
        )
    return work


def _apply_within_cluster_fraction_floor(
    composition_df: pd.DataFrame,
    *,
    min_within_cluster_fraction: float,
) -> pd.DataFrame:
    """Roll entity slices below a within-cluster fraction into ``Other (...)``."""
    if composition_df.empty or min_within_cluster_fraction <= 0.0:
        return composition_df
    parts: list[pd.DataFrame] = []
    for cid, grp in composition_df.groupby("cluster", sort=False):
        cluster_id = int(cid)
        entity_rows = grp[~grp["entity"].astype(str).str.startswith("Other (")].copy()
        other_rows = grp[grp["entity"].astype(str).str.startswith("Other (")]
        other_mass = float(other_rows["count"].sum()) if len(other_rows) else 0.0
        total = float(entity_rows["count"].sum()) + other_mass
        if total <= 0.0:
            parts.append(grp)
            continue
        keep_mask = (
            entity_rows["count"].astype(float) / total >= min_within_cluster_fraction
        )
        kept = entity_rows.loc[keep_mask]
        dropped_mass = float(entity_rows.loc[~keep_mask, "count"].sum())
        rolled = other_mass + dropped_mass
        if rolled > 0.0:
            n_other = int((~keep_mask).sum()) + (
                int(len(other_rows)) if len(other_rows) else 0
            )
            other_row = pd.DataFrame(
                [
                    {
                        "cluster": cluster_id,
                        "entity": f"Other ({n_other} terms)",
                        "count": rolled,
                    }
                ]
            )
            if "cluster_label" in grp.columns:
                other_row["cluster_label"] = grp["cluster_label"].iloc[0]
            parts.append(pd.concat([kept, other_row], ignore_index=True))
        else:
            parts.append(kept)
    if not parts:
        return composition_df
    return pd.concat(parts, ignore_index=True)


def cluster_entity_mass_summary(assignments: pd.DataFrame) -> dict[str, int | float]:
    """Counts for fit logs and composition JSON metadata."""
    if "cluster" not in assignments.columns:
        return {
            "n_mention_rows": 0,
            "n_emergent_clusters": 0,
            "n_noise_mentions": 0,
            "noise_fraction": 0.0,
        }
    clusters = assignments["cluster"].astype(int)
    n_rows = int(len(assignments))
    n_noise = int((clusters == HDBSCAN_NOISE_CLUSTER_ID).sum())
    return {
        "n_mention_rows": n_rows,
        "n_emergent_clusters": count_emergent_clusters(assignments),
        "n_noise_mentions": n_noise,
        "noise_fraction": float(n_noise) / float(n_rows) if n_rows > 0 else 0.0,
    }
