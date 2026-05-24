"""Entity-weighted cluster composition tables and emergent-cluster reporting."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from pelinker.config import ClusterCompositionSnapshot

ENTITY_WEIGHTING_INV_SQRT = "inv_sqrt_mention_count"
HDBSCAN_NOISE_CLUSTER_ID = -1
DEFAULT_MAX_CLUSTERS_FOR_PLOTS = 48
DEFAULT_MAX_ENTITIES_FOR_FLOW_PLOTS = 24


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
) -> list[int]:
    """Cluster ids ordered by descending total mass (optionally truncated)."""
    if mass.empty:
        return []
    totals = (
        mass.groupby("cluster", sort=False)["count"].sum().sort_values(ascending=False)
    )
    ids = [int(c) for c in totals.index.tolist()]
    if max_clusters is not None and max_clusters > 0:
        return ids[: int(max_clusters)]
    return ids


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
    """
    counts = aggregate_cluster_entity_mass(
        assignments,
        weight_by_entity=weight_by_entity,
        exclude_noise=exclude_noise,
    )
    if counts.empty:
        return counts
    keep_ids = top_cluster_ids_by_mass(counts, max_clusters=max_clusters)
    counts = counts.loc[counts["cluster"].isin(keep_ids)]
    return pd.concat(
        [
            _bundle_top_n_and_other(int(cid), grp, top_n=top_n)
            for cid, grp in counts.groupby("cluster", sort=True)
            if int(cid) in keep_ids
        ],
        ignore_index=True,
    )


def limit_composition_for_flow_plots(
    composition_df: pd.DataFrame,
    *,
    max_clusters: int | None = DEFAULT_MAX_CLUSTERS_FOR_PLOTS,
    max_entities: int | None = DEFAULT_MAX_ENTITIES_FOR_FLOW_PLOTS,
) -> pd.DataFrame:
    """
      Subset a long composition table for Sankey/bump charts.

      Keeps top clusters by mass and top entities by total mass (drops ``Other (...)`` rows
    from entity ranking, then re-adds per-cluster Other slices when needed).
    """
    if composition_df.empty:
        return composition_df
    work = composition_df.copy()
    work["cluster"] = work["cluster"].astype(int)
    work["entity"] = work["entity"].astype(str)

    cluster_totals = work.groupby("cluster", sort=False)["count"].sum()
    keep_clusters = top_cluster_ids_by_mass(
        cluster_totals.reset_index().rename(columns={"index": "cluster"}),
        max_clusters=max_clusters,
    )
    if not keep_clusters and max_clusters is not None:
        keep_clusters = top_cluster_ids_by_mass(
            cluster_totals.reset_index().rename(columns={"index": "cluster"}),
            max_clusters=None,
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


def build_emergent_clusters_catalog(
    composition: ClusterCompositionSnapshot,
    consensus_names: dict[int, str],
    assignments: pd.DataFrame,
    *,
    min_cluster_size: int,
    top_entities_per_cluster: int = 5,
    weight_by_entity: bool = True,
) -> dict[str, Any]:
    """
    Build JSON-serializable emergent-cluster catalog with stable cluster entity ids.

      Each cluster gets ``entity_id`` ``cluster:{id}``, display name, mass, and top entities
      with within-cluster fractions.
    """
    mass = aggregate_cluster_entity_mass(
        assignments, weight_by_entity=weight_by_entity, exclude_noise=True
    )
    cluster_totals: dict[int, float] = {}
    if not mass.empty:
        for cid, grp in mass.groupby("cluster", sort=False):
            cluster_totals[int(cid)] = float(grp["count"].sum())

    ordered_ids = sorted(
        cluster_totals.keys(),
        key=lambda c: (-cluster_totals[c], c),
    )

    clusters_out: list[dict[str, Any]] = []
    for cid in ordered_ids:
        if cid == HDBSCAN_NOISE_CLUSTER_ID:
            continue
        mass_frac = composition.cluster_within_fraction.get(cid, {})
        capture = composition.cluster_fraction_of_property_mass.get(cid, {})
        top_sorted = sorted(mass_frac.items(), key=lambda kv: (-kv[1], kv[0]))[
            :top_entities_per_cluster
        ]
        top_entities = [
            {
                "entity": ent,
                "within_cluster_fraction": float(frac),
                "capture_fraction_of_entity_mass": float(capture.get(ent, 0.0)),
            }
            for ent, frac in top_sorted
        ]
        dominant_fraction = float(top_sorted[0][1]) if top_sorted else 0.0
        emergent = filter_emergent_assignments(assignments)
        mention_count = int((emergent["cluster"].astype(int) == cid).sum())
        clusters_out.append(
            {
                "cluster_id": int(cid),
                "entity_id": f"cluster:{cid}",
                "display_name": consensus_names.get(cid, str(cid)),
                "weighted_mass": cluster_totals.get(cid, 0.0),
                "mention_count": mention_count,
                "dominant_entity_fraction": dominant_fraction,
                "top_entities": top_entities,
            }
        )

    summary = cluster_entity_mass_summary(assignments)
    return {
        "schema": "pelinker.emergent_clusters.v1",
        "min_cluster_size": int(min_cluster_size),
        "n_emergent_clusters": int(summary["n_emergent_clusters"]),
        "n_noise_mentions": int(summary["n_noise_mentions"]),
        "noise_fraction": float(summary["noise_fraction"]),
        "clusters": clusters_out,
    }
