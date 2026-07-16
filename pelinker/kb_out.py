"""KB-out catalog: provenance-bearing cluster entity ids, display names, and ambiguity indices."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pelinker.cluster_composition_viz import (
    HDBSCAN_NOISE_CLUSTER_ID,
    aggregate_cluster_entity_mass,
    cluster_entity_mass_summary,
    filter_emergent_assignments,
)
from pelinker.config import ClusterCompositionSnapshot, KBConfig
from pelinker.linker_cluster_training import (
    _disambiguate_consensus_names,
    consensus_cluster_names,
)

KB_OUT_SCHEMA = "pelinker.kb_out.v1"
_MAX_SPLIT_MENTIONS = 500
_SLUG_INVALID_RE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True, slots=True)
class KbOutNamingConfig:
    """Parameters for composite cluster display names."""

    min_fraction: float = 0.05
    top_n: int = 3
    ambiguity_min_capture: float = 0.10
    style: str = "weighted_dash"


@dataclass(frozen=True, slots=True)
class KbOutFitProvenance:
    """Fit-time parameters stored in the KB-out catalog."""

    min_cluster_size: int
    clustering_sample_index: int = 0
    seed: int | None = None


def kb_slug_from_config(kb_config: KBConfig | None, *, fallback: str = "kb") -> str:
    """Sanitized ``{name}_{version}`` slug for KB-out entity id prefixes."""
    if kb_config is None:
        return _sanitize_kb_slug(fallback)
    name = kb_config.name.strip() or fallback
    return _sanitize_kb_slug(f"{name}_{kb_config.version}")


def _sanitize_kb_slug(raw: str) -> str:
    slug = _SLUG_INVALID_RE.sub("_", raw.strip())
    slug = slug.strip("_")
    return slug or "kb"


def format_kb_out_entity_id(kb_slug: str, cluster_id: int) -> str:
    """Provenance prefix + zero-padded HDBSCAN cluster id."""
    return f"{kb_slug}::C{int(cluster_id):04d}"


def format_cluster_display_name(
    mass_frac: dict[str, float],
    *,
    min_fraction: float = 0.05,
    top_n: int = 3,
    pct_decimals: int = 0,
) -> str:
    """
    Composite display name from within-cluster entity fractions.

    Example: ``alpha-67--beta-33--gamma-15`` (top components above ``min_fraction``).
    Falls back to the single top entity when none meet the threshold.
    """
    if not mass_frac:
        return ""
    sorted_entities = sorted(mass_frac.items(), key=lambda kv: (-kv[1], kv[0]))
    significant = [e for e, f in sorted_entities if f >= min_fraction][:top_n]
    if not significant:
        significant = [sorted_entities[0][0]]

    parts: list[str] = []
    for ent in significant:
        frac = mass_frac[ent]
        if pct_decimals <= 0:
            pct = int(round(frac * 100.0))
        else:
            pct = round(frac * 100.0, pct_decimals)
        parts.append(f"{ent}-{pct}")
    return "--".join(parts)


def build_entity_membership(
    composition: ClusterCompositionSnapshot,
    cluster_id_to_entity_id: dict[int, str],
) -> dict[str, list[dict[str, Any]]]:
    """KB-in entity label → all emergent clusters where it has mass."""
    by_entity: dict[str, list[dict[str, Any]]] = {}
    for cid, mass_frac in composition.cluster_within_fraction.items():
        if int(cid) == HDBSCAN_NOISE_CLUSTER_ID:
            continue
        capture = composition.cluster_fraction_of_property_mass.get(int(cid), {})
        entity_id = cluster_id_to_entity_id.get(int(cid))
        for ent, within_frac in mass_frac.items():
            by_entity.setdefault(str(ent), []).append(
                {
                    "cluster_id": int(cid),
                    "entity_id": entity_id,
                    "within_cluster_fraction": float(within_frac),
                    "capture_fraction": float(capture.get(ent, 0.0)),
                }
            )
    for ent in by_entity:
        by_entity[ent] = sorted(
            by_entity[ent],
            key=lambda row: (-float(row["capture_fraction"]), int(row["cluster_id"])),
        )
    return by_entity


def find_ambiguous_entities(
    entity_membership: dict[str, list[dict[str, Any]]],
    *,
    min_capture: float,
) -> list[dict[str, Any]]:
    """Entities with capture ≥ ``min_capture`` in two or more clusters (polysemy)."""
    out: list[dict[str, Any]] = []
    for ent, rows in sorted(entity_membership.items()):
        significant = [r for r in rows if float(r["capture_fraction"]) >= min_capture]
        if len(significant) < 2:
            continue
        out.append(
            {
                "entity": ent,
                "clusters": significant,
                "n_clusters": len(significant),
            }
        )
    return out


def find_split_mentions(
    assignments: pd.DataFrame,
    *,
    min_clusters: int = 2,
    max_rows: int = _MAX_SPLIT_MENTIONS,
) -> list[dict[str, Any]]:
    """
    Mentions assigned to more than one emergent cluster (homonymy signal).

    Groups by ``(pmid, mention, a_abs)`` when ``a_abs`` is present; otherwise
    ``(pmid, mention)``.
    """
    required = {"pmid", "mention", "cluster"}
    if not required.issubset(assignments.columns):
        return []
    work = filter_emergent_assignments(assignments)
    if len(work) == 0:
        return []

    has_abs = "a_abs" in work.columns
    keys: list[tuple[str, ...]] = []
    cluster_lists: list[list[int]] = []
    entity_lists: list[list[str]] = []

    if has_abs:
        grouped = work.groupby(
            [
                work["pmid"].astype(str),
                work["mention"].astype(str),
                work["a_abs"].astype("Int64"),
            ],
            sort=False,
        )
    else:
        grouped = work.groupby(
            [work["pmid"].astype(str), work["mention"].astype(str)],
            sort=False,
        )

    for key, grp in grouped:
        clusters = sorted({int(c) for c in grp["cluster"].astype(int)})
        if len(clusters) < min_clusters:
            continue
        if isinstance(key, tuple):
            key_tuple = tuple(str(k) if k is not pd.NA else "" for k in key)
        else:
            key_tuple = (str(key),)
        keys.append(key_tuple)
        cluster_lists.append(clusters)
        entity_lists.append(sorted({str(e) for e in grp["entity"].astype(str)}))

    if not keys:
        return []

    order = sorted(
        range(len(keys)),
        key=lambda i: (-len(cluster_lists[i]), keys[i]),
    )[: int(max_rows)]

    out: list[dict[str, Any]] = []
    for i in order:
        key = keys[i]
        row: dict[str, Any] = {
            "pmid": key[0],
            "mention": key[1],
            "cluster_ids": cluster_lists[i],
            "entities": entity_lists[i],
            "n_clusters": len(cluster_lists[i]),
        }
        if has_abs and len(key) > 2:
            abs_val = key[2]
            row["a_abs"] = int(abs_val) if abs_val != "" else None
        out.append(row)
    return out


def build_kb_out_catalog(
    composition: ClusterCompositionSnapshot,
    assignments: pd.DataFrame,
    kb_in_labels_map: dict[str, str],
    *,
    kb_config: KBConfig | None,
    fit_provenance: KbOutFitProvenance,
    naming: KbOutNamingConfig | None = None,
    kb_in_labels_map_path: str | None = None,
) -> dict[str, Any]:
    """Build the canonical KB-out catalog (schema ``pelinker.kb_out.v1``)."""
    naming_cfg = naming or KbOutNamingConfig()
    kb_slug = kb_slug_from_config(kb_config)

    mass = aggregate_cluster_entity_mass(
        assignments, weight_by_entity=True, exclude_noise=True
    )
    cluster_totals: dict[int, float] = {}
    if not mass.empty:
        for cid, grp in mass.groupby("cluster", sort=False):
            cluster_totals[int(cid)] = float(grp["count"].sum())

    ordered_ids = sorted(
        cluster_totals.keys(),
        key=lambda c: (-cluster_totals[c], c),
    )

    short_labels = consensus_cluster_names(composition)
    raw_display: dict[int, str] = {}
    for cid in ordered_ids:
        if int(cid) == HDBSCAN_NOISE_CLUSTER_ID:
            continue
        mass_frac = composition.cluster_within_fraction.get(int(cid), {})
        raw_display[int(cid)] = format_cluster_display_name(
            mass_frac,
            min_fraction=naming_cfg.min_fraction,
            top_n=naming_cfg.top_n,
        )
    display_names = _disambiguate_consensus_names(raw_display)

    cluster_id_to_entity_id: dict[int, str] = {}
    clusters_out: list[dict[str, Any]] = []
    labels_map: dict[str, str] = {}

    emergent = filter_emergent_assignments(assignments)
    for cid in ordered_ids:
        if int(cid) == HDBSCAN_NOISE_CLUSTER_ID:
            continue
        icid = int(cid)
        entity_id = format_kb_out_entity_id(kb_slug, icid)
        cluster_id_to_entity_id[icid] = entity_id
        display_name = display_names.get(icid, str(icid))
        labels_map[entity_id] = display_name

        mass_frac = composition.cluster_within_fraction.get(icid, {})
        capture = composition.cluster_fraction_of_property_mass.get(icid, {})
        top_sorted = sorted(mass_frac.items(), key=lambda kv: (-kv[1], kv[0]))[
            : naming_cfg.top_n
        ]
        components = [
            {
                "entity": ent,
                "within_cluster_fraction": float(frac),
                "capture_fraction": float(capture.get(ent, 0.0)),
            }
            for ent, frac in top_sorted
        ]
        dominant_fraction = float(top_sorted[0][1]) if top_sorted else 0.0
        mention_count = int((emergent["cluster"].astype(int) == icid).sum())
        clusters_out.append(
            {
                "cluster_id": icid,
                "entity_id": entity_id,
                "display_name": display_name,
                "short_label": short_labels.get(icid, str(icid)),
                "weighted_mass": cluster_totals.get(icid, 0.0),
                "mention_count": mention_count,
                "dominant_entity_fraction": dominant_fraction,
                "components": components,
                "provenance": {
                    "hdbscan_cluster_id": icid,
                    "kb_slug": kb_slug,
                },
            }
        )

    entity_membership = build_entity_membership(composition, cluster_id_to_entity_id)
    ambiguous_entities = find_ambiguous_entities(
        entity_membership,
        min_capture=naming_cfg.ambiguity_min_capture,
    )
    split_mentions = find_split_mentions(assignments)

    summary = cluster_entity_mass_summary(assignments)
    kb_out_meta: dict[str, Any] = {
        "entity_count": len(labels_map),
    }
    if kb_config is not None:
        kb_out_meta.update(
            {
                "name": kb_config.name,
                "version": kb_config.version,
                "created_at": kb_config.created_at.isoformat(),
                "description": kb_config.description,
            }
        )

    kb_in_provenance: dict[str, Any] = {
        "entity_count": len(kb_in_labels_map),
        "labels_map": dict(kb_in_labels_map),
    }
    if kb_in_labels_map_path is not None:
        kb_in_provenance["labels_map_path"] = kb_in_labels_map_path
    if kb_config is not None:
        kb_in_provenance["name"] = kb_config.name
        kb_in_provenance["version"] = kb_config.version

    fit_prov: dict[str, Any] = {
        "min_cluster_size": int(fit_provenance.min_cluster_size),
        "clustering_sample_index": int(fit_provenance.clustering_sample_index),
    }
    if fit_provenance.seed is not None:
        fit_prov["seed"] = int(fit_provenance.seed)

    return {
        "schema": KB_OUT_SCHEMA,
        "kb_out": kb_out_meta,
        "provenance": {
            "kb_in": kb_in_provenance,
            "fit": fit_prov,
        },
        "naming": {
            "min_fraction": float(naming_cfg.min_fraction),
            "top_n": int(naming_cfg.top_n),
            "ambiguity_min_capture": float(naming_cfg.ambiguity_min_capture),
            "style": naming_cfg.style,
        },
        "n_emergent_clusters": int(summary["n_emergent_clusters"]),
        "n_noise_mentions": int(summary["n_noise_mentions"]),
        "noise_fraction": float(summary["noise_fraction"]),
        "clusters": clusters_out,
        "entity_membership": entity_membership,
        "ambiguous_entities": ambiguous_entities,
        "split_mentions": split_mentions,
        "labels_map": labels_map,
        "cluster_id_to_entity_id": {
            str(k): v for k, v in sorted(cluster_id_to_entity_id.items())
        },
    }


def project_to_legacy_emergent_clusters(catalog: dict[str, Any]) -> dict[str, Any]:
    """Project a KB-out catalog to legacy ``pelinker.emergent_clusters.v1`` shape."""
    fit_prov = catalog.get("provenance", {}).get("fit", {})
    clusters_legacy: list[dict[str, Any]] = []
    for cluster in catalog.get("clusters", []):
        top_entities = [
            {
                "entity": comp["entity"],
                "within_cluster_fraction": comp["within_cluster_fraction"],
                "capture_fraction_of_entity_mass": comp["capture_fraction"],
            }
            for comp in cluster.get("components", [])
        ]
        clusters_legacy.append(
            {
                "cluster_id": cluster["cluster_id"],
                "entity_id": cluster["entity_id"],
                "display_name": cluster["display_name"],
                "weighted_mass": cluster["weighted_mass"],
                "mention_count": cluster["mention_count"],
                "dominant_entity_fraction": cluster["dominant_entity_fraction"],
                "top_entities": top_entities,
            }
        )
    return {
        "schema": "pelinker.emergent_clusters.v1",
        "min_cluster_size": int(fit_prov.get("min_cluster_size", 0)),
        "n_emergent_clusters": int(catalog.get("n_emergent_clusters", 0)),
        "n_noise_mentions": int(catalog.get("n_noise_mentions", 0)),
        "noise_fraction": float(catalog.get("noise_fraction", 0.0)),
        "clusters": clusters_legacy,
    }


def cluster_labels_from_catalog(
    catalog: dict[str, Any],
    *,
    label_kind: str = "display",
) -> dict[int, str]:
    """Build ``cluster_id → label`` from a KB-out or legacy emergent catalog."""
    clusters = catalog.get("clusters", [])
    out: dict[int, str] = {}
    for cluster in clusters:
        cid = int(cluster["cluster_id"])
        if label_kind == "short":
            out[cid] = str(cluster.get("short_label", cluster.get("display_name", cid)))
        else:
            out[cid] = str(cluster.get("display_name", cid))
    return out
