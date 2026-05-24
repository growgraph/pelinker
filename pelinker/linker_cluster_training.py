"""Training-frame cluster composition and provisional entity→cluster maps for :class:`~pelinker.model.Linker`."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from pelinker.config import ClusterCompositionSnapshot


def _modal_cluster_deterministic(clusters: list[int]) -> int | None:
    """Most frequent cluster among ``clusters``, excluding HDBSCAN noise (-1); ties → smallest id."""
    vals = [int(c) for c in clusters if int(c) != -1]
    if not vals:
        return None
    cnt = Counter(vals)
    best_n = max(cnt.values())
    candidates = sorted(k for k, v in cnt.items() if v == best_n)
    return candidates[0]


def cluster_composition_from_training_frame(
    training: pd.DataFrame,
) -> ClusterCompositionSnapshot:
    """
    Aggregate mention counts per ``entity`` and per ``cluster`` from a fitted training frame.

    Rows are weighted equally (each row is one mention). Proportions in
    :attr:`ClusterCompositionSnapshot.cluster_within_fraction` are relative to each cluster’s
    total mass; :attr:`ClusterCompositionSnapshot.cluster_fraction_of_property_mass` is
    relative to each entity's global mass in this frame.
    """
    if "entity" not in training.columns or "cluster" not in training.columns:
        raise ValueError("training frame must contain 'entity' and 'cluster' columns")
    work = training[["entity", "cluster"]].copy()
    work["entity"] = work["entity"].astype(str)
    global_vc = work["entity"].value_counts()
    global_property_mass = {str(k): int(v) for k, v in global_vc.items()}
    cluster_within: dict[int, dict[str, float]] = {}
    cluster_capture: dict[int, dict[str, float]] = {}
    for cid, grp in work.groupby("cluster", sort=True):
        c = int(cid)
        counts = grp["entity"].value_counts()
        total = int(counts.sum())
        if total == 0:
            continue
        cluster_within[c] = {
            str(p): float(counts[p]) / float(total) for p in counts.index
        }
        cap: dict[str, float] = {}
        for p, cnt in counts.items():
            gp = global_property_mass[str(p)]
            if gp > 0:
                cap[str(p)] = float(int(cnt)) / float(gp)
        cluster_capture[c] = cap
    return ClusterCompositionSnapshot(
        global_property_mass=global_property_mass,
        cluster_within_fraction=cluster_within,
        cluster_fraction_of_property_mass=cluster_capture,
    )


def _is_near_uniform_mixture(mass_frac: dict[str, float], *, width_tol: float) -> bool:
    """Several properties with similar shares (flat admixture), not a single dominant."""
    if len(mass_frac) <= 1:
        return False
    vals = list(mass_frac.values())
    return (max(vals) - min(vals)) <= width_tol


def _singular_dominant_property(
    mass_frac: dict[str, float],
    *,
    min_share: float,
    min_gap: float,
) -> str | None:
    """Return the dominant entity label if one clearly leads, else ``None``."""
    if len(mass_frac) == 1:
        return next(iter(mass_frac))
    items = sorted(mass_frac.items(), key=lambda x: (-x[1], x[0]))
    top_p, top_v = items[0]
    second_v = items[1][1] if len(items) > 1 else 0.0
    if top_v >= min_share and (top_v - second_v) >= min_gap:
        return top_p
    return None


def _hyphen_join_properties(mass_frac: dict[str, float]) -> str:
    return "-".join(sorted(mass_frac.keys()))


def consensus_cluster_names(
    composition: ClusterCompositionSnapshot,
    *,
    uniform_width_tol: float = 0.15,
    dominance_min_share: float = 0.52,
    dominance_min_gap: float = 0.12,
    noise_cluster_label: str = "noise",
) -> dict[int, str]:
    """
    Derive a short human-readable name per cluster from within-cluster entity mixtures.

    * Single-property clusters use that property name.
    * Flat / near-uniform admixture uses hyphenated sorted property names.
    * Clear single dominant property uses that name; duplicate dominant names across clusters
      get ``_A``, ``_B``, … suffixes (stable order by cluster id).
    * Remaining mixed cases use hyphenated sorted names; collisions are disambiguated the same way.
    * Cluster ``-1`` (HDBSCAN noise) is named ``noise_cluster_label`` unless overridden by callers.
    """
    raw: dict[int, str] = {}
    for cid, mass_frac in composition.cluster_within_fraction.items():
        if cid == -1:
            raw[cid] = noise_cluster_label
            continue
        if not mass_frac:
            raw[cid] = str(cid)
            continue
        k = len(mass_frac)
        width_tol = min(uniform_width_tol, 0.5 / float(max(k, 1)))
        if k == 1:
            base = next(iter(mass_frac))
        elif _is_near_uniform_mixture(mass_frac, width_tol=width_tol):
            base = _hyphen_join_properties(mass_frac)
        else:
            dom = _singular_dominant_property(
                mass_frac,
                min_share=dominance_min_share,
                min_gap=dominance_min_gap,
            )
            base = dom if dom is not None else _hyphen_join_properties(mass_frac)
        raw[cid] = base
    return _disambiguate_consensus_names(raw)


def _disambiguate_consensus_names(names: dict[int, str]) -> dict[int, str]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for cid in sorted(names.keys()):
        buckets[names[cid]].append(cid)
    out: dict[int, str] = {}
    for name, cids in buckets.items():
        if len(cids) == 1:
            out[cids[0]] = name
            continue
        for i, cid in enumerate(sorted(cids)):
            suffix = chr(ord("A") + i)
            out[cid] = f"{name}_{suffix}"
    return out


def cluster_derived_labels_map(
    labels_map: dict[str, str],
    cluster_assignments: dict[str, int],
    composition: ClusterCompositionSnapshot,
    *,
    min_fraction: float = 0.05,
    top_n: int = 3,
    noise_label: str = "noise",
) -> dict[str, str]:
    """
    Build a new labels_map where each entity_id maps to a cluster-derived name.

    For each cluster, entities are ranked by their within-cluster fraction.  Only those
    with ``fraction >= min_fraction`` are kept; at most ``top_n`` survive.  If none
    survive the threshold the single top entity is used as a fallback.  The selected
    labels are joined with ``" / "`` to form the cluster name.

    HDBSCAN noise (cluster ``-1``) maps to ``noise_label``.

    Entity ids that have no cluster assignment in ``cluster_assignments`` are omitted.
    """
    cluster_names: dict[int, str] = {}
    for cid, mass_frac in composition.cluster_within_fraction.items():
        c = int(cid)
        if c == -1:
            cluster_names[c] = noise_label
            continue
        if not mass_frac:
            cluster_names[c] = str(c)
            continue
        sorted_entities = sorted(mass_frac.items(), key=lambda kv: (-kv[1], kv[0]))
        significant = [e for e, f in sorted_entities if f >= min_fraction][:top_n]
        if not significant:
            significant = [sorted_entities[0][0]]
        cluster_names[c] = " / ".join(significant)

    result: dict[str, str] = {}
    for entity_id in labels_map:
        cid = cluster_assignments.get(str(entity_id))
        if cid is None:
            continue
        name = cluster_names.get(int(cid))
        if name is None:
            continue
        result[str(entity_id)] = name
    return result


def provisional_cluster_assignments_from_training_frame(
    labels_map: dict[str, str],
    training: pd.DataFrame,
) -> dict[str, int]:
    """
    Map each ``entity_id`` to a single cluster id for ``predict`` compatibility.

    Heuristic: modal training cluster among rows whose ``entity`` equals
    ``labels_map[entity_id]``, ignoring -1. Interpretation of clusters is otherwise
    left to downstream analysis (see ``Linker.training_cluster_frame``).
    """
    out: dict[str, int] = {}
    if "entity" not in training.columns or "cluster" not in training.columns:
        return out
    for entity_id, label in labels_map.items():
        rows = training.loc[training["entity"] == label, "cluster"]
        if len(rows) == 0:
            continue
        mode = _modal_cluster_deterministic(rows.astype(int).tolist())
        if mode is None:
            continue
        out[str(entity_id)] = int(mode)
    return out
