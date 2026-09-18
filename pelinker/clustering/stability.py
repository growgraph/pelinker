"""Cluster-identity stability across bootstrap resamples.

The submitted evaluation reported a cluster *count* with a spread (e.g. 63 ± 4.4) and
nothing about whether the clusters themselves are the same objects from draw to draw. A
count can be perfectly stable while every cluster's membership churns, so the count alone
does not support "these are the emergent predicate senses".

This module answers the identity question. Given per-bootstrap cluster assignments over a
shared row space, it matches clusters *between* pairs of draws with the Hungarian
algorithm — maximizing total member overlap (Jaccard) — and reports:

- **per-cluster persistence** — for each cluster of a reference draw, the mean Jaccard of
  its best match across the other draws, and how often it is matched at all;
- **matched-cluster counts** — how many clusters survive a Jaccard floor per pair;
- **cross-draw ARI** — agreement of two draws' partitions on the rows they share, which is
  matching-free and so a useful independent check on the matched numbers.

Only rows *shared* by two draws are compared: bootstrap draws overlap partially, and
scoring a cluster on rows one draw never saw would report resampling as instability.

Reading the numbers: Jaccard has a non-trivial chance floor when the number of clusters
is small — two equal clusters over the same rows overlap ~0.5 under random reassignment —
so a persistence figure is only interpretable against the cluster count it was measured
at, and `jaccard_floor` should sit well above chance for that count. `ari_shared` is
matching-free and does not share this failure mode, which is why both are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from pelinker.clustering.composition import HDBSCAN_NOISE_CLUSTER_ID

RowLabels = Mapping[Any, int]
"""``row_key -> cluster_id`` for one bootstrap draw (noise included; filtered here)."""


def _emergent(labels: RowLabels) -> dict[Any, int]:
    return {
        key: int(cluster)
        for key, cluster in labels.items()
        if int(cluster) != HDBSCAN_NOISE_CLUSTER_ID
    }


def _members_by_cluster(labels: Mapping[Any, int]) -> dict[int, set]:
    out: dict[int, set] = {}
    for key, cluster in labels.items():
        out.setdefault(int(cluster), set()).add(key)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class PairMatching:
    """Hungarian matching between two draws, restricted to their shared rows."""

    n_shared_rows: int
    n_clusters_a: int
    n_clusters_b: int
    matches: tuple[tuple[int, int, float], ...]
    """``(cluster_a, cluster_b, jaccard)`` for each matched pair."""
    ari_shared: float | None

    def matched_above(self, floor: float) -> int:
        return sum(1 for _, _, j in self.matches if j >= floor)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_shared_rows": self.n_shared_rows,
            "n_clusters_a": self.n_clusters_a,
            "n_clusters_b": self.n_clusters_b,
            "n_matched": len(self.matches),
            "mean_jaccard": (
                float(np.mean([j for _, _, j in self.matches]))
                if self.matches
                else None
            ),
            "ari_shared": self.ari_shared,
        }


def match_two_draws(labels_a: RowLabels, labels_b: RowLabels) -> PairMatching:
    """Optimal cluster correspondence between two draws by shared-member Jaccard."""
    a = _emergent(labels_a)
    b = _emergent(labels_b)
    shared = set(a) & set(b)

    clusters_a = _members_by_cluster({k: a[k] for k in shared})
    clusters_b = _members_by_cluster({k: b[k] for k in shared})
    ids_a = sorted(clusters_a)
    ids_b = sorted(clusters_b)

    if not ids_a or not ids_b:
        return PairMatching(
            n_shared_rows=len(shared),
            n_clusters_a=len(ids_a),
            n_clusters_b=len(ids_b),
            matches=(),
            ari_shared=None,
        )

    scores = np.zeros((len(ids_a), len(ids_b)), dtype=float)
    for i, ca in enumerate(ids_a):
        for j, cb in enumerate(ids_b):
            scores[i, j] = jaccard(clusters_a[ca], clusters_b[cb])

    # Hungarian on the negated overlap: maximize total Jaccard across the assignment.
    rows, cols = linear_sum_assignment(-scores)
    matches = tuple(
        (ids_a[i], ids_b[j], float(scores[i, j]))
        for i, j in zip(rows, cols)
        if scores[i, j] > 0.0
    )

    ordered = sorted(shared)
    ari = (
        float(
            adjusted_rand_score(
                [a[k] for k in ordered],
                [b[k] for k in ordered],
            )
        )
        if ordered
        else None
    )
    return PairMatching(
        n_shared_rows=len(shared),
        n_clusters_a=len(ids_a),
        n_clusters_b=len(ids_b),
        matches=matches,
        ari_shared=ari,
    )


@dataclass(frozen=True)
class StabilityReport:
    """Aggregate identity stability over all ordered draw pairs."""

    n_draws: int
    jaccard_floor: float
    cluster_counts: tuple[int, ...]
    pairs: tuple[dict[str, Any], ...]
    per_cluster: dict[int, dict[str, float]]
    """Reference-draw cluster id → ``{mean_jaccard, match_rate}`` across other draws."""

    def to_jsonable(self) -> dict[str, Any]:
        mean_j = [
            p["mean_jaccard"] for p in self.pairs if p["mean_jaccard"] is not None
        ]
        aris = [p["ari_shared"] for p in self.pairs if p["ari_shared"] is not None]
        persistent = [
            cid
            for cid, stats in self.per_cluster.items()
            if stats["mean_jaccard"] >= self.jaccard_floor
        ]
        return {
            "n_draws": self.n_draws,
            "jaccard_floor": self.jaccard_floor,
            "cluster_count_mean": float(np.mean(self.cluster_counts))
            if self.cluster_counts
            else None,
            "cluster_count_std": float(np.std(self.cluster_counts, ddof=1))
            if len(self.cluster_counts) > 1
            else None,
            "n_pairs": len(self.pairs),
            "mean_pair_jaccard": float(np.mean(mean_j)) if mean_j else None,
            "mean_cross_draw_ari": float(np.mean(aris)) if aris else None,
            "n_reference_clusters": len(self.per_cluster),
            "n_persistent_clusters": len(persistent),
            "persistent_fraction": (
                len(persistent) / len(self.per_cluster) if self.per_cluster else None
            ),
        }


def analyze_stability(
    draws: Sequence[RowLabels],
    *,
    jaccard_floor: float = 0.5,
    reference_index: int = 0,
) -> StabilityReport:
    """Match every draw pair and summarize how well cluster identity persists.

    Args:
        draws: One ``row_key -> cluster_id`` mapping per bootstrap draw.
        jaccard_floor: Overlap at or above which a cluster counts as the "same" cluster.
        reference_index: Draw whose clusters the per-cluster persistence is reported for.
    """
    if len(draws) < 2:
        raise ValueError("stability needs at least two draws")

    counts = tuple(len(_members_by_cluster(_emergent(d))) for d in draws)

    pairs: list[dict[str, Any]] = []
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            matching = match_two_draws(draws[i], draws[j])
            payload = matching.to_jsonable()
            payload["draw_a"] = i
            payload["draw_b"] = j
            payload["n_matched_above_floor"] = matching.matched_above(jaccard_floor)
            pairs.append(payload)

    reference = _emergent(draws[reference_index])
    ref_clusters = sorted(_members_by_cluster(reference))
    per_cluster: dict[int, dict[str, float]] = {}
    for cid in ref_clusters:
        jaccards: list[float] = []
        for k, other in enumerate(draws):
            if k == reference_index:
                continue
            matching = match_two_draws(draws[reference_index], other)
            best = [j for a, _, j in matching.matches if a == cid]
            jaccards.append(best[0] if best else 0.0)
        per_cluster[cid] = {
            "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
            "match_rate": (
                float(np.mean([j >= jaccard_floor for j in jaccards]))
                if jaccards
                else 0.0
            ),
        }

    return StabilityReport(
        n_draws=len(draws),
        jaccard_floor=jaccard_floor,
        cluster_counts=counts,
        pairs=tuple(pairs),
        per_cluster=per_cluster,
    )
