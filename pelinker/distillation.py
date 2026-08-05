"""Held-out fidelity of the compact entity head against its HDBSCAN teacher.

Why this exists
---------------
Compact fit distills HDBSCAN into an MLP: the clusterer is fit on the clustering
subsample, ``approximate_predict`` extends labels to the full manifold, an MLP is
trained on those labels, and then the clusterer is **discarded**
(``Linker.clusterer = None``). Nothing about the student ever reached the fit report,
and the student was trained on 100 % of the teacher's labels — so even an in-sample
agreement number would have been circular.

This module holds out a slice *before* the head is trained and scores the student
against the teacher on it.

Three things it measures that a naive agreement number hides
------------------------------------------------------------
1. **Exact vs approximate teacher labels, separately.** Rows inside the clustering
   subsample carry HDBSCAN's own labels; every other row carries an
   ``approximate_predict`` guess. Agreement against the second kind partly measures how
   well the student imitates the teacher's *extrapolation errors*. Pooling the two hides
   which is degrading.
2. **Forced noise.** The head is trained only on ``cluster != -1``
   (:func:`~pelinker.entity_head._non_noise_xy`), so it *structurally cannot* emit noise:
   every mention the teacher would have abstained on gets assigned some cluster. The
   abstain responsibility silently moves to the screeners and ``thr_score``, so we report
   what fraction of teacher-noise rows get forced and with what confidence.
3. **Score comparability.** The teacher's score is HDBSCAN soft membership; the
   student's is ``max(predict_proba)``. ``thr_score`` is applied to both as if they were
   the same quantity. Emit rates at a reference threshold say whether that holds.

The holdout is grouped by ``pmid`` by default: mentions from one document are strongly
correlated, so a plain row shuffle leaks and reports an optimistic number.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from pelinker.config import DistillationGateConfig
from pelinker.entity_head import EntityHead
from pelinker.linker_cluster_training import cluster_composition_from_training_frame

logger = logging.getLogger(__name__)

HoldoutGrouping = Literal["pmid", "row"]
DEFAULT_GROUP_COLUMN = "pmid"


@dataclass(frozen=True)
class HoldoutSplit:
    """Positional train/holdout indices into the full-manifold row order."""

    train_idx: np.ndarray
    holdout_idx: np.ndarray
    grouping: HoldoutGrouping
    n_groups: int
    """Distinct groups on the holdout side (equal to ``len(holdout_idx)`` for ``row``)."""

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train_idx, self.holdout_idx)
        if overlap.size:
            raise ValueError(
                f"train and holdout indices overlap on {overlap.size} row(s)"
            )


def grouped_holdout_split(
    frame: pd.DataFrame,
    *,
    holdout_fraction: float,
    random_state: int,
    group_col: str = DEFAULT_GROUP_COLUMN,
) -> HoldoutSplit:
    """Split rows into train/holdout, keeping every ``group_col`` value on one side.

    Falls back to a row-level split (recorded as ``grouping="row"``) when the column is
    absent, or when grouping cannot produce a non-empty split — e.g. a corpus whose
    mentions all share one ``pmid``. The fallback is reported, never silent, because a
    row-level split on correlated mentions yields an optimistic agreement number.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    n = len(frame)
    if n < 2:
        raise ValueError(f"need at least 2 rows to split, got {n}")

    rng = np.random.default_rng(random_state)
    target = max(1, int(round(holdout_fraction * n)))

    if group_col in frame.columns:
        groups = frame[group_col].astype(str).to_numpy()
        unique = np.unique(groups)
        if len(unique) >= 2:
            order = rng.permutation(len(unique))
            counts = {g: int(np.count_nonzero(groups == g)) for g in unique}
            chosen: list[str] = []
            taken = 0
            for gi in order:
                g = unique[gi]
                # Stop before swallowing every group; train must stay non-empty.
                if taken >= target and chosen:
                    break
                if len(chosen) == len(unique) - 1:
                    break
                chosen.append(str(g))
                taken += counts[g]
            hold_mask = np.isin(groups, chosen)
            if hold_mask.any() and not hold_mask.all():
                return HoldoutSplit(
                    train_idx=np.flatnonzero(~hold_mask),
                    holdout_idx=np.flatnonzero(hold_mask),
                    grouping="pmid",
                    n_groups=len(chosen),
                )
        logger.warning(
            "Grouped holdout on %r not possible (%d distinct group(s) across %d rows); "
            "falling back to a row-level split. Correlated mentions may inflate the "
            "measured agreement.",
            group_col,
            len(unique),
            n,
        )
    else:
        logger.warning(
            "Mention frame has no %r column; falling back to a row-level holdout split. "
            "Correlated mentions may inflate the measured agreement.",
            group_col,
        )

    perm = rng.permutation(n)
    hold = np.sort(perm[:target])
    train = np.sort(perm[target:])
    if train.size == 0:  # pathological holdout_fraction on a tiny frame
        train, hold = hold[:1], hold[1:]
    return HoldoutSplit(
        train_idx=train,
        holdout_idx=hold,
        grouping="row",
        n_groups=int(hold.size),
    )


def cluster_to_entity_map(
    entities: Sequence[str] | np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[int, str]:
    """Majority KB entity per emergent cluster (noise excluded).

    Reuses :func:`~pelinker.linker_cluster_training.cluster_composition_from_training_frame`
    so the student/teacher comparison resolves clusters exactly the way the fitted linker
    does, rather than via a parallel majority-vote implementation.
    """
    frame = pd.DataFrame(
        {
            "entity": np.asarray(entities).astype(str),
            "cluster": np.asarray(cluster_labels, dtype=np.int64),
        }
    )
    frame = frame.loc[frame["cluster"] != -1]
    if frame.empty:
        return {}
    composition = cluster_composition_from_training_frame(frame)
    out: dict[int, str] = {}
    for cid, mixture in composition.cluster_within_fraction.items():
        if not mixture:
            continue
        # Ties broken by entity name so the map is deterministic across runs.
        out[int(cid)] = max(sorted(mixture), key=lambda e: mixture[e])
    return out


def labels_to_entities(
    cluster_labels: np.ndarray, mapping: dict[int, str], *, noise_label: str
) -> np.ndarray:
    """Resolve cluster ids to entity ids; noise and unmapped ids become ``noise_label``."""
    return np.asarray(
        [
            mapping.get(int(c), noise_label) if int(c) != -1 else noise_label
            for c in np.asarray(cluster_labels, dtype=np.int64).ravel()
        ],
        dtype=object,
    )


@dataclass(frozen=True)
class DistillationFidelityMetrics:
    """Held-out agreement between the compact entity head and its HDBSCAN teacher."""

    student_kind: str
    holdout_grouping: HoldoutGrouping
    n_holdout: int
    n_groups: int

    entity_agreement: float
    """Share of holdout rows where student and teacher resolve to the same entity id.

    Computed over rows the **teacher** assigned to a real cluster; teacher-noise rows are
    counted by :attr:`noise_forced_fraction` instead, since the student cannot emit noise
    and would fail every one of them by construction.
    """
    entity_agreement_exact: float | None
    """Agreement restricted to holdout rows carrying HDBSCAN's own labels."""
    entity_agreement_approx: float | None
    """Agreement on rows whose teacher label came from ``approximate_predict``.

    Materially below :attr:`entity_agreement_exact` means the student is chasing the
    teacher's extrapolation rather than its clustering.
    """
    n_holdout_exact: int
    n_holdout_approx: int

    ari_vs_teacher: float | None
    score_corr_vs_teacher: float | None
    """Pearson correlation of student and teacher scores; low means ``thr_score`` does
    not mean the same thing on both paths."""

    noise_forced_fraction: float
    """Share of holdout rows the teacher called noise that the student assigns anyway.

    Always 1.0 while the head trains on non-noise labels only; the useful signal is the
    accompanying score distribution — if those scores sit below ``thr_score`` the abstain
    survives, otherwise it was lost."""
    n_teacher_noise: int
    forced_noise_score_median: float | None
    forced_noise_score_p90: float | None

    emit_rate_student: float
    emit_rate_teacher: float
    emit_rate_threshold: float
    """Reference ``thr_score`` the two emit rates were measured at."""

    def emit_rate_relative_delta(self) -> float:
        """|student - teacher| / teacher, or 0.0 when the teacher emits nothing."""
        if self.emit_rate_teacher <= 0.0:
            return 0.0
        return (
            abs(self.emit_rate_student - self.emit_rate_teacher)
            / self.emit_rate_teacher
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "student_kind": self.student_kind,
            "holdout_grouping": self.holdout_grouping,
            "n_holdout": int(self.n_holdout),
            "n_groups": int(self.n_groups),
            "entity_agreement": _f(self.entity_agreement),
            "entity_agreement_exact": _f(self.entity_agreement_exact),
            "entity_agreement_approx": _f(self.entity_agreement_approx),
            "n_holdout_exact": int(self.n_holdout_exact),
            "n_holdout_approx": int(self.n_holdout_approx),
            "ari_vs_teacher": _f(self.ari_vs_teacher),
            "score_corr_vs_teacher": _f(self.score_corr_vs_teacher),
            "noise_forced_fraction": _f(self.noise_forced_fraction),
            "n_teacher_noise": int(self.n_teacher_noise),
            "forced_noise_score_median": _f(self.forced_noise_score_median),
            "forced_noise_score_p90": _f(self.forced_noise_score_p90),
            "emit_rate_student": _f(self.emit_rate_student),
            "emit_rate_teacher": _f(self.emit_rate_teacher),
            "emit_rate_threshold": _f(self.emit_rate_threshold),
            "emit_rate_relative_delta": _f(self.emit_rate_relative_delta()),
        }


def _f(x: float | None) -> float | None:
    if x is None:
        return None
    v = float(x)
    return None if (np.isnan(v) or np.isinf(v)) else v


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_distillation_fidelity(
    *,
    head: EntityHead,
    umap_holdout: np.ndarray,
    teacher_labels: np.ndarray,
    teacher_scores: np.ndarray,
    exact_label_mask: np.ndarray,
    cluster_entity_map: dict[int, str],
    grouping: HoldoutGrouping,
    n_groups: int,
    emit_rate_threshold: float,
    noise_label: str,
) -> DistillationFidelityMetrics:
    """Score ``head`` against its teacher on a held-out slice.

    Args:
        umap_holdout: Clustering-space coordinates of the holdout rows.
        teacher_labels: HDBSCAN cluster ids for those rows (``-1`` = noise).
        teacher_scores: HDBSCAN soft membership for those rows.
        exact_label_mask: True where ``teacher_labels`` came from HDBSCAN's own fit
            rather than ``approximate_predict``.
        cluster_entity_map: Cluster → entity map built on the **training** rows only.
    """
    student_labels, student_scores = head.predict(umap_holdout)
    teacher_labels = np.asarray(teacher_labels, dtype=np.int64).ravel()
    teacher_scores = np.asarray(teacher_scores, dtype=np.float64).ravel()
    exact_label_mask = np.asarray(exact_label_mask, dtype=bool).ravel()

    n_holdout = int(len(teacher_labels))
    teacher_ent = labels_to_entities(
        teacher_labels, cluster_entity_map, noise_label=noise_label
    )
    student_ent = labels_to_entities(
        student_labels, cluster_entity_map, noise_label=noise_label
    )

    scored = teacher_ent != noise_label
    agree = student_ent == teacher_ent

    def _rate(mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        return float(np.mean(agree[mask]))

    overall = _rate(scored)
    exact_mask = scored & exact_label_mask
    approx_mask = scored & ~exact_label_mask

    ari: float | None = None
    if scored.sum() >= 2 and len(np.unique(teacher_labels[scored])) >= 2:
        ari = float(adjusted_rand_score(teacher_labels[scored], student_labels[scored]))

    noise_mask = teacher_labels == -1
    n_noise = int(noise_mask.sum())
    forced_scores = student_scores[noise_mask]
    return DistillationFidelityMetrics(
        student_kind=head.kind,
        holdout_grouping=grouping,
        n_holdout=n_holdout,
        n_groups=int(n_groups),
        entity_agreement=float("nan") if overall is None else overall,
        entity_agreement_exact=_rate(exact_mask),
        entity_agreement_approx=_rate(approx_mask),
        n_holdout_exact=int(exact_mask.sum()),
        n_holdout_approx=int(approx_mask.sum()),
        ari_vs_teacher=ari,
        score_corr_vs_teacher=_safe_corr(student_scores, teacher_scores),
        # The head cannot emit -1, so every teacher-noise row is forced by construction;
        # kept explicit so the number is asserted rather than assumed.
        noise_forced_fraction=(
            0.0 if n_noise == 0 else float(np.mean(student_labels[noise_mask] != -1))
        ),
        n_teacher_noise=n_noise,
        forced_noise_score_median=(
            None if n_noise == 0 else float(np.median(forced_scores))
        ),
        forced_noise_score_p90=(
            None if n_noise == 0 else float(np.percentile(forced_scores, 90))
        ),
        emit_rate_student=float(np.mean(student_scores >= emit_rate_threshold)),
        emit_rate_teacher=float(
            np.mean((teacher_labels != -1) & (teacher_scores >= emit_rate_threshold))
        ),
        emit_rate_threshold=float(emit_rate_threshold),
    )


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: float | None
    bound: float
    detail: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "observed": _f(self.observed),
            "bound": float(self.bound),
            "detail": self.detail,
        }


def evaluate_gates(
    metrics: DistillationFidelityMetrics,
    config: DistillationGateConfig,
) -> list[GateResult]:
    """Check ``metrics`` against the configured bounds.

    A gate whose input is undefined (e.g. no holdout rows the teacher scored) is treated
    as **passing**, with the reason recorded — an absent measurement is not evidence of
    failure, and failing the fit over it would be worse than saying so.
    """
    results: list[GateResult] = []

    agreement = _f(metrics.entity_agreement)
    if agreement is None:
        results.append(
            GateResult(
                name="entity_agreement",
                passed=True,
                observed=None,
                bound=config.min_entity_agreement,
                detail="no holdout rows the teacher assigned to a cluster; not measured",
            )
        )
    else:
        ok = agreement >= config.min_entity_agreement
        results.append(
            GateResult(
                name="entity_agreement",
                passed=ok,
                observed=agreement,
                bound=config.min_entity_agreement,
                detail=(
                    f"student reproduces {agreement:.4f} of teacher entity decisions "
                    f"on {metrics.n_holdout_exact + metrics.n_holdout_approx} scored "
                    f"holdout rows (bound {config.min_entity_agreement:.2f})"
                ),
            )
        )

    rel = metrics.emit_rate_relative_delta()
    ok_emit = rel <= config.max_emit_rate_rel_delta
    results.append(
        GateResult(
            name="emit_rate",
            passed=ok_emit,
            observed=rel,
            bound=config.max_emit_rate_rel_delta,
            detail=(
                f"student emits {metrics.emit_rate_student:.4f} vs teacher "
                f"{metrics.emit_rate_teacher:.4f} at thr={metrics.emit_rate_threshold:.2f} "
                f"({rel:.1%} relative)"
            ),
        )
    )
    return results


def apply_gates(
    metrics: DistillationFidelityMetrics,
    config: DistillationGateConfig,
) -> list[GateResult]:
    """Evaluate gates and warn (or raise) on failures per ``config.on_failure``.

    The default is to warn: a fit consumes an expensive corpus embedding, and discarding
    it mid-run over a quality bound is a worse outcome than shipping a model whose
    report says plainly that it failed.
    """
    results = evaluate_gates(metrics, config)
    failed = [r for r in results if not r.passed]
    if not failed:
        logger.info(
            "Distillation gates passed (%d/%d): %s",
            len(results),
            len(results),
            "; ".join(r.detail for r in results),
        )
        return results

    summary = "; ".join(f"{r.name}: {r.detail}" for r in failed)
    if config.on_failure == "raise":
        raise ValueError(f"Distillation fidelity gates failed — {summary}")
    logger.warning(
        "Distillation fidelity gate(s) FAILED — %s. The model was still fitted; see "
        "distillation_fidelity in the fit report.",
        summary,
    )
    return results
