"""Held-out distillation fidelity: the split, the metrics, and the gates."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pelinker.config import DistillationGateConfig
from pelinker.distillation import (
    DistillationFidelityMetrics,
    apply_gates,
    cluster_to_entity_map,
    evaluate_distillation_fidelity,
    evaluate_gates,
    grouped_holdout_split,
)
from pelinker.entity_head import fit_mlp_entity_head

NEG = "__NEGATIVE__"


# --------------------------------------------------------------------------- split


def _frame(n_groups: int, per_group: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pmid": [f"pm{g}" for g in range(n_groups) for _ in range(per_group)],
            "entity": [f"e{g % 4}" for g in range(n_groups) for _ in range(per_group)],
        }
    )


def test_no_pmid_group_straddles_the_split() -> None:
    frame = _frame(n_groups=40, per_group=5)

    split = grouped_holdout_split(frame, holdout_fraction=0.2, random_state=0)

    assert split.grouping == "pmid"
    groups = frame["pmid"].to_numpy()
    train_groups = set(groups[split.train_idx])
    hold_groups = set(groups[split.holdout_idx])
    assert train_groups.isdisjoint(hold_groups)
    assert len(hold_groups) == split.n_groups


def test_split_indices_partition_the_frame_without_overlap() -> None:
    frame = _frame(n_groups=30, per_group=4)

    split = grouped_holdout_split(frame, holdout_fraction=0.25, random_state=1)

    combined = np.sort(np.concatenate([split.train_idx, split.holdout_idx]))
    assert np.array_equal(combined, np.arange(len(frame)))
    assert len(split.train_idx) > 0
    assert len(split.holdout_idx) > 0


def test_holdout_fraction_is_roughly_honoured() -> None:
    frame = _frame(n_groups=100, per_group=3)

    split = grouped_holdout_split(frame, holdout_fraction=0.2, random_state=2)

    frac = len(split.holdout_idx) / len(frame)
    # Groups are atomic, so the realized fraction lands near, not exactly on, target.
    assert 0.15 <= frac <= 0.30


def test_split_is_deterministic_for_a_seed() -> None:
    frame = _frame(n_groups=25, per_group=4)

    a = grouped_holdout_split(frame, holdout_fraction=0.2, random_state=7)
    b = grouped_holdout_split(frame, holdout_fraction=0.2, random_state=7)
    c = grouped_holdout_split(frame, holdout_fraction=0.2, random_state=8)

    assert np.array_equal(a.holdout_idx, b.holdout_idx)
    assert not np.array_equal(a.holdout_idx, c.holdout_idx)


def test_single_group_falls_back_to_row_split_with_a_warning(caplog) -> None:
    """One pmid cannot be split by group; the fallback must be reported, not silent."""
    frame = _frame(n_groups=1, per_group=20)

    with caplog.at_level(logging.WARNING, logger="pelinker.distillation"):
        split = grouped_holdout_split(frame, holdout_fraction=0.2, random_state=0)

    assert split.grouping == "row"
    assert len(split.holdout_idx) > 0
    assert len(split.train_idx) > 0
    assert "falling back to a row-level split" in caplog.text


def test_missing_group_column_falls_back_to_row_split_with_a_warning(caplog) -> None:
    frame = pd.DataFrame({"entity": [f"e{i % 3}" for i in range(20)]})

    with caplog.at_level(logging.WARNING, logger="pelinker.distillation"):
        split = grouped_holdout_split(frame, holdout_fraction=0.2, random_state=0)

    assert split.grouping == "row"
    assert "no 'pmid' column" in caplog.text


def test_split_rejects_degenerate_arguments() -> None:
    frame = _frame(n_groups=10, per_group=2)

    with pytest.raises(ValueError, match="holdout_fraction must be in"):
        grouped_holdout_split(frame, holdout_fraction=0.0, random_state=0)
    with pytest.raises(ValueError, match="holdout_fraction must be in"):
        grouped_holdout_split(frame, holdout_fraction=1.0, random_state=0)
    with pytest.raises(ValueError, match="at least 2 rows"):
        grouped_holdout_split(frame.head(1), holdout_fraction=0.2, random_state=0)


# ------------------------------------------------------------- cluster→entity map


def test_cluster_entity_map_takes_the_majority_entity_and_skips_noise() -> None:
    entities = np.array(["a", "a", "b", "c", "c", "c", "z"])
    clusters = np.array([0, 0, 0, 1, 1, 1, -1])

    mapping = cluster_to_entity_map(entities, clusters)

    assert mapping == {0: "a", 1: "c"}
    assert -1 not in mapping


def test_cluster_entity_map_breaks_ties_deterministically() -> None:
    entities = np.array(["b", "a"])
    clusters = np.array([0, 0])

    assert cluster_to_entity_map(entities, clusters) == {0: "a"}


def test_cluster_entity_map_on_all_noise_is_empty() -> None:
    assert cluster_to_entity_map(np.array(["a", "b"]), np.array([-1, -1])) == {}


# ------------------------------------------------------------------------ metrics


def _blobs(n_clusters: int = 5, per: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(n_clusters, 3))
    labels = np.repeat(np.arange(n_clusters), per)
    X = centers[labels] + rng.normal(scale=0.2, size=(len(labels), 3))
    return X, labels


def _metrics_for(
    X, labels, *, teacher_scores=None, exact_mask=None, seed: int = 0
) -> DistillationFidelityMetrics:
    head = fit_mlp_entity_head(
        X, labels, hidden_layer_sizes=(32, 16), random_state=seed
    )
    entities = np.array([f"e{int(c)}" if c != -1 else "zzz" for c in labels])
    return evaluate_distillation_fidelity(
        head=head,
        umap_holdout=X,
        teacher_labels=labels,
        teacher_scores=(
            np.ones(len(labels)) if teacher_scores is None else teacher_scores
        ),
        exact_label_mask=(
            np.ones(len(labels), dtype=bool) if exact_mask is None else exact_mask
        ),
        cluster_entity_map=cluster_to_entity_map(entities, labels),
        grouping="pmid",
        n_groups=4,
        emit_rate_threshold=0.3,
        noise_label=NEG,
    )


def test_a_faithful_student_scores_near_one() -> None:
    X, labels = _blobs()

    m = _metrics_for(X, labels)

    assert m.entity_agreement > 0.95
    assert m.ari_vs_teacher > 0.95
    assert m.student_kind == "mlp"
    assert m.holdout_grouping == "pmid"
    assert m.n_holdout == len(labels)


def test_exact_and_approximate_agreement_are_reported_separately() -> None:
    """Pooling the two hides which kind of teacher label the student is failing."""
    X, labels = _blobs()
    exact = np.zeros(len(labels), dtype=bool)
    exact[: len(labels) // 2] = True

    m = _metrics_for(X, labels, exact_mask=exact)

    assert m.entity_agreement_exact is not None
    assert m.entity_agreement_approx is not None
    assert m.n_holdout_exact + m.n_holdout_approx == len(labels)
    assert m.n_holdout_exact == pytest.approx(len(labels) // 2, abs=1)


def test_agreement_is_none_for_a_side_with_no_rows() -> None:
    X, labels = _blobs()

    m = _metrics_for(X, labels, exact_mask=np.ones(len(labels), dtype=bool))

    assert m.entity_agreement_exact is not None
    assert m.entity_agreement_approx is None
    assert m.n_holdout_approx == 0


def test_teacher_noise_rows_are_forced_and_their_scores_reported() -> None:
    """The head cannot emit -1, so every teacher-noise row gets assigned regardless."""
    X, labels = _blobs()
    noisy = labels.copy()
    noisy[:30] = -1  # teacher abstains on these

    m = _metrics_for(X, noisy)

    assert m.n_teacher_noise == 30
    assert m.noise_forced_fraction == 1.0
    assert m.forced_noise_score_median is not None
    assert 0.0 <= m.forced_noise_score_median <= 1.0
    assert m.forced_noise_score_p90 is not None


def test_noise_fields_are_neutral_when_the_teacher_abstains_on_nothing() -> None:
    X, labels = _blobs()

    m = _metrics_for(X, labels)

    assert m.n_teacher_noise == 0
    assert m.noise_forced_fraction == 0.0
    assert m.forced_noise_score_median is None


def test_teacher_noise_rows_do_not_drag_down_entity_agreement() -> None:
    """Agreement is scored where the teacher decided; forced noise has its own metric."""
    X, labels = _blobs()
    noisy = labels.copy()
    noisy[:30] = -1

    m = _metrics_for(X, noisy)

    assert m.entity_agreement > 0.9


def test_emit_rate_relative_delta_and_its_zero_teacher_guard() -> None:
    m = _metrics_for(*_blobs())
    assert m.emit_rate_relative_delta() >= 0.0

    from dataclasses import replace

    zero = replace(m, emit_rate_teacher=0.0, emit_rate_student=0.5)
    assert zero.emit_rate_relative_delta() == 0.0


def test_metrics_serialize_to_json_safe_values() -> None:
    import json

    payload = _metrics_for(*_blobs()).to_jsonable()

    json.dumps(payload)
    assert payload["holdout_grouping"] == "pmid"
    assert payload["entity_agreement"] is not None
    assert "emit_rate_relative_delta" in payload


# -------------------------------------------------------------------------- gates


def _metrics_with(**over) -> DistillationFidelityMetrics:
    base = dict(
        student_kind="mlp",
        holdout_grouping="pmid",
        n_holdout=100,
        n_groups=10,
        entity_agreement=0.99,
        entity_agreement_exact=0.99,
        entity_agreement_approx=0.99,
        n_holdout_exact=50,
        n_holdout_approx=50,
        ari_vs_teacher=0.98,
        score_corr_vs_teacher=0.9,
        noise_forced_fraction=0.0,
        n_teacher_noise=0,
        forced_noise_score_median=None,
        forced_noise_score_p90=None,
        emit_rate_student=0.80,
        emit_rate_teacher=0.80,
        emit_rate_threshold=0.3,
    )
    base.update(over)
    return DistillationFidelityMetrics(**base)


def test_gates_pass_on_a_faithful_student() -> None:
    results = evaluate_gates(_metrics_with(), DistillationGateConfig())

    assert all(r.passed for r in results)
    assert {r.name for r in results} == {"entity_agreement", "emit_rate"}


def test_agreement_gate_fires_below_the_bound() -> None:
    results = evaluate_gates(
        _metrics_with(entity_agreement=0.80), DistillationGateConfig()
    )

    agreement = next(r for r in results if r.name == "entity_agreement")
    assert not agreement.passed
    assert agreement.observed == pytest.approx(0.80)
    assert agreement.bound == pytest.approx(0.95)


def test_emit_rate_gate_fires_on_a_large_relative_shift() -> None:
    results = evaluate_gates(
        _metrics_with(emit_rate_student=0.50, emit_rate_teacher=0.80),
        DistillationGateConfig(),
    )

    emit = next(r for r in results if r.name == "emit_rate")
    assert not emit.passed
    assert emit.observed == pytest.approx(0.375)


def test_an_unmeasurable_agreement_gate_passes_and_says_why() -> None:
    """An absent measurement is not evidence of failure."""
    results = evaluate_gates(
        _metrics_with(entity_agreement=float("nan")), DistillationGateConfig()
    )

    agreement = next(r for r in results if r.name == "entity_agreement")
    assert agreement.passed
    assert agreement.observed is None
    assert "not measured" in agreement.detail


def test_apply_gates_warns_by_default_and_keeps_going(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="pelinker.distillation"):
        results = apply_gates(
            _metrics_with(entity_agreement=0.50), DistillationGateConfig()
        )

    assert any(not r.passed for r in results)
    assert "gate(s) FAILED" in caplog.text


def test_apply_gates_can_be_configured_to_raise() -> None:
    with pytest.raises(ValueError, match="gates failed"):
        apply_gates(
            _metrics_with(entity_agreement=0.50),
            DistillationGateConfig(on_failure="raise"),
        )


def test_gate_config_validates_its_bounds() -> None:
    with pytest.raises(ValueError, match="min_entity_agreement"):
        DistillationGateConfig(min_entity_agreement=1.5)
    with pytest.raises(ValueError, match="max_emit_rate_rel_delta"):
        DistillationGateConfig(max_emit_rate_rel_delta=-0.1)
    with pytest.raises(ValueError, match="on_failure"):
        DistillationGateConfig(on_failure="explode")
