"""Distillation fidelity through a real compact Linker.fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pelinker.core.config import (
    DistillationGateConfig,
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    LinkerFitConfig,
    ManifoldOovScreenerConfig,
    TransformConfig,
)
from pelinker.model import Linker
from pelinker.reports.io import clustering_report_to_jsonable_dict


def _mentions_parquet(tmp_path: Path, n_ent: int = 24) -> tuple[Path, dict[str, str]]:
    """Well-separated blobs spread over many pmids so a grouped split is possible."""
    rng = np.random.default_rng(0)
    p = tmp_path / "mentions.parquet"
    rows = []
    for k in range(n_ent):
        for j in range(8):
            v = np.array([float(k), float(k) * 0.1, 1.0, 0.5])
            v = v + rng.normal(scale=0.01, size=4)
            rows.append(
                {
                    "pmid": f"pm{k * 8 + j}",
                    "entity": f"p{k}",
                    "mention": "m",
                    "embed": v.tolist(),
                }
            )
    pd.DataFrame(rows).to_parquet(p)
    return p, {f"e{k}": f"p{k}" for k in range(n_ent)}


def _fit(tmp_path: Path, **fit_kwargs) -> Linker:
    parquet, labels_map = _mentions_parquet(tmp_path)
    linker = Linker(
        labels_map=labels_map,
        embedding_metadata=EmbeddingModelMetadata(
            sources=(EmbeddingSourceSpec(model_type="a", layers_spec="1"),)
        ),
    )
    defaults = dict(
        batch_size=500,
        predict_mode="compact",
        entity_head_hidden_layers=(32, 16),
        projection_screener=ManifoldOovScreenerConfig(enabled=False),
    )
    defaults.update(fit_kwargs)
    linker.fit(
        parquet,
        transform_config=TransformConfig(
            pca_components=4,
            umap_components=2,
            cluster_viz_components=2,
            # manifold_kind left at its default: predict_mode decides it.
            parametric_umap_n_training_epochs=1,
            umap_seed=13,
        ),
        min_cluster_size=2,
        fit_config=LinkerFitConfig(**defaults),
    )
    return linker


def test_compact_fit_measures_and_records_distillation_fidelity(tmp_path: Path) -> None:
    linker = _fit(tmp_path)

    m = linker.distillation_fidelity
    assert m is not None
    assert m.student_kind == "mlp"
    assert m.n_holdout > 0
    # Many distinct pmids are available, so the split must be group-based.
    assert m.holdout_grouping == "pmid"
    assert m.n_groups > 0
    assert 0.0 <= m.entity_agreement <= 1.0
    # The shipped artifact is still the compact one.
    assert linker.clusterer is None
    assert linker.entity_head is not None


def test_holdout_fraction_zero_skips_the_measurement(tmp_path: Path) -> None:
    """The escape hatch for reproducing pre-fidelity artifacts."""
    linker = _fit(tmp_path, entity_head_holdout_fraction=0.0)

    assert linker.distillation_fidelity is None
    assert linker.entity_head is not None
    assert linker.clusterer is None


def test_legacy_mode_reports_no_distillation_fidelity(tmp_path: Path) -> None:
    linker = _fit(tmp_path, predict_mode="legacy")

    assert linker.distillation_fidelity is None
    assert linker.entity_head is None
    assert linker.clusterer is not None


def test_fidelity_block_reaches_the_fit_report(tmp_path: Path) -> None:
    linker = _fit(tmp_path)
    report = linker.take_fit_clustering_report()
    assert report is not None

    payload = clustering_report_to_jsonable_dict(report)

    block = payload["distillation_fidelity"]
    assert block is not None
    assert block["student_kind"] == "mlp"
    assert block["holdout_grouping"] == "pmid"
    assert block["n_holdout"] > 0
    assert "entity_agreement" in block
    assert "noise_forced_fraction" in block
    json.dumps(payload)


# A single hidden unit cannot separate 24 clusters — the study's "underfit control"
# idea, used here to make a gate failure deterministic rather than data-dependent.
_UNDERFIT_HEAD = (1,)


def test_gates_can_abort_a_fit_whose_student_underfits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gates failed"):
        _fit(
            tmp_path,
            entity_head_hidden_layers=_UNDERFIT_HEAD,
            distillation_gates=DistillationGateConfig(on_failure="raise"),
        )


def test_a_failing_gate_only_warns_by_default(tmp_path: Path) -> None:
    """Default is warn: a fit consumes an expensive embedding, so it is not discarded."""
    linker = _fit(tmp_path, entity_head_hidden_layers=_UNDERFIT_HEAD)

    assert linker.entity_head is not None
    m = linker.distillation_fidelity
    assert m is not None
    assert m.entity_agreement < 0.95  # the gate did have something to fire on


def test_disabled_gates_never_abort(tmp_path: Path) -> None:
    linker = _fit(
        tmp_path,
        entity_head_hidden_layers=_UNDERFIT_HEAD,
        distillation_gates=DistillationGateConfig(enabled=False, on_failure="raise"),
    )

    # Metrics are still measured and reported; only the enforcement is off.
    assert linker.distillation_fidelity is not None


def test_the_shipped_head_is_trained_on_every_row(tmp_path: Path) -> None:
    """The holdout measures the recipe; it must not be withheld from the artifact.

    A head trained only on the training split would know fewer clusters than the fit
    produced, so the shipped classes_ must cover every non-noise teacher cluster.
    """
    linker = _fit(tmp_path)

    assert linker.entity_head is not None
    shipped_classes = set(int(c) for c in linker.entity_head.classes_)
    fit_clusters = {int(c) for c in linker.cluster_id_to_entity_id if int(c) != -1}
    assert fit_clusters.issubset(shipped_classes)


def test_explicit_manifold_kind_conflicting_with_predict_mode_is_rejected(
    tmp_path: Path,
) -> None:
    """Silently overriding this used to move MCS across a coordinate-system change."""
    parquet, labels_map = _mentions_parquet(tmp_path)
    linker = Linker(
        labels_map=labels_map,
        embedding_metadata=EmbeddingModelMetadata(
            sources=(EmbeddingSourceSpec(model_type="a", layers_spec="1"),)
        ),
    )
    with pytest.raises(ValueError, match="requires manifold_kind"):
        linker.fit(
            parquet,
            transform_config=TransformConfig(
                pca_components=4,
                umap_components=2,
                cluster_viz_components=2,
                manifold_kind="parametric",  # conflicts with predict_mode="legacy"
                umap_seed=13,
            ),
            min_cluster_size=2,
            fit_config=LinkerFitConfig(
                batch_size=500,
                predict_mode="legacy",
                projection_screener=ManifoldOovScreenerConfig(enabled=False),
            ),
        )


def test_default_manifold_kind_is_filled_in_from_predict_mode(tmp_path: Path) -> None:
    """Leaving manifold_kind at its default still gets aligned, as before."""
    linker = _fit(tmp_path, predict_mode="legacy")

    assert linker.transform_config is not None
    assert linker.transform_config.manifold_kind == "umap"
