"""Mention-level fused parquets for ``Linker.fit`` (inner join + concat, analysis-aligned)."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from numpy.random import RandomState

from pelinker.config import (
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    LinkerFitConfig,
    NegativeScreenerConfig,
)
from pelinker.model import Linker
from pelinker.onto import NEGATIVE_LABEL
from pelinker.reporting import (
    LINKER_FIT_CLUSTERING_REPORT_BASENAME,
    read_clustering_report_json,
    write_clustering_report_json,
)
from pelinker.transform import TransformConfig


def _two_source_parquets(tmp_path, n_ent: int, pmids: tuple[str, ...]):
    p1 = tmp_path / "s0.parquet"
    p2 = tmp_path / "s1.parquet"
    rows1 = []
    rows2 = []
    for k in range(n_ent):
        for pmid in pmids:
            rows1.append(
                {
                    "pmid": pmid,
                    "entity": f"p{k}",
                    "mention": "m",
                    "embed": [float(k), 0.1],
                }
            )
            rows2.append(
                {
                    "pmid": pmid,
                    "entity": f"p{k}",
                    "mention": "m",
                    "embed": [0.2, float(k) * 0.05],
                }
            )
    pd.DataFrame(rows1).to_parquet(p1)
    pd.DataFrame(rows2).to_parquet(p2)
    return p1, p2


def test_fused_fit_two_parquets_stacks_embedding_dim(tmp_path):
    metadata = EmbeddingModelMetadata(
        sources=(
            EmbeddingSourceSpec(model_type="a", layers_spec="1"),
            EmbeddingSourceSpec(model_type="a", layers_spec="2"),
        )
    )
    n_ent = 24
    labels_map = {f"e{k}": f"p{k}" for k in range(n_ent)}

    p1 = tmp_path / "s0.parquet"
    p2 = tmp_path / "s1.parquet"
    rows1 = []
    rows2 = []
    for k in range(n_ent):
        # Two mentions per entity so HDBSCAN min_cluster_size=2 can assign non-noise labels.
        for pmid in ("1", "2"):
            rows1.append(
                {
                    "pmid": pmid,
                    "entity": f"p{k}",
                    "mention": "m",
                    "embed": [float(k), 1.0],
                }
            )
            rows2.append(
                {
                    "pmid": pmid,
                    "entity": f"p{k}",
                    "mention": "m",
                    "embed": [0.5, float(k) * 0.1],
                }
            )
    pd.DataFrame(rows1).to_parquet(p1)
    pd.DataFrame(rows2).to_parquet(p2)

    fit_cfg = LinkerFitConfig(min_class_size=1, batch_size=500)
    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        [p1, p2],
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        min_cluster_size=2,
        fit_config=fit_cfg,
    )
    assert linker.transformer is not None
    assert len(linker.vocabulary) == n_ent
    assert linker.transformer.pca is not None
    assert linker.transformer.pca.n_features_in_ == 4
    assert linker.clusterer is not None
    assert getattr(linker.clusterer, "prediction_data_", None) is not None
    assert linker.cluster_composition is not None
    assert linker.cluster_composition.global_property_mass
    assert linker.cluster_consensus_names
    assert linker.screener_in_sample_metrics is None
    assert linker.clustering_fit_metrics is not None
    assert linker.clustering_fit_metrics.min_cluster_size == 2
    assert 0.0 <= linker.clustering_fit_metrics.noise_fraction <= 1.0
    assert linker.clustering_fit_metrics.n_samples == 2 * n_ent
    report = linker.take_fit_clustering_report()
    assert report is not None
    assert len(report.assignments) == 2 * n_ent
    assert set(report.assignments.columns) >= {
        "pmid",
        "entity",
        "mention",
        "cluster",
    }
    assert len(report.pca_residuals) == 2 * n_ent
    assert report.umap_clustering.shape == (2 * n_ent, 2)


def test_fit_with_synthetic_negatives_screener_metrics_and_dump_load(tmp_path):
    metadata = EmbeddingModelMetadata(
        sources=(
            EmbeddingSourceSpec(model_type="a", layers_spec="1"),
            EmbeddingSourceSpec(model_type="a", layers_spec="2"),
        )
    )
    n_ent = 10
    labels_map = {f"e{k}": f"p{k}" for k in range(n_ent)}
    p1, p2 = _two_source_parquets(tmp_path, n_ent, ("a", "b"))

    rows_neg1 = [
        {"pmid": "z", "entity": NEGATIVE_LABEL, "mention": "n", "embed": [9.0, 9.0]}
    ]
    rows_neg2 = [
        {"pmid": "z", "entity": NEGATIVE_LABEL, "mention": "n", "embed": [9.1, 8.9]}
    ]
    d1 = pd.read_parquet(p1)
    d2 = pd.read_parquet(p2)
    pd.concat([d1, pd.DataFrame(rows_neg1)], ignore_index=True).to_parquet(p1)
    pd.concat([d2, pd.DataFrame(rows_neg2)], ignore_index=True).to_parquet(p2)

    fit_cfg = LinkerFitConfig(
        min_class_size=1,
        batch_size=500,
        ambient_screener=NegativeScreenerConfig(
            kind="lda",
            negative_label=NEGATIVE_LABEL,
        ),
    )
    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        [p1, p2],
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        min_cluster_size=2,
        fit_config=fit_cfg,
    )
    assert linker.projection is not None
    assert linker.screener is not None
    assert linker.screener_in_sample_metrics is not None
    assert linker.screener_in_sample_metrics.n_negative_label_mentions >= 1
    assert linker.screener_in_sample_metrics.n_kb_mentions >= 1
    fit_report = linker.take_fit_clustering_report()
    assert fit_report is not None
    assert fit_report.all_screener_cv is None
    assert fit_report.training_diagnostics is not None
    diag = fit_report.training_diagnostics
    # Inner-joined KB rows (two pmids per entity) plus one fused synthetic-negative row.
    n_prep = 2 * n_ent + 1
    assert diag.n_total == n_prep
    assert len(diag.pca_residual) <= fit_cfg.diagnostics_sample_size
    assert len(diag.pca_residual) == len(diag.oov_label)
    assert set(np.unique(diag.oov_label).tolist()) == {0, 1}
    assert linker.projection is not None
    assert np.isfinite(diag.projection_score).any()
    assert NEGATIVE_LABEL not in set(
        fit_report.assignments["entity"].astype(str).unique()
    )

    model_path = tmp_path / "linker_metrics"
    linker.dump(model_path)
    loaded = Linker.load(model_path)
    assert loaded.projection is not None
    assert loaded.screener_in_sample_metrics == linker.screener_in_sample_metrics
    assert loaded.clustering_fit_metrics == linker.clustering_fit_metrics


def test_evaluate_selection_from_paths_multi_file_paths(tmp_path):
    from pelinker.selection import evaluate_selection_from_paths

    p1 = tmp_path / "s0.parquet"
    p2 = tmp_path / "s1.parquet"
    n = 25
    pd.DataFrame(
        {
            "pmid": [str(i) for i in range(n)],
            "entity": ["foo" if i < (n // 2) else "bar" for i in range(n)],
            "mention": ["m"] * n,
            "embed": [[float(i + 1), float((i % 5) + 1)] for i in range(n)],
        }
    ).to_parquet(p1)
    pd.DataFrame(
        {
            "pmid": [str(i) for i in range(n)],
            "entity": ["foo" if i < (n // 2) else "bar" for i in range(n)],
            "mention": ["m"] * n,
            "embed": [[float((i % 7) + 1), float(i + 2)] for i in range(n)],
        }
    ).to_parquet(p2)

    report = evaluate_selection_from_paths(
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        file_paths=[p1, p2],
        optimization_config=ClusteringOptimizationConfig(
            min_class_size=1,
            min_scale=2,
            max_scale=20,
            clustering_grid_step=2,
            rns=RandomState(0),
            base_seed=0,
            frac=1.0,
            eval_max_rows=None,
        ),
    )
    if report is not None:
        assert report.assignments is not None
        assert "cluster" in report.assignments.columns
        assert report.umap_clustering.shape[1] == 2


def test_fit_stores_and_serializes_training_pca_metrics(tmp_path):
    metadata = EmbeddingModelMetadata(
        sources=(
            EmbeddingSourceSpec(model_type="a", layers_spec="1"),
            EmbeddingSourceSpec(model_type="a", layers_spec="2"),
        )
    )
    n_ent = 12
    labels_map = {f"e{k}": f"p{k}" for k in range(n_ent)}
    p1, p2 = _two_source_parquets(tmp_path, n_ent, ("a", "b", "c"))

    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        [p1, p2],
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        min_cluster_size=2,
        fit_config=LinkerFitConfig(min_class_size=1, batch_size=500),
    )

    report = linker.take_fit_clustering_report()
    assert report is not None
    n = len(report.assignments)
    assert report.pca_residuals.shape == (n,)
    assert report.pca_mahalanobis.shape == (n,)
    assert report.pca_spectral_entropy.shape == (n,)
    assert (report.pca_residuals >= 0.0).all()
    assert (report.pca_mahalanobis >= 0.0).all()
    assert (report.pca_spectral_entropy >= 0.0).all()

    model_path = tmp_path / "linker_residual_test"
    linker.dump(model_path)
    loaded = Linker.load(model_path)
    assert loaded.clustering_fit_metrics == linker.clustering_fit_metrics


def test_fit_clustering_report_json_roundtrip(tmp_path: Path) -> None:
    metadata = EmbeddingModelMetadata(
        sources=(
            EmbeddingSourceSpec(model_type="a", layers_spec="1"),
            EmbeddingSourceSpec(model_type="a", layers_spec="2"),
        )
    )
    n_ent = 8
    labels_map = {f"e{k}": f"p{k}" for k in range(n_ent)}
    p1, p2 = _two_source_parquets(tmp_path, n_ent, ("1", "2"))

    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        [p1, p2],
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        min_cluster_size=2,
        fit_config=LinkerFitConfig(min_class_size=1, batch_size=500),
    )
    report = linker.take_fit_clustering_report()
    assert report is not None
    out = tmp_path / LINKER_FIT_CLUSTERING_REPORT_BASENAME
    write_clustering_report_json(out, report)
    with gzip.open(out, mode="rt", encoding="utf-8") as fh:
        blob = json.load(fh)
    assert blob["schema"] == "pelinker.clustering_report.v8"
    n = len(report.assignments)
    assert len(blob["pca_residuals"]) == n
    assert len(blob["pca_mahalanobis"]) == n
    assert len(blob["pca_spectral_entropy"]) == n
    assert len(blob["oov_label"]) == n
    assert blob["oov_label"] == [0] * n
    assert blob["training_diagnostics"] is not None
    td = blob["training_diagnostics"]
    assert td["n_total"] == n
    assert len(td["pca_residual"]) == n

    roundtrip = read_clustering_report_json(out)
    assert roundtrip.training_diagnostics is not None
    d = roundtrip.training_diagnostics
    assert d.n_total == n
    assert len(d.pca_residual) == n
    np.testing.assert_array_almost_equal(d.pca_residual, report.pca_residuals)
    np.testing.assert_array_equal(d.oov_label, np.zeros(n, dtype=np.int64))
