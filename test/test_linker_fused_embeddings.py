"""Mention-level fused parquets for ``Linker.fit`` (inner join + concat, analysis-aligned)."""

import pandas as pd

from numpy.random import RandomState

from pelinker.config import (
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
)
from pelinker.model import Linker
from pelinker.transform import TransformConfig


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
        # Two mentions per property so HDBSCAN min_cluster_size=2 can assign non-noise labels.
        for pmid in ("1", "2"):
            rows1.append(
                {
                    "pmid": pmid,
                    "property": f"p{k}",
                    "mention": "m",
                    "embed": [float(k), 1.0],
                }
            )
            rows2.append(
                {
                    "pmid": pmid,
                    "property": f"p{k}",
                    "mention": "m",
                    "embed": [0.5, float(k) * 0.1],
                }
            )
    pd.DataFrame(rows1).to_parquet(p1)
    pd.DataFrame(rows2).to_parquet(p2)

    opt = ClusteringOptimizationConfig(
        min_class_size=1,
        max_scale=30,
        batch_size=500,
    )
    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        [p1, p2],
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        min_cluster_size=2,
        clustering_optimization_config=opt,
    )
    assert linker.transformer is not None
    assert len(linker.vocabulary) == n_ent
    assert linker.transformer.pca is not None
    assert linker.transformer.pca.n_features_in_ == 4
    assert linker.clusterer is not None
    assert getattr(linker.clusterer, "prediction_data_", None) is not None
    assert linker.training_cluster_frame is not None
    assert len(linker.training_cluster_frame) == 2 * n_ent
    assert set(linker.training_cluster_frame.columns) >= {
        "pmid",
        "property",
        "mention",
        "cluster",
    }
    assert linker.cluster_composition is not None
    assert linker.cluster_composition.global_property_mass
    assert linker.cluster_consensus_names


def test_fused_fit_optimize_clustering_grid_then_full_fit(tmp_path):
    """Grid search uses ``frac`` subsample; final model fits on all prepared rows."""
    metadata = EmbeddingModelMetadata(
        sources=(
            EmbeddingSourceSpec(model_type="a", layers_spec="1"),
            EmbeddingSourceSpec(model_type="a", layers_spec="2"),
        )
    )
    n_ent = 20
    labels_map = {f"e{k}": f"p{k}" for k in range(n_ent)}
    p1 = tmp_path / "s0.parquet"
    p2 = tmp_path / "s1.parquet"
    rows1 = []
    rows2 = []
    for k in range(n_ent):
        for pmid in ("a", "b", "c", "d"):
            rows1.append(
                {
                    "pmid": pmid,
                    "property": f"p{k}",
                    "mention": "m",
                    "embed": [float(k), 0.1],
                }
            )
            rows2.append(
                {
                    "pmid": pmid,
                    "property": f"p{k}",
                    "mention": "m",
                    "embed": [0.2, float(k) * 0.05],
                }
            )
    pd.DataFrame(rows1).to_parquet(p1)
    pd.DataFrame(rows2).to_parquet(p2)

    opt = ClusteringOptimizationConfig(
        min_class_size=4,
        max_scale=25,
        frac=0.7,
        rns=RandomState(7),
        batch_size=500,
    )
    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        [p1, p2],
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        optimize_clustering=True,
        clustering_optimization_config=opt,
    )
    assert linker.clusterer is not None
    assert linker.training_cluster_frame is not None
    assert len(linker.training_cluster_frame) == 4 * n_ent


def test_fit_writes_clustering_report_artifacts_when_report_dir_set(tmp_path):
    """``clustering_report_dir`` mirrors clustering_quality-style CSV, pickle, and metrics plot."""
    metadata = EmbeddingModelMetadata(
        sources=(
            EmbeddingSourceSpec(model_type="a", layers_spec="1"),
            EmbeddingSourceSpec(model_type="a", layers_spec="2"),
        )
    )
    n_ent = 12
    labels_map = {f"e{k}": f"p{k}" for k in range(n_ent)}
    p1 = tmp_path / "s0.parquet"
    p2 = tmp_path / "s1.parquet"
    rows1 = []
    rows2 = []
    for k in range(n_ent):
        for pmid in ("a", "b", "c", "d"):
            rows1.append(
                {
                    "pmid": pmid,
                    "property": f"p{k}",
                    "mention": "m",
                    "embed": [float(k), 0.1],
                }
            )
            rows2.append(
                {
                    "pmid": pmid,
                    "property": f"p{k}",
                    "mention": "m",
                    "embed": [0.2, float(k) * 0.05],
                }
            )
    pd.DataFrame(rows1).to_parquet(p1)
    pd.DataFrame(rows2).to_parquet(p2)

    opt = ClusteringOptimizationConfig(
        min_class_size=4,
        max_scale=25,
        frac=1.0,
        rns=RandomState(0),
        batch_size=500,
    )
    report_dir = tmp_path / "clustering_report"
    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        [p1, p2],
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        optimize_clustering=True,
        clustering_optimization_config=opt,
        clustering_report_dir=report_dir,
    )

    assert (report_dir / "results.csv").is_file()
    assert (report_dir / "results_grid_per_sample.csv").is_file()
    assert (report_dir / "fine_clustering_metadata.pkl.gz").is_file()
    assert (report_dir / "fusion2_a_1__a_2.png").is_file()


def test_estimate_model_clustering_multi_file_paths(tmp_path):
    from pelinker.analysis import estimate_model_clustering

    p1 = tmp_path / "s0.parquet"
    p2 = tmp_path / "s1.parquet"
    n = 25
    pd.DataFrame(
        {
            "pmid": [str(i) for i in range(n)],
            "property": ["foo" if i < (n // 2) else "bar" for i in range(n)],
            "mention": ["m"] * n,
            "embed": [[float(i + 1), float((i % 5) + 1)] for i in range(n)],
        }
    ).to_parquet(p1)
    pd.DataFrame(
        {
            "pmid": [str(i) for i in range(n)],
            "property": ["foo" if i < (n // 2) else "bar" for i in range(n)],
            "mention": ["m"] * n,
            "embed": [[float((i % 7) + 1), float(i + 2)] for i in range(n)],
        }
    ).to_parquet(p2)

    report = estimate_model_clustering(
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        file_paths=[p1, p2],
        optimization_config=ClusteringOptimizationConfig(
            min_class_size=10,
            max_scale=40,
            rns=RandomState(0),
            frac=1.0,
        ),
    )
    if report is not None:
        assert report.df is not None
        assert "u_00" in report.df.columns


def test_fit_stores_and_serializes_training_pca_metrics(tmp_path):
    metadata = EmbeddingModelMetadata(
        sources=(
            EmbeddingSourceSpec(model_type="a", layers_spec="1"),
            EmbeddingSourceSpec(model_type="a", layers_spec="2"),
        )
    )
    n_ent = 12
    labels_map = {f"e{k}": f"p{k}" for k in range(n_ent)}
    p1 = tmp_path / "s0.parquet"
    p2 = tmp_path / "s1.parquet"
    rows1 = []
    rows2 = []
    for k in range(n_ent):
        for pmid in ("a", "b", "c"):
            rows1.append(
                {
                    "pmid": pmid,
                    "property": f"p{k}",
                    "mention": "m",
                    "embed": [float(k), 0.1],
                }
            )
            rows2.append(
                {
                    "pmid": pmid,
                    "property": f"p{k}",
                    "mention": "m",
                    "embed": [0.2, float(k) * 0.05],
                }
            )
    pd.DataFrame(rows1).to_parquet(p1)
    pd.DataFrame(rows2).to_parquet(p2)

    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        [p1, p2],
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        min_cluster_size=2,
        clustering_optimization_config=ClusteringOptimizationConfig(
            min_class_size=1,
            max_scale=20,
            batch_size=500,
        ),
    )

    assert linker.training_cluster_frame is not None
    assert linker.training_pca_residuals is not None
    assert linker.training_pca_mahalanobis is not None
    assert linker.training_pca_residuals.shape == (len(linker.training_cluster_frame),)
    assert linker.training_pca_mahalanobis.shape == (
        len(linker.training_cluster_frame),
    )
    assert (linker.training_pca_residuals >= 0.0).all()
    assert (linker.training_pca_mahalanobis >= 0.0).all()
    summary = linker.training_anomaly_metric_summary()
    assert summary is not None
    assert set(summary.keys()) == {"residual", "mahalanobis", "combined_max_z"}

    model_path = tmp_path / "linker_residual_test"
    linker.dump(model_path)
    loaded = Linker.load(model_path)
    assert loaded.training_pca_residuals is not None
    assert loaded.training_pca_mahalanobis is not None
    assert loaded.training_pca_residuals.shape == linker.training_pca_residuals.shape
    assert (
        loaded.training_pca_mahalanobis.shape == linker.training_pca_mahalanobis.shape
    )
