"""Cluster viz scope: HDBSCAN-fit + screener/OOV-pass rows only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pelinker.analysis import drop_entities_with_few_mentions, split_by_negative_label
from pelinker.config import (
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    LinkerFitConfig,
    ManifoldOovScreenerConfig,
    NegativeScreenerConfig,
    TransformConfig,
)
from pelinker.model import Linker
from pelinker.onto import NEGATIVE_LABEL
from pelinker.plotting import (
    CLUSTER_VIZ_MEMBERSHIP_COLUMNS,
    build_fit_cluster_viz_plot_df,
    filter_assignments_for_cluster_viz,
)
from pelinker.reporting import (
    ClusteringHyperparameters,
    ModelSelectionReport,
    read_clustering_report_json,
    write_clustering_report_json,
)

_SUBSAMPLE_ROWS = 80


def _mention_frame(*, dim: int = 12, n_pos: int = 120, n_neg: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    pos = rng.standard_normal((n_pos, dim)).astype(np.float32)
    neg = rng.standard_normal((n_neg, dim)).astype(np.float32) + 1.2
    rows: list[dict[str, object]] = []
    entities = [f"prop_{i % 8}" for i in range(n_pos)]
    for i in range(n_pos):
        rows.append(
            {
                "pmid": str(i // 3),
                "entity": entities[i],
                "mention": f"m{i}",
                "embed": pos[i],
            }
        )
    for j in range(n_neg):
        rows.append(
            {
                "pmid": str(900 + j),
                "entity": NEGATIVE_LABEL,
                "mention": f"n{j}",
                "embed": neg[j],
            }
        )
    return pd.DataFrame(rows)


def _filtered_frame() -> pd.DataFrame:
    raw = _mention_frame()
    return drop_entities_with_few_mentions(
        raw, min_mentions_per_entity=5, negative_label=NEGATIVE_LABEL
    )


def test_filter_assignments_for_cluster_viz_membership() -> None:
    assign = pd.DataFrame(
        {
            "entity": ["a", "a", "b", "b"],
            "cluster": [0, 1, 0, -1],
            "clustering_in_sample": [True, False, True, True],
            "screener_pass": [True, True, False, True],
            "manifold_oov_pass": [True, True, True, False],
        }
    )
    scoped = filter_assignments_for_cluster_viz(assign, hdbscan_fit_scope=True)
    assert len(scoped) == 1
    assert int(scoped.iloc[0]["cluster"]) == 0

    all_kb = filter_assignments_for_cluster_viz(assign, hdbscan_fit_scope=False)
    assert len(all_kb) == 3


def test_filter_assignments_includes_noise_when_requested() -> None:
    assign = pd.DataFrame(
        {
            "entity": ["a", "b"],
            "cluster": [0, -1],
            "clustering_in_sample": [True, True],
            "screener_pass": [True, True],
            "manifold_oov_pass": [True, True],
        }
    )
    with_noise = filter_assignments_for_cluster_viz(
        assign, exclude_noise=False, hdbscan_fit_scope=True
    )
    assert set(with_noise["cluster"].astype(int)) == {0, -1}
    without = filter_assignments_for_cluster_viz(
        assign, exclude_noise=True, hdbscan_fit_scope=True
    )
    assert set(without["cluster"].astype(int)) == {0}


def test_filter_assignments_legacy_report_without_membership_cols() -> None:
    assign = pd.DataFrame({"entity": ["a", "b"], "cluster": [0, 1]})
    out = filter_assignments_for_cluster_viz(assign, hdbscan_fit_scope=True)
    assert len(out) == 2


def test_build_fit_cluster_viz_plot_df_scoped_vs_all_kb(
    tmp_path: Path,
) -> None:
    frame = _filtered_frame()
    parquet = tmp_path / "emb.parquet"
    frame.to_parquet(parquet)
    labels_map = {f"prop_{i}": f"prop_{i}" for i in range(8)}
    tc = TransformConfig(
        pca_components=8,
        umap_components=3,
        cluster_viz_components=3,
        pca_seed=13,
        umap_seed=13,
    )
    linker = Linker(
        labels_map=labels_map,
        transform_config=tc,
        embedding_metadata=EmbeddingModelMetadata(
            sources=(EmbeddingSourceSpec(model_type="t", layers_spec="1"),)
        ),
    )
    linker.fit(
        embeddings=parquet,
        transform_config=tc,
        min_cluster_size=5,
        fit_config=LinkerFitConfig(
            clustering_sample_rows=_SUBSAMPLE_ROWS,
            base_seed=13,
            clustering_sample_index=0,
            screener_seed=13,
            ambient_screener=NegativeScreenerConfig(kind="lda"),
            projection_screener=ManifoldOovScreenerConfig(enabled=False),
        ),
    )
    fit_report = linker.take_fit_clustering_report()
    assert fit_report is not None
    for col in CLUSTER_VIZ_MEMBERSHIP_COLUMNS:
        assert col in fit_report.assignments.columns

    _, manifold_full = split_by_negative_label(frame, NEGATIVE_LABEL)
    assert len(fit_report.assignments) == len(manifold_full)

    n_in_sample = int(fit_report.assignments["clustering_in_sample"].sum())
    assert 0 < n_in_sample < len(manifold_full)

    scoped_df, _ = build_fit_cluster_viz_plot_df(
        fit_report, exclude_noise=True, hdbscan_fit_scope=True
    )
    all_kb_df, _ = build_fit_cluster_viz_plot_df(
        fit_report, exclude_noise=True, hdbscan_fit_scope=False
    )
    assert scoped_df is not None
    assert all_kb_df is not None
    assert len(scoped_df) <= len(all_kb_df)
    assert len(scoped_df) <= n_in_sample

    eligible = fit_report.assignments.loc[
        fit_report.assignments["clustering_in_sample"]
        & fit_report.assignments["screener_pass"]
        & fit_report.assignments["manifold_oov_pass"]
        & (fit_report.assignments["cluster"].astype(int) != -1)
    ]
    assert len(scoped_df) == len(eligible)


def test_membership_columns_roundtrip_json(tmp_path: Path) -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["a", "b"],
            "cluster": [0, 1],
            "clustering_in_sample": [True, False],
            "screener_pass": [True, False],
            "manifold_oov_pass": [True, True],
        }
    )
    report = ModelSelectionReport(
        hyperparameters=ClusteringHyperparameters(min_cluster_size=5),
        best_score=0.5,
        number_properties=2,
        n_clusters_emergent=2,
        metrics_df=pd.DataFrame(),
        assignments=assignments,
        pca_residuals=np.zeros(2),
        pca_mahalanobis=np.zeros(2),
        pca_spectral_entropy=np.zeros(2),
        oov_label=np.zeros(2, dtype=np.int64),
        umap_clustering=np.zeros((2, 2)),
        cluster_viz=np.zeros((2, 2)),
        cluster_viz_method="pca",
        pca_reduced=np.zeros((2, 2)),
    )
    path = tmp_path / "report.json.gz"
    write_clustering_report_json(path, report)
    loaded = read_clustering_report_json(path)
    assert bool(loaded.assignments.iloc[0]["clustering_in_sample"]) is True
    assert bool(loaded.assignments.iloc[1]["screener_pass"]) is False
