"""Parity between model-selection clustering subsample and Linker.fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pelinker.analysis import drop_entities_with_few_mentions, split_by_negative_label
from pelinker.clustering_fit import fit_manifold_clustering
from pelinker.config import (
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    LinkerFitConfig,
    ManifoldOovScreenerConfig,
    TransformConfig,
)
from pelinker.model import Linker
from pelinker.onto import NEGATIVE_LABEL
from pelinker.sampling import draw_selection_sample


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


def test_clustering_subsample_draw_matches_linker_fit_config() -> None:
    frame = _filtered_frame()
    opt = ClusteringOptimizationConfig(
        min_class_size=5,
        frac=0.5,
        eval_max_rows=None,
        base_seed=13,
    )
    fit_cfg = LinkerFitConfig(
        min_class_size=5,
        frac=0.5,
        eval_max_rows=None,
        base_seed=13,
        clustering_sample_index=0,
    )
    sample_sel = draw_selection_sample(frame, opt, sample_index=0)
    sample_fit = draw_selection_sample(
        frame, fit_cfg.to_clustering_sample_config(), sample_index=0
    )
    pd.testing.assert_frame_equal(
        sample_sel.reset_index(drop=True),
        sample_fit.reset_index(drop=True),
    )


def test_fit_manifold_clustering_parity_on_shared_subsample() -> None:
    frame = _filtered_frame()
    opt = ClusteringOptimizationConfig(
        min_class_size=5,
        frac=0.5,
        eval_max_rows=None,
        base_seed=13,
    )
    sample = draw_selection_sample(frame, opt, sample_index=0)
    _, manifold = split_by_negative_label(sample, NEGATIVE_LABEL)
    tc = TransformConfig(
        pca_components=8,
        umap_components=3,
        cluster_viz_components=3,
        seed=13,
    )
    mcs = 5
    sel_result = fit_manifold_clustering(
        manifold,
        transform_config=tc,
        min_cluster_size=mcs,
        prediction_data=False,
    )
    fit_result = fit_manifold_clustering(
        manifold,
        transform_config=tc,
        min_cluster_size=mcs,
        prediction_data=True,
    )
    assert (
        sel_result.fit_metrics.n_clusters_emergent
        == fit_result.fit_metrics.n_clusters_emergent
    )
    np.testing.assert_array_equal(sel_result.cluster_labels, fit_result.cluster_labels)


def test_linker_fit_assigns_all_manifold_rows_when_frac_subsampled(
    tmp_path: Path,
) -> None:
    frame = _filtered_frame()
    tc = TransformConfig(
        pca_components=8,
        umap_components=3,
        cluster_viz_components=3,
        seed=13,
    )
    mcs = 5
    opt = ClusteringOptimizationConfig(
        min_class_size=5,
        frac=0.5,
        eval_max_rows=None,
        base_seed=13,
    )
    sample = draw_selection_sample(frame, opt, sample_index=0)
    _, manifold = split_by_negative_label(sample, NEGATIVE_LABEL)
    sel_result = fit_manifold_clustering(
        manifold,
        transform_config=tc,
        min_cluster_size=mcs,
        prediction_data=False,
    )

    parquet = tmp_path / "emb.parquet"
    frame.to_parquet(parquet)
    labels_map = {f"e{i}": f"prop_{i}" for i in range(8)}
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
        min_cluster_size=mcs,
        fit_config=LinkerFitConfig(
            min_class_size=5,
            frac=0.5,
            eval_max_rows=None,
            base_seed=13,
            clustering_sample_index=0,
            screener_seed=13,
            projection_screener=ManifoldOovScreenerConfig(enabled=False),
        ),
    )
    fit_report = linker.take_fit_clustering_report()
    assert fit_report is not None
    assert fit_report.n_clusters_emergent == sel_result.fit_metrics.n_clusters_emergent

    _, manifold_full = split_by_negative_label(frame, NEGATIVE_LABEL)
    assert len(fit_report.assignments) == len(manifold_full)
