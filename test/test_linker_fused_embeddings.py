"""Fused embedding paths for ``Linker.fit`` (property-level concat)."""

import pandas as pd

from pelinker.config import EmbeddingModelMetadata, EmbeddingSourceSpec
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
        rows1.append(
            {"pmid": "1", "property": f"p{k}", "mention": "m", "embed": [float(k), 1.0]}
        )
        rows2.append(
            {
                "pmid": "1",
                "property": f"p{k}",
                "mention": "m",
                "embed": [0.5, float(k) * 0.1],
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
        min_cluster_size=5,
    )
    assert linker.transformer is not None
    assert len(linker.vocabulary) == n_ent
    assert linker.transformer.pca is not None
    assert linker.transformer.pca.n_features_in_ == 4


def test_estimate_model_clustering_multi_file_paths(tmp_path):
    from pelinker.analysis import estimate_model_clustering
    from pelinker.config import ClusteringOptimizationConfig
    from numpy.random import RandomState

    p1 = tmp_path / "s0.parquet"
    p2 = tmp_path / "s1.parquet"
    n = 25
    pd.DataFrame(
        {
            "pmid": [str(i) for i in range(n)],
            "property": ["foo"] * n,
            "mention": ["m"] * n,
            "embed": [[float(i), 0.0] for i in range(n)],
        }
    ).to_parquet(p1)
    pd.DataFrame(
        {
            "pmid": [str(i) for i in range(n)],
            "property": ["foo"] * n,
            "mention": ["m"] * n,
            "embed": [[0.0, float(i)] for i in range(n)],
        }
    ).to_parquet(p2)

    report = estimate_model_clustering(
        transform_config=TransformConfig(
            pca_components=4, umap_components=2, umap_viz_components=2
        ),
        file_paths=[p1, p2],
        optimization_config=ClusteringOptimizationConfig(
            min_class_size=20,
            max_scale=40,
            rns=RandomState(0),
            frac=1.0,
        ),
    )
    assert report is not None
    assert report.df is not None
    assert "u_00" in report.df.columns
