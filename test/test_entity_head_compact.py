"""Unit tests for compact entity head and ParametricUMAP dump/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pelinker.config import (
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    LinkerFitConfig,
    ManifoldOovScreenerConfig,
    TransformConfig,
)
from pelinker.entity_head import fit_linear_svc_entity_head, fit_mlp_entity_head
from pelinker.model import Linker
from pelinker.transform import is_parametric_umap


def test_mlp_entity_head_recovers_blob_labels() -> None:
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(6, 4))
    y = rng.integers(0, 6, size=400)
    X = centers[y] + rng.normal(scale=0.15, size=(400, 4))
    head = fit_mlp_entity_head(X, y, hidden_layer_sizes=(64, 32), random_state=0)
    pred, scores = head.predict(X)
    assert pred.shape == (400,)
    assert scores.shape == (400,)
    assert float(np.mean(pred == y)) >= 0.95
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_linear_svc_entity_head_underfit_control() -> None:
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(4, 3))
    y = rng.integers(0, 4, size=200)
    X = centers[y] + rng.normal(scale=0.1, size=(200, 3))
    head = fit_linear_svc_entity_head(X, y, random_state=1)
    pred, scores = head.predict(X)
    assert pred.shape == (200,)
    assert float(np.mean(pred == y)) >= 0.9
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_entity_head_rejects_noise_only() -> None:
    X = np.zeros((10, 3), dtype=np.float64)
    y = np.full(10, -1, dtype=np.int64)
    with pytest.raises(ValueError, match="No non-noise"):
        fit_mlp_entity_head(X, y)


def _compact_fit_parquets(
    tmp_path: Path, n_ent: int = 24
) -> tuple[Path, dict[str, str]]:
    p = tmp_path / "mentions.parquet"
    rows = []
    for k in range(n_ent):
        for pmid in ("1", "2", "3", "4"):
            rows.append(
                {
                    "pmid": pmid,
                    "entity": f"p{k}",
                    "mention": "m",
                    "embed": [float(k), float(k) * 0.1, 1.0, 0.5],
                }
            )
    pd.DataFrame(rows).to_parquet(p)
    labels_map = {f"e{k}": f"p{k}" for k in range(n_ent)}
    return p, labels_map


def test_compact_fit_strips_clusterer_and_sets_mlp_head(tmp_path: Path) -> None:
    parquet, labels_map = _compact_fit_parquets(tmp_path)
    metadata = EmbeddingModelMetadata(
        sources=(EmbeddingSourceSpec(model_type="a", layers_spec="1"),)
    )
    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        parquet,
        transform_config=TransformConfig(
            pca_components=4,
            umap_components=2,
            cluster_viz_components=2,
            manifold_kind="parametric",
            parametric_umap_n_training_epochs=1,
            umap_seed=13,
        ),
        min_cluster_size=2,
        fit_config=LinkerFitConfig(
            batch_size=500,
            predict_mode="compact",
            entity_head_hidden_layers=(32, 16),
            projection_screener=ManifoldOovScreenerConfig(enabled=False),
        ),
    )
    assert linker.predict_mode == "compact"
    assert linker.clusterer is None
    assert linker.entity_head is not None
    assert linker.entity_head.kind == "mlp"
    assert linker.transformer is not None
    assert is_parametric_umap(linker.transformer.umap)
    assert len(linker.vocabulary) > 0


def test_compact_dump_load_roundtrip(tmp_path: Path) -> None:
    parquet, labels_map = _compact_fit_parquets(tmp_path, n_ent=20)
    metadata = EmbeddingModelMetadata(
        sources=(EmbeddingSourceSpec(model_type="a", layers_spec="1"),)
    )
    linker = Linker(labels_map=labels_map, embedding_metadata=metadata)
    linker.fit(
        parquet,
        transform_config=TransformConfig(
            pca_components=4,
            umap_components=2,
            cluster_viz_components=2,
            manifold_kind="parametric",
            parametric_umap_n_training_epochs=1,
            umap_seed=7,
        ),
        min_cluster_size=2,
        fit_config=LinkerFitConfig(
            batch_size=500,
            predict_mode="compact",
            entity_head_hidden_layers=(32, 16),
            projection_screener=ManifoldOovScreenerConfig(enabled=False),
        ),
    )
    _ = linker.take_fit_clustering_report()
    model_path = tmp_path / "compact_linker"
    linker.dump(model_path)
    gz = Path(str(model_path) + ".gz")
    assert gz.is_file()
    sidecar = Path(str(model_path) + ".parametric_umap")
    assert sidecar.is_dir()

    loaded = Linker.load(model_path)
    assert loaded.entity_head is not None
    assert loaded.clusterer is None
    assert loaded.transformer is not None
    assert is_parametric_umap(loaded.transformer.umap)
    assert loaded.transformer.umap is not None

    # Sidecar + joblib must stay far below legacy UMAP+HDBSCAN (~tens of MB).
    gz_bytes = gz.stat().st_size
    sidecar_bytes = sum(p.stat().st_size for p in sidecar.rglob("*") if p.is_file())
    total = gz_bytes + sidecar_bytes
    assert total < 5_000_000, f"compact artifact too large: {total} bytes"

    # Transform + head must work after reload (encoder was restored from sidecar).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4)).astype(np.float64)
    assert loaded.transformer.umap is not None
    coords = loaded.transformer.umap.transform(X)
    pred, scores = loaded.entity_head.predict(coords)
    assert pred.shape == (8,)
    assert scores.shape == (8,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_mlp_entity_head_recovers_non_linear_xor_like_regions() -> None:
    """HDBSCAN-like non-convex regions: LinearSVC underfits; MLP recovers."""
    rng = np.random.default_rng(42)
    n = 600
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    # Four quadrant blobs with XOR-style label pairing (non-linearly separable).
    y = np.zeros(n, dtype=np.int64)
    y[(X[:, 0] >= 0) & (X[:, 1] >= 0)] = 0
    y[(X[:, 0] < 0) & (X[:, 1] < 0)] = 0
    y[(X[:, 0] >= 0) & (X[:, 1] < 0)] = 1
    y[(X[:, 0] < 0) & (X[:, 1] >= 0)] = 1
    X = X + rng.normal(scale=0.05, size=X.shape)

    mlp = fit_mlp_entity_head(X, y, hidden_layer_sizes=(64, 32, 32), random_state=42)
    lin = fit_linear_svc_entity_head(X, y, random_state=42)
    mlp_acc = float(np.mean(mlp.predict(X)[0] == y))
    lin_acc = float(np.mean(lin.predict(X)[0] == y))
    assert mlp_acc >= 0.90
    assert mlp_acc > lin_acc + 0.15
