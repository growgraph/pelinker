"""The selection objective and the fit score agreement in the configured class view."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from pelinker.clustering.fit import fit_hdbscan_on_umap
from pelinker.clustering.grid import evaluate_cluster_size_grid
from pelinker.core.config import (
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    LinkerFitConfig,
    ManifoldOovScreenerConfig,
    TransformConfig,
)
from pelinker.core.onto import NEGATIVE_LABEL
from pelinker.kb.classes import ENTITY_CLASS_COLUMN, ClassView, file_sha256
from pelinker.kb.kb_out import kb_out_to_reldir_map
from pelinker.model import Linker
from pelinker.search.selection import apply_class_view, load_selection_frame


def _pairs_kb(tmp_path: Path) -> Path:
    path = tmp_path / "kb.pairs.csv"
    pd.DataFrame(
        {
            "entity_id": ["RO.1", "RO.2", "PEL.3"],
            "label": ["regulates", "regulated by", "inhibits"],
            "is_canonical": [True, False, True],
            "canonical_entity_id": ["RO.1", "RO.1", "PEL.3"],
            "is_symmetric": [False, False, False],
        }
    ).to_csv(path, index=False)
    return path


def _two_voice_blobs(n: int = 30) -> pd.DataFrame:
    """Two tight blobs: active ``inhibits`` and passive ``inhibits``.

    The KB has no converse entry for ``inhibits``, so both blobs carry the same raw label
    and differ only in ``direction`` — the case where raw labels hide the voice split.
    """
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 0.05, size=(n, 2))
    b = rng.normal(5.0, 0.05, size=(n, 2))
    return pd.DataFrame(
        {
            "u0": np.concatenate([a[:, 0], b[:, 0]]),
            "u1": np.concatenate([a[:, 1], b[:, 1]]),
            "entity": ["inhibits"] * (2 * n),
            "direction": ["forward"] * n + ["inverse"] * n,
        }
    )


def test_grid_ari_follows_the_class_view(tmp_path: Path) -> None:
    frame = _two_voice_blobs()
    cfg = ClusteringOptimizationConfig(
        class_view="reldir", class_kb_path=str(_pairs_kb(tmp_path))
    )

    raw = evaluate_cluster_size_grid(frame[["u0", "u1", "entity"]], ["u0", "u1"], [5])
    viewed_frame = apply_class_view(frame, cfg)
    viewed = evaluate_cluster_size_grid(
        viewed_frame[["u0", "u1", "entity", ENTITY_CLASS_COLUMN]], ["u0", "u1"], [5]
    )

    # One raw class split into two clusters scores zero; in reldir the split is exact.
    assert raw["ari"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert viewed["ari"].iloc[0] == pytest.approx(1.0)


def test_fit_metrics_ari_follows_the_class_view(tmp_path: Path) -> None:
    frame = apply_class_view(
        _two_voice_blobs(),
        ClusteringOptimizationConfig(
            class_view="reldir", class_kb_path=str(_pairs_kb(tmp_path))
        ),
    )
    umap = frame[["u0", "u1"]].to_numpy(dtype=np.float32)

    viewed = fit_hdbscan_on_umap(umap, frame, 5).fit_metrics
    raw = fit_hdbscan_on_umap(umap, frame.drop(columns=[ENTITY_CLASS_COLUMN]), 5)

    assert viewed.ari == pytest.approx(1.0)
    assert raw.fit_metrics.ari == pytest.approx(0.0, abs=1e-9)


def test_a_view_without_a_kb_is_refused() -> None:
    with pytest.raises(ValueError, match="needs class_kb_path"):
        apply_class_view(
            _two_voice_blobs(), ClusteringOptimizationConfig(class_view="rel")
        )


def test_raw_view_leaves_the_frame_untouched() -> None:
    frame = _two_voice_blobs()
    out = apply_class_view(frame, ClusteringOptimizationConfig())
    assert out is frame


def _embedding_frame(dim: int = 12, n: int = 30, n_neg: int = 40) -> pd.DataFrame:
    """Three separated blobs: active regulates, passive regulates (its converse entry),
    active inhibits; plus synthetic negatives."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    blobs = [
        ("regulates", "forward", 0),
        ("regulated by", "forward", 1),
        ("inhibits", "forward", 2),
    ]
    for label, direction, axis in blobs:
        centre = np.zeros(dim, dtype=np.float32)
        centre[axis] = 6.0
        for i in range(n):
            rows.append(
                {
                    "pmid": f"{label}-{i // 3}",
                    "entity": label,
                    "mention": f"{label}-{i}",
                    "direction": direction,
                    "embed": (centre + rng.normal(0, 0.1, dim)).astype(np.float32),
                }
            )
    for j in range(n_neg):
        rows.append(
            {
                "pmid": f"neg-{j}",
                "entity": NEGATIVE_LABEL,
                "mention": f"n{j}",
                "direction": None,
                "embed": rng.normal(0, 1.0, dim).astype(np.float32) - 4.0,
            }
        )
    return pd.DataFrame(rows)


def _fit(tmp_path: Path, *, class_view: ClassView) -> Linker:
    kb_path = _pairs_kb(tmp_path)
    parquet = tmp_path / f"emb-{class_view}.parquet"
    _embedding_frame().to_parquet(parquet)
    tc = TransformConfig(
        pca_components=8,
        umap_components=3,
        cluster_viz_components=3,
        pca_seed=13,
        umap_seed=13,
    )
    linker = Linker(
        labels_map={"RO.1": "regulates", "RO.2": "regulated by", "PEL.3": "inhibits"},
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
            base_seed=13,
            screener_seed=13,
            projection_screener=ManifoldOovScreenerConfig(enabled=False),
            predict_mode="legacy",
            class_view=class_view,
            class_kb_path=None if class_view == "raw" else str(kb_path),
        ),
    )
    return linker


def _component_labels(linker: Linker) -> set[str]:
    assert linker.kb_out_catalog is not None
    catalog = cast(dict[str, Any], linker.kb_out_catalog)
    return {
        c["entity"] for cluster in catalog["clusters"] for c in cluster["components"]
    }


def test_reldir_fit_reads_composition_between_relations_with_a_direction(
    tmp_path: Path,
) -> None:
    linker = _fit(tmp_path, class_view="reldir")

    # The converse entry never appears as its own entity: its mentions are regulates.
    assert _component_labels(linker) == {"regulates", "inhibits"}
    assert sorted(set(linker.cluster_direction.values())) == ["forward", "inverse"]

    assert linker.kb_out_catalog is not None
    catalog = cast(dict[str, Any], linker.kb_out_catalog)
    fit_prov = catalog["provenance"]["fit"]
    assert fit_prov["class_view"] == "reldir"
    assert fit_prov["class_kb_sha256"] == file_sha256(tmp_path / "kb.pairs.csv")

    bridge = kb_out_to_reldir_map(catalog)
    assert ("RO.1", "inverse") in set(bridge.values())
    assert ("RO.1", "forward") in set(bridge.values())


def test_raw_fit_keeps_matched_labels_and_no_direction(tmp_path: Path) -> None:
    linker = _fit(tmp_path, class_view="raw")

    assert "regulated by" in _component_labels(linker)
    assert linker.cluster_direction == {}


def test_load_selection_frame_applies_the_view(tmp_path: Path) -> None:
    parquet = tmp_path / "emb.parquet"
    _embedding_frame().to_parquet(parquet)
    cfg = ClusteringOptimizationConfig(
        class_view="reldir", class_kb_path=str(_pairs_kb(tmp_path))
    )

    loaded = load_selection_frame(file_path=parquet, config=cfg)

    assert loaded is not None
    kb_rows = loaded.loc[loaded["entity"] != NEGATIVE_LABEL]
    assert set(kb_rows[ENTITY_CLASS_COLUMN]) == {
        "regulates|forward",
        "regulates|inverse",
        "inhibits|forward",
    }


def _fingerprint(
    class_view: ClassView = "raw", class_kb_path: Path | None = None
) -> dict[str, Any]:
    from pelinker.search.checkpoint_base import shared_fingerprint_fields

    return shared_fingerprint_fields(
        cluster_viz_method="pca",
        min_class_size=20,
        seed=13,
        pca_seed=13,
        umap_seed=None,
        clustering_sample_rows=None,
        batch_size=1000,
        n_sample=3,
        selected_labels_kb_path=None,
        max_scale=100,
        class_view=class_view,
        class_kb_path=class_kb_path,
    )


def test_a_raw_run_keeps_the_fingerprint_it_had_before_views_existed() -> None:
    assert _fingerprint() == _fingerprint(class_view="raw")
    assert "class_view" not in _fingerprint()


def test_a_view_enters_the_fingerprint_by_kb_content(tmp_path: Path) -> None:
    kb_path = _pairs_kb(tmp_path)
    viewed = _fingerprint(class_view="reldir", class_kb_path=kb_path)
    assert viewed["class_view"] == "reldir"
    assert viewed["class_kb_sha256"] == file_sha256(kb_path)

    # The same path with different content is a different run.
    kb_path.write_text(kb_path.read_text() + "PEL.9,binds,False,PEL.9,False\n")
    assert _fingerprint(class_view="reldir", class_kb_path=kb_path) != viewed
