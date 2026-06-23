from datetime import date

import pathlib

import pytest

from pelinker.config import ClusteringOptimizationConfig, KBConfig


def test_kb_config_ok():
    c = KBConfig(
        name="test-kb",
        version="1.2.3",
        created_at=date(2025, 1, 15),
        description="demo",
        entity_count=42,
    )
    assert c.entity_count == 42


def test_kb_config_entity_count_optional():
    c = KBConfig(name="x", version="0.0.1", created_at=date.today())
    assert c.entity_count is None


def test_kb_config_semver_prerelease_build():
    KBConfig(
        name="x",
        version="1.0.0-rc.1+build.9",
        created_at=date.today(),
    )


@pytest.mark.parametrize(
    "version",
    ["", "1", "1.0", "v1.0.0", "01.0.0", "1.0.a"],
)
def test_kb_config_rejects_bad_semver(version: str):
    with pytest.raises(ValueError):
        KBConfig(name="x", version=version, created_at=date.today())


def test_kb_config_rejects_empty_name():
    with pytest.raises(ValueError):
        KBConfig(name="   ", version="1.0.0", created_at=date.today())


def test_kb_config_rejects_negative_entity_count():
    with pytest.raises(ValueError):
        KBConfig(
            name="x",
            version="1.0.0",
            created_at=date.today(),
            entity_count=-1,
        )


def test_clustering_optimization_resolved_min_scale_default() -> None:
    c = ClusteringOptimizationConfig(min_class_size=20, max_scale=100)
    assert c.resolved_min_scale() == 10


def test_clustering_optimization_resolved_min_scale_explicit() -> None:
    c = ClusteringOptimizationConfig(
        min_class_size=20, min_scale=5, max_scale=100, clustering_grid_step=5
    )
    assert c.resolved_min_scale() == 5


def test_clustering_optimization_rejects_max_below_resolved_min() -> None:
    with pytest.raises(ValueError, match="max_scale must be >="):
        ClusteringOptimizationConfig(min_class_size=20, min_scale=50, max_scale=40)


def test_clustering_optimization_rejects_invalid_clustering_sample_rows() -> None:
    with pytest.raises(ValueError, match="clustering_sample_rows"):
        ClusteringOptimizationConfig(clustering_sample_rows=0)


def test_clustering_optimization_rejects_negative_cluster_count_reward() -> None:
    with pytest.raises(ValueError, match="grid_cluster_count_reward"):
        ClusteringOptimizationConfig(grid_cluster_count_reward=-0.1)


def test_clustering_optimization_rejects_invalid_grid_n_entities() -> None:
    with pytest.raises(ValueError, match="grid_n_entities"):
        ClusteringOptimizationConfig(grid_n_entities=0)


def test_linker_fit_config_load_fields_validate() -> None:
    from pelinker.config import LinkerFitConfig

    LinkerFitConfig(
        drop_rare_entities=True,
        min_mentions_per_entity=5,
        max_mentions_per_entity=100,
        max_mentions_negative=50,
    )


def test_linker_fit_config_rejects_invalid_max_mentions() -> None:
    from pelinker.config import LinkerFitConfig

    with pytest.raises(ValueError, match="max_mentions_per_entity"):
        LinkerFitConfig(max_mentions_per_entity=0)


def test_fingerprint_includes_mention_load_fields(tmp_path: pathlib.Path) -> None:
    from pelinker.model_selection_checkpoint import fingerprint_config_from_cli

    d = tmp_path / "in"
    d.mkdir()
    fp = fingerprint_config_from_cli(
        input_dir=d,
        umap_dim=8,
        pca_components=100,
        cluster_viz_method="pca",
        min_class_size=20,
        seed=1,
        pca_seed=13,
        umap_seed=None,
        clustering_sample_rows=None,
        batch_size=1000,
        prefix="p",
        n_sample=1,
        selected_labels_kb_path=None,
        max_scale=60,
        drop_rare_entities=True,
        min_mentions_per_entity=15,
        max_mentions_per_entity=200,
        max_mentions_negative=1000,
        mention_cap_seed=99,
    )
    assert fp["drop_rare_entities"] is True
    assert fp["max_mentions_per_entity"] == 200
    assert fp["mention_cap_seed"] == 99
