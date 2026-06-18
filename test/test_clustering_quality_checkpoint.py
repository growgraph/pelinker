"""Tests for model selection checkpoint and reporting round-trip helpers."""

from __future__ import annotations

import json
import pathlib

import pytest

from pelinker.model_selection_checkpoint import (
    CHECKPOINT_VERSION,
    ModelSelectionCheckpoint,
    combination_key_from_members,
    compute_run_fingerprint,
    fingerprint_config_from_cli,
    load_checkpoint,
    new_checkpoint,
    reconcile_fusion_checkpoint_params,
    save_checkpoint_atomic,
)
from pelinker.reporting import (
    ClusteringSearchSummaryRow,
    HyperparameterSearchStats,
    MeanWithUncertainty,
    clustering_search_summary_row_from_flat_dict,
)


def test_combination_key_sorted_stable() -> None:
    a = combination_key_from_members([("b", "L2"), ("a", "L1")])
    b = combination_key_from_members([("a", "L1"), ("b", "L2")])
    assert a == b == "2:a/L1+b/L2"
    assert combination_key_from_members([("m", "l")]) == "1:m/l"


def test_checkpoint_save_load_roundtrip(tmp_path: pathlib.Path) -> None:
    fp = compute_run_fingerprint({"a": 1, "b": 2})
    ckpt = new_checkpoint(fp)
    ckpt.completed_combinations.append("1:m/l")
    ckpt.summaries_by_key["1:m/l"] = {"model": "m", "layer": "l", "best_score": 0.9}
    path = tmp_path / "st.json"
    save_checkpoint_atomic(path, ckpt)
    loaded = load_checkpoint(path)
    assert loaded.run_fingerprint == fp
    assert loaded.completed_combinations == ["1:m/l"]
    assert loaded.summaries_by_key["1:m/l"]["best_score"] == 0.9


def test_checkpoint_save_load_roundtrip_gzip(tmp_path: pathlib.Path) -> None:
    fp = compute_run_fingerprint({"a": 1, "b": 2})
    ckpt = new_checkpoint(fp)
    ckpt.completed_combinations.append("1:m/l")
    ckpt.summaries_by_key["1:m/l"] = {"model": "m", "layer": "l", "best_score": 0.9}
    path = tmp_path / "st.json.gz"
    save_checkpoint_atomic(path, ckpt)
    loaded = load_checkpoint(path)
    assert loaded.run_fingerprint == fp
    assert loaded.completed_combinations == ["1:m/l"]
    assert loaded.summaries_by_key["1:m/l"]["best_score"] == 0.9


def test_load_wrong_version_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"version": 999, "run_fingerprint": "x"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        load_checkpoint(path)


def test_fingerprint_differs_for_different_config() -> None:
    a = compute_run_fingerprint({"seed": 1})
    b = compute_run_fingerprint({"seed": 2})
    assert a != b


def test_reconcile_drops_fusion_when_fusion_params_change(
    tmp_path: pathlib.Path,
) -> None:
    fp = compute_run_fingerprint({"k": 1})
    ckpt = new_checkpoint(fp)
    ckpt.checkpoint_fusion_pairs = 0
    ckpt.checkpoint_fusion_triples = 0
    ckpt.completed_combinations = ["1:a/l", "2:a/L1+b/L2"]
    ckpt.summaries_by_key = {
        "1:a/l": {"model": "a", "layer": "l"},
        "2:a/L1+b/L2": {"model": "fusion2", "layer": "a/L1+b/L2"},
    }
    n = reconcile_fusion_checkpoint_params(ckpt, fusion_pairs=5, fusion_triples=0)
    assert n == 1
    assert ckpt.completed_combinations == ["1:a/l"]
    assert list(ckpt.summaries_by_key.keys()) == ["1:a/l"]
    assert ckpt.checkpoint_fusion_pairs == 5


def test_fingerprint_depends_on_input_dir_not_fusion_cli(
    tmp_path: pathlib.Path,
) -> None:
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    common = dict(
        umap_dim=8,
        pca_components=100,
        cluster_viz_method="pca",
        min_class_size=20,
        seed=1,
        frac=0.1,
        eval_max_rows=100_000,
        n_embedding_batches=None,
        batch_size=1000,
        prefix="p",
        n_sample=1,
        selected_labels_kb_path=None,
        max_scale=60,
    )
    fa = compute_run_fingerprint(fingerprint_config_from_cli(input_dir=a_dir, **common))
    fb = compute_run_fingerprint(fingerprint_config_from_cli(input_dir=b_dir, **common))
    assert fa != fb


def test_summary_flat_dict_round_trip() -> None:
    row = ClusteringSearchSummaryRow(
        model="m",
        layer="l",
        hyperparameters=HyperparameterSearchStats(
            min_cluster_size=MeanWithUncertainty(mean=10.0, std=1.0),
        ),
        number_properties=MeanWithUncertainty(mean=5.0, std=0.0),
        n_clusters_emergent=MeanWithUncertainty(mean=3.0, std=0.5),
        dbcv=MeanWithUncertainty(mean=0.44, std=0.01),
        ari=MeanWithUncertainty(mean=0.8, std=0.05),
    )
    flat = row.to_flat_dict()
    back = clustering_search_summary_row_from_flat_dict(flat)
    assert back.model == row.model
    assert back.layer == row.layer
    assert back.dbcv.mean == row.dbcv.mean
    assert back.ari is not None
    assert back.ari.mean == 0.8


def test_checkpoint_to_json_dict_sorted_keys(tmp_path: pathlib.Path) -> None:
    ckpt = ModelSelectionCheckpoint(
        version=CHECKPOINT_VERSION,
        run_fingerprint="ab",
        summaries_by_key={"z": {"model": "z"}, "a": {"model": "a"}},
    )
    d = ckpt.to_json_dict()
    keys = list(d["summaries_by_key"].keys())
    assert keys == ["a", "z"]
