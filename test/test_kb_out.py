"""KB-out catalog naming, entity ids, and ambiguity indices."""

from __future__ import annotations

import pandas as pd

from pelinker.config import ClusterCompositionSnapshot, KBConfig
from pelinker.kb_out import (
    KbOutFitProvenance,
    KbOutNamingConfig,
    build_kb_out_catalog,
    cluster_labels_from_catalog,
    find_ambiguous_entities,
    find_split_mentions,
    format_cluster_display_name,
    format_kb_out_entity_id,
)
from pelinker.reporting import read_kb_out_json, write_kb_out_json


def test_format_cluster_display_name_top_components() -> None:
    name = format_cluster_display_name(
        {"alpha": 0.67, "beta": 0.33, "gamma": 0.05},
        min_fraction=0.05,
        top_n=3,
    )
    assert name == "alpha-67--beta-33--gamma-5"


def test_format_cluster_display_name_fallback_top_one() -> None:
    name = format_cluster_display_name(
        {"alpha": 0.99, "beta": 0.01},
        min_fraction=0.05,
        top_n=3,
    )
    assert name == "alpha-99"


def test_format_kb_out_entity_id() -> None:
    assert format_kb_out_entity_id("mykb_0.1.0", 2) == "mykb_0.1.0::C0002"


def test_build_kb_out_catalog_disambiguates_display_names() -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["alpha", "alpha", "beta", "beta"],
            "cluster": [0, 0, 1, 1],
            "pmid": ["1", "1", "2", "2"],
            "mention": ["m", "m", "n", "n"],
            "cluster_score": [0.9, 0.85, 0.8, 0.7],
        }
    )
    composition = ClusterCompositionSnapshot(
        global_property_mass={"alpha": 2, "beta": 2},
        cluster_within_fraction={
            0: {"alpha": 1.0},
            1: {"beta": 1.0},
        },
        cluster_fraction_of_property_mass={
            0: {"alpha": 1.0},
            1: {"beta": 1.0},
        },
    )
    kb_config = KBConfig(
        name="mykb",
        version="0.1.0",
        created_at=__import__("datetime").date(2026, 1, 1),
    )
    catalog = build_kb_out_catalog(
        composition,
        assignments,
        {"e1": "alpha", "e2": "beta"},
        kb_config=kb_config,
        fit_provenance=KbOutFitProvenance(min_cluster_size=2),
        naming=KbOutNamingConfig(min_fraction=0.05, top_n=3),
    )
    names = {c["display_name"] for c in catalog["clusters"]}
    assert len(names) == 2
    assert catalog["clusters"][0]["entity_id"].startswith("mykb_0.1.0::C")
    assert (
        catalog["labels_map"][catalog["clusters"][0]["entity_id"]]
        == catalog["clusters"][0]["display_name"]
    )
    labels = cluster_labels_from_catalog(catalog, label_kind="display")
    assert labels[0] == catalog["clusters"][0]["display_name"]
    assert "noise" in catalog
    assert catalog["noise"]["n_mentions"] == 0
    assert -1 not in {int(c["cluster_id"]) for c in catalog["clusters"]}
    assert "-1" not in catalog["cluster_id_to_entity_id"]


def test_build_kb_out_catalog_noise_diagnostic() -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["alpha", "alpha", "gamma"],
            "cluster": [0, 0, -1],
            "pmid": ["1", "1", "2"],
            "mention": ["m", "m", "n"],
            "cluster_score": [0.9, 0.8, 0.05],
        }
    )
    composition = ClusterCompositionSnapshot(
        global_property_mass={"alpha": 2, "gamma": 1},
        cluster_within_fraction={
            0: {"alpha": 1.0},
            -1: {"gamma": 1.0},
        },
        cluster_fraction_of_property_mass={
            0: {"alpha": 1.0},
            -1: {"gamma": 1.0},
        },
    )
    catalog = build_kb_out_catalog(
        composition,
        assignments,
        {"e1": "alpha", "e2": "gamma"},
        kb_config=None,
        fit_provenance=KbOutFitProvenance(min_cluster_size=2),
    )
    assert catalog["n_noise_mentions"] == 1
    assert catalog["noise"]["top_entities"][0]["entity"] == "gamma"
    assert "-1" not in catalog["cluster_id_to_entity_id"]
    labels = cluster_labels_from_catalog(catalog)
    assert labels[-1] == "noise"


def test_find_ambiguous_entities_polysemy() -> None:
    membership = {
        "alpha": [
            {
                "cluster_id": 0,
                "entity_id": "kb::C0000",
                "within_cluster_fraction": 0.6,
                "capture_fraction": 0.6,
            },
            {
                "cluster_id": 1,
                "entity_id": "kb::C0001",
                "within_cluster_fraction": 0.4,
                "capture_fraction": 0.4,
            },
        ]
    }
    ambiguous = find_ambiguous_entities(membership, min_capture=0.10)
    assert len(ambiguous) == 1
    assert ambiguous[0]["entity"] == "alpha"


def test_find_split_mentions_homonymy() -> None:
    assignments = pd.DataFrame(
        {
            "pmid": ["1", "1"],
            "mention": ["bank", "bank"],
            "a_abs": [10, 10],
            "entity": ["alpha", "beta"],
            "cluster": [0, 1],
        }
    )
    splits = find_split_mentions(assignments)
    assert len(splits) == 1
    assert splits[0]["n_clusters"] == 2


def test_kb_out_json_round_trip(tmp_path) -> None:
    path = tmp_path / "kb_out.json"
    payload = {
        "schema": "pelinker.kb_out.v1",
        "kb_out": {"entity_count": 1},
        "provenance": {"kb_in": {}, "fit": {"min_cluster_size": 2}},
        "naming": {"min_fraction": 0.05, "top_n": 3, "style": "weighted_dash"},
        "n_emergent_clusters": 1,
        "n_noise_mentions": 0,
        "noise_fraction": 0.0,
        "noise": {
            "label": "noise",
            "n_mentions": 0,
            "noise_fraction": 0.0,
            "top_entities": [],
            "cluster_score_percentiles": {},
        },
        "clusters": [],
        "entity_membership": {},
        "ambiguous_entities": [],
        "split_mentions": [],
        "labels_map": {},
        "cluster_id_to_entity_id": {},
    }
    write_kb_out_json(path, payload)
    loaded = read_kb_out_json(path)
    assert loaded["schema"] == "pelinker.kb_out.v1"
    assert loaded["n_emergent_clusters"] == 1
