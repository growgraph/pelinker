import math

import pandas as pd
import pytest

from pelinker.cluster_composition_viz import (
    HDBSCAN_NOISE_CLUSTER_ID,
    build_cluster_composition_df,
    build_emergent_clusters_catalog,
    cluster_entity_mass_summary,
    count_emergent_clusters,
    entity_mention_weights,
    filter_emergent_assignments,
    limit_composition_for_flow_plots,
)
from pelinker.config import ClusterCompositionSnapshot
from pelinker.linker_cluster_training import consensus_cluster_names


def test_entity_mention_weights_inv_sqrt() -> None:
    entities = pd.Series(["a", "a", "b"])
    w = entity_mention_weights(entities)
    assert w.iloc[0] == pytest.approx(1.0 / math.sqrt(2.0))
    assert w.iloc[2] == pytest.approx(1.0)


def test_build_cluster_composition_top_n_other() -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["a", "b", "c", "d"],
            "cluster": [1, 1, 1, 1],
        }
    )
    out = build_cluster_composition_df(assignments, top_n=2, weight_by_entity=False)
    assert len(out) == 3
    assert out["count"].sum() == pytest.approx(4.0)
    assert any(out["entity"].astype(str).str.startswith("Other ("))


def test_exclude_noise_from_composition() -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["a", "b", "c"],
            "cluster": [0, 0, HDBSCAN_NOISE_CLUSTER_ID],
        }
    )
    assert count_emergent_clusters(assignments) == 1
    out = build_cluster_composition_df(assignments, top_n=3, weight_by_entity=False)
    assert set(out["cluster"].astype(int).tolist()) == {0}


def test_max_clusters_caps_facets() -> None:
    rows = []
    for cid in range(10):
        rows.append({"entity": f"p{cid}", "cluster": cid})
    assignments = pd.DataFrame(rows)
    out = build_cluster_composition_df(
        assignments, top_n=1, weight_by_entity=False, max_clusters=3
    )
    assert out["cluster"].nunique() == 3


def test_limit_composition_for_flow_plots_entity_cap() -> None:
    comp = pd.DataFrame(
        {
            "cluster": [0, 0, 0, 1, 1],
            "entity": ["a", "b", "c", "a", "d"],
            "count": [3.0, 2.0, 1.0, 4.0, 1.0],
        }
    )
    limited = limit_composition_for_flow_plots(comp, max_clusters=2, max_entities=2)
    entities = set(limited["entity"].astype(str))
    assert "d" not in entities or "c" not in entities
    assert len(entities) <= 3


def test_emergent_clusters_catalog_entity_ids() -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["alpha", "alpha", "beta"],
            "cluster": [2, 2, 2],
        }
    )
    composition = ClusterCompositionSnapshot(
        global_property_mass={"alpha": 2, "beta": 1},
        cluster_within_fraction={2: {"alpha": 2 / 3, "beta": 1 / 3}},
        cluster_fraction_of_property_mass={
            2: {"alpha": 1.0, "beta": 1.0},
        },
    )
    names = consensus_cluster_names(composition)
    catalog = build_emergent_clusters_catalog(
        composition,
        names,
        assignments,
        min_cluster_size=55,
    )
    assert catalog["n_emergent_clusters"] == 1
    assert catalog["clusters"][0]["entity_id"] == "cluster:2"
    assert catalog["clusters"][0]["top_entities"][0]["entity"] == "alpha"


def test_cluster_entity_mass_summary() -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["a", "b"],
            "cluster": [0, HDBSCAN_NOISE_CLUSTER_ID],
        }
    )
    summary = cluster_entity_mass_summary(assignments)
    assert summary["n_emergent_clusters"] == 1
    assert summary["n_noise_mentions"] == 1
    emergent = filter_emergent_assignments(assignments)
    assert len(emergent) == 1
