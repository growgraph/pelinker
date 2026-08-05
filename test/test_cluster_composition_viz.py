import math

import pandas as pd
import pytest

from pelinker.cluster_composition_viz import (
    HDBSCAN_NOISE_CLUSTER_ID,
    aggregate_cluster_entity_mass,
    build_cluster_composition_df,
    cluster_entity_mass_summary,
    cluster_score_percentile_summary,
    count_emergent_clusters,
    entity_mention_weights,
    filter_emergent_assignments,
    limit_composition_for_flow_plots,
    limit_entity_flow_for_plots,
)


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


def test_include_noise_in_composition() -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["a", "b", "c"],
            "cluster": [0, 0, HDBSCAN_NOISE_CLUSTER_ID],
        }
    )
    out = build_cluster_composition_df(
        assignments, top_n=3, weight_by_entity=False, exclude_noise=False
    )
    assert set(out["cluster"].astype(int).tolist()) == {0, HDBSCAN_NOISE_CLUSTER_ID}


def test_max_clusters_still_keeps_noise() -> None:
    rows = [{"entity": f"p{cid}", "cluster": cid} for cid in range(10)]
    rows.append({"entity": "noise_ent", "cluster": HDBSCAN_NOISE_CLUSTER_ID})
    assignments = pd.DataFrame(rows)
    out = build_cluster_composition_df(
        assignments,
        top_n=1,
        weight_by_entity=False,
        exclude_noise=False,
        max_clusters=3,
    )
    assert HDBSCAN_NOISE_CLUSTER_ID in set(out["cluster"].astype(int))
    emergent = set(out["cluster"].astype(int)) - {HDBSCAN_NOISE_CLUSTER_ID}
    assert len(emergent) == 3


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


def test_limit_composition_for_flow_plots_min_within_cluster_fraction() -> None:
    comp = pd.DataFrame(
        {
            "cluster": [0, 0, 0],
            "entity": ["a", "b", "c"],
            "count": [10.0, 1.0, 0.5],
        }
    )
    limited = limit_composition_for_flow_plots(
        comp, max_clusters=1, max_entities=10, min_within_cluster_fraction=0.10
    )
    entities = set(limited["entity"].astype(str))
    assert "a" in entities
    assert "b" not in entities
    assert "c" not in entities
    assert any(e.startswith("Other (") for e in entities)


def test_limit_entity_flow_for_plots_caps_and_drops() -> None:
    flow = pd.DataFrame(
        {
            "cluster": [0, 0, 0, 1, 1],
            "entity": ["a", "b", "c", "a", "d"],
            "count": [3.0, 2.0, 1.0, 4.0, 1.0],
        }
    )
    limited = limit_entity_flow_for_plots(flow, max_clusters=2, max_entities=2)
    entities = set(limited["entity"].astype(str))
    assert entities == {"a", "b"}
    assert not any(e.startswith("Other (") for e in entities)


def test_limit_entity_flow_for_plots_min_within_cluster_fraction_drops() -> None:
    flow = pd.DataFrame(
        {
            "cluster": [0, 0, 0],
            "entity": ["a", "b", "c"],
            "count": [10.0, 1.0, 0.5],
        }
    )
    limited = limit_entity_flow_for_plots(
        flow, max_clusters=1, max_entities=10, min_within_cluster_fraction=0.10
    )
    entities = set(limited["entity"].astype(str))
    assert entities == {"a"}
    assert not any(e.startswith("Other (") for e in entities)


def test_aggregate_cluster_entity_mass_excludes_noise() -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["a", "a", "b", "c"],
            "cluster": [0, 0, 0, HDBSCAN_NOISE_CLUSTER_ID],
        }
    )
    out = aggregate_cluster_entity_mass(
        assignments, weight_by_entity=False, exclude_noise=True
    )
    assert set(out.columns) == {"cluster", "entity", "count"}
    assert set(out["cluster"].astype(int)) == {0}
    by_entity = out.set_index("entity")["count"].to_dict()
    assert by_entity["a"] == pytest.approx(2.0)
    assert by_entity["b"] == pytest.approx(1.0)
    assert "c" not in by_entity


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


def test_cluster_score_percentile_summary() -> None:
    assignments = pd.DataFrame(
        {
            "entity": ["a", "b", "c", "d"],
            "cluster": [0, 0, HDBSCAN_NOISE_CLUSTER_ID, HDBSCAN_NOISE_CLUSTER_ID],
            "cluster_score": [0.9, 0.8, 0.1, 0.2],
        }
    )
    summary = cluster_score_percentile_summary(assignments)
    assert summary["emergent"]["p50"] == pytest.approx(0.85)
    assert summary["noise"]["p50"] == pytest.approx(0.15)
    assert summary["overall"]["p50"] == pytest.approx(0.5)
