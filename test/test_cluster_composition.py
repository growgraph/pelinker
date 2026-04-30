"""Cluster entity-composition metadata and consensus naming after ``Linker.fit``."""

import pandas as pd

from pelinker.config import ClusterCompositionSnapshot
from pelinker.model import (
    cluster_composition_from_training_frame,
    consensus_cluster_names,
)


def test_cluster_composition_mass_and_fractions():
    rows = [("a", 0)] * 10 + [("b", 0)] * 10 + [("a", 1)] * 90 + [("b", 1)] * 10
    df = pd.DataFrame(rows, columns=["entity", "cluster"])
    snap = cluster_composition_from_training_frame(df)
    assert snap.global_property_mass == {"a": 100, "b": 20}
    assert snap.cluster_within_fraction[0] == {"a": 0.5, "b": 0.5}
    assert snap.cluster_within_fraction[1] == {"a": 0.9, "b": 0.1}
    assert snap.cluster_fraction_of_property_mass[0]["a"] == 10 / 100
    assert snap.cluster_fraction_of_property_mass[1]["a"] == 90 / 100


def test_consensus_uniform_hyphen():
    snap = ClusterCompositionSnapshot(
        global_property_mass={"a": 20, "b": 20},
        cluster_within_fraction={0: {"a": 0.5, "b": 0.5}},
        cluster_fraction_of_property_mass={0: {"a": 0.5, "b": 0.5}},
    )
    names = consensus_cluster_names(snap)
    assert names[0] == "a-b"


def test_consensus_duplicate_dominant_gets_suffix():
    snap = ClusterCompositionSnapshot(
        global_property_mass={"x": 100, "y": 10},
        cluster_within_fraction={
            0: {"x": 0.95, "y": 0.05},
            1: {"x": 0.93, "y": 0.07},
        },
        cluster_fraction_of_property_mass={0: {"x": 0.5}, 1: {"x": 0.5}},
    )
    names = consensus_cluster_names(snap)
    assert set(names.values()) == {"x_A", "x_B"}
    assert names[0] != names[1]


def test_consensus_noise_cluster_label():
    snap = ClusterCompositionSnapshot(
        global_property_mass={"a": 1},
        cluster_within_fraction={-1: {"a": 1.0}},
        cluster_fraction_of_property_mass={-1: {"a": 1.0}},
    )
    names = consensus_cluster_names(snap)
    assert names[-1] == "noise"
