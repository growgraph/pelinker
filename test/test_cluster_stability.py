"""Cluster-identity stability across bootstrap draws."""

from __future__ import annotations

import pytest

from pelinker.clustering.stability import (
    analyze_stability,
    jaccard,
    match_two_draws,
)


def _draw(**groups: list[int]) -> dict[int, int]:
    """``cluster_name=[row, ...]`` → ``row -> cluster_id`` (names must be ``c<int>``)."""
    out: dict[int, int] = {}
    for name, rows in groups.items():
        cid = int(name[1:])
        for row in rows:
            out[row] = cid
    return out


def test_jaccard_basics() -> None:
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert jaccard({1, 2}, {3, 4}) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard({1, 2, 3, 4}, {3, 4}) == 0.5


def test_identical_draws_match_perfectly() -> None:
    a = _draw(c0=[1, 2, 3], c1=[4, 5, 6])

    m = match_two_draws(a, dict(a))

    assert m.n_shared_rows == 6
    assert {(x, y) for x, y, _ in m.matches} == {(0, 0), (1, 1)}
    assert all(j == 1.0 for _, _, j in m.matches)
    assert m.ari_shared == 1.0


def test_renumbered_clusters_still_match() -> None:
    """Cluster ids are arbitrary per fit; identity is membership, not the label."""
    a = _draw(c0=[1, 2, 3], c1=[4, 5, 6])
    b = _draw(c7=[4, 5, 6], c9=[1, 2, 3])

    m = match_two_draws(a, b)

    assert {(x, y) for x, y, _ in m.matches} == {(0, 9), (1, 7)}
    assert m.ari_shared == 1.0


def test_noise_rows_are_excluded() -> None:
    a = _draw(c0=[1, 2, 3])
    a[99] = -1
    b = _draw(c0=[1, 2, 3])
    b[99] = -1

    m = match_two_draws(a, b)

    assert m.n_shared_rows == 3  # the noise row is not compared
    assert m.matches[0][2] == 1.0


def test_only_shared_rows_are_compared() -> None:
    """Bootstrap draws overlap partially; unseen rows must not read as churn."""
    a = _draw(c0=[1, 2, 3, 4])
    b = _draw(c0=[3, 4, 5, 6])

    m = match_two_draws(a, b)

    assert m.n_shared_rows == 2
    # On the shared rows {3,4} the two clusters agree completely.
    assert m.matches[0][2] == 1.0


def test_a_split_cluster_shows_partial_overlap() -> None:
    a = _draw(c0=[1, 2, 3, 4])
    b = _draw(c0=[1, 2], c1=[3, 4])

    m = match_two_draws(a, b)

    assert len(m.matches) == 1  # one-to-one: c0 can only claim one half
    assert m.matches[0][2] == 0.5
    assert m.ari_shared is not None and m.ari_shared < 1.0


def test_stability_report_over_three_stable_draws() -> None:
    draws = [_draw(c0=[1, 2, 3], c1=[4, 5, 6]) for _ in range(3)]

    report = analyze_stability(draws).to_jsonable()

    assert report["n_draws"] == 3
    assert report["n_pairs"] == 3
    assert report["cluster_count_mean"] == 2.0
    assert report["cluster_count_std"] == 0.0
    assert report["mean_pair_jaccard"] == 1.0
    assert report["mean_cross_draw_ari"] == 1.0
    assert report["persistent_fraction"] == 1.0


def test_stability_report_detects_churn_at_a_stable_count() -> None:
    """The failure the cluster count cannot see: same k, different members.

    Four clusters, not two — with few clusters, chance overlap alone sits near the
    default floor, so a churn fixture has to be wide enough to separate signal from it.
    """
    draws = [
        _draw(c0=[1, 2, 3], c1=[4, 5, 6], c2=[7, 8, 9], c3=[10, 11, 12]),
        _draw(c0=[1, 4, 7], c1=[2, 5, 10], c2=[3, 8, 11], c3=[6, 9, 12]),
        _draw(c0=[1, 5, 11], c1=[3, 6, 7], c2=[2, 9, 10], c3=[4, 8, 12]),
    ]

    report = analyze_stability(draws).to_jsonable()

    assert report["cluster_count_std"] == 0.0  # count is perfectly stable
    assert report["mean_pair_jaccard"] < 0.35  # identity is not
    assert report["persistent_fraction"] == 0.0
    assert report["mean_cross_draw_ari"] < 0.1


def test_at_least_two_draws_are_required() -> None:
    with pytest.raises(ValueError, match="at least two draws"):
        analyze_stability([_draw(c0=[1, 2])])


def test_per_cluster_persistence_is_reported_for_the_reference_draw() -> None:
    stable = _draw(c0=[1, 2, 3], c1=[4, 5, 6])
    churned = _draw(c0=[1, 2, 3], c1=[7, 8, 9])

    report = analyze_stability([stable, churned], jaccard_floor=0.5)

    assert report.per_cluster[0]["mean_jaccard"] == 1.0
    assert report.per_cluster[0]["match_rate"] == 1.0
    # Cluster 1's members are gone in the second draw: no surviving identity.
    assert report.per_cluster[1]["mean_jaccard"] == 0.0
    assert report.per_cluster[1]["match_rate"] == 0.0
