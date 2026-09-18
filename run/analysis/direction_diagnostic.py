"""How badly do converse predicate pairs collide in the learned space?

Sizes the directionality problem *before* any model is built. For every KB inverse pair
(``owl:inverseOf``, carried by ``run/preprocessing/extract_properties_ro.py``) whose two
members both appear in the mention frame, the script reports:

- **collision rate** — pairs whose two members' mention mass concentrates in the same
  cluster, i.e. the clustering cannot tell a predicate from its converse;
- **co-membership** — per pair, the fraction of shared mention mass in the top common
  cluster and the Jensen–Shannon-free overlap of their cluster distributions;
- **separation** — pairs whose members occupy disjoint clusters (the direction signal is
  already there and only the *labeling* collapses it).

Input is the mention frame with cluster assignments, from any of: a fit's clustering
report (``--fit-report``, the ``assignments`` block — no refit needed), the training
frame on a fitted linker (``--model-path``), or a parquet with ``entity`` + ``cluster``
columns (``--assignments-parquet``). Distinguishing collision from separation matters —
if collisions are rare, the paper claim only needs scoping; if they dominate, a direction
stage is load-bearing.

Note the cluster-composition artifact is deliberately *not* accepted as input: it stores
only the top-N entities per cluster for plotting, so per-entity distributions read off it
are truncated and would understate overlap.

Numbers are measurement output: write them outside this repository and cite them from
the measurement writeup.

Usage:

    uv run python run/analysis/direction_diagnostic.py \
        --model-path <models>/pelinker.pubmedbert.1 \
        --kb-csv-path data/derived/properties.synthesis.3.csv \
        --report-dir <workdir>/eval-runs/direction-diagnostic
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd

from pelinker.clustering.composition import filter_emergent_assignments
from pelinker.core.paths import ExpandedPath

logger = logging.getLogger(__name__)


def load_inverse_pairs(kb: pd.DataFrame) -> list[tuple[str, str]]:
    """Unordered ``owl:inverseOf`` pairs where both members carry a KB label."""
    if "inverse_entity_id" not in kb.columns:
        raise click.ClickException(
            "KB CSV has no 'inverse_entity_id' column — regenerate it with "
            "run/preprocessing/extract_properties_ro.py and merge_properties.py"
        )
    labelled = kb[kb["label"].notna()]
    known = set(labelled["entity_id"].astype(str))
    seen: set[tuple[str, str]] = set()
    for _, row in labelled.iterrows():
        inverse = row["inverse_entity_id"]
        if pd.isna(inverse):
            continue
        a, b = str(row["entity_id"]), str(inverse)
        if b not in known or a == b:
            continue
        seen.add((a, b) if a < b else (b, a))
    return sorted(seen)


def cluster_distribution(assignments: pd.DataFrame, label: str) -> dict[int, float]:
    """Normalized mention mass of one KB label over emergent clusters."""
    rows = assignments.loc[assignments["entity"].astype(str) == label]
    if rows.empty:
        return {}
    counts = rows["cluster"].astype(int).value_counts()
    total = float(counts.sum())
    return {int(c): float(n) / total for c, n in counts.items()}


def overlap(dist_a: dict[int, float], dist_b: dict[int, float]) -> float:
    """Bhattacharyya-free overlap: total shared probability mass over clusters."""
    return sum(
        min(dist_a.get(c, 0.0), dist_b.get(c, 0.0)) for c in set(dist_a) | set(dist_b)
    )


def diagnose(
    assignments: pd.DataFrame,
    pairs: list[tuple[str, str]],
    labels: dict[str, str],
    *,
    collision_threshold: float,
) -> tuple[list[dict], dict]:
    """Per-pair collision measures plus the aggregate summary."""
    rows: list[dict] = []
    for id_a, id_b in pairs:
        label_a, label_b = labels.get(id_a), labels.get(id_b)
        if label_a is None or label_b is None:
            continue
        dist_a = cluster_distribution(assignments, label_a)
        dist_b = cluster_distribution(assignments, label_b)
        if not dist_a or not dist_b:
            continue  # at least one member never appears in the corpus
        top_a = max(dist_a, key=lambda c: dist_a[c])
        top_b = max(dist_b, key=lambda c: dist_b[c])
        shared = overlap(dist_a, dist_b)
        rows.append(
            {
                "entity_id_a": id_a,
                "entity_id_b": id_b,
                "label_a": label_a,
                "label_b": label_b,
                "n_mentions_a": int(
                    (assignments["entity"].astype(str) == label_a).sum()
                ),
                "n_mentions_b": int(
                    (assignments["entity"].astype(str) == label_b).sum()
                ),
                "top_cluster_a": top_a,
                "top_cluster_b": top_b,
                "same_top_cluster": bool(top_a == top_b),
                "mass_overlap": shared,
                "disjoint": shared == 0.0,
                "collides": bool(top_a == top_b or shared >= collision_threshold),
            }
        )

    n = len(rows)
    summary = {
        "n_pairs_in_kb": len(pairs),
        "n_pairs_observed": n,
        "collision_threshold": collision_threshold,
        "n_same_top_cluster": sum(r["same_top_cluster"] for r in rows),
        "n_colliding": sum(r["collides"] for r in rows),
        "n_disjoint": sum(r["disjoint"] for r in rows),
        "collision_rate": (sum(r["collides"] for r in rows) / n) if n else None,
        "disjoint_rate": (sum(r["disjoint"] for r in rows) / n) if n else None,
        "mean_mass_overlap": (sum(r["mass_overlap"] for r in rows) / n) if n else None,
    }
    return rows, summary


def _assignments_from_linker(model_path: str) -> pd.DataFrame:
    from pelinker.model import Linker

    linker = Linker.load(model_path)
    frame = linker.training_cluster_frame
    if frame is None:
        raise click.ClickException(
            "artifact carries no training_cluster_frame; pass --assignments-parquet"
        )
    return frame[["entity", "cluster"]].copy()


def _assignments_from_fit_report(path: str) -> pd.DataFrame:
    from pelinker.reports.io import read_clustering_report_json

    report = read_clustering_report_json(path)
    assignments = report.assignments
    if assignments is None or assignments.empty:
        raise click.ClickException(f"{path}: no assignments in the fit report")
    return assignments[["entity", "cluster"]].copy()


@click.command()
@click.option("--kb-csv-path", required=True, type=ExpandedPath(exists=True))
@click.option("--report-dir", required=True, type=ExpandedPath())
@click.option(
    "--model-path", default=None, type=ExpandedPath(), help="Fitted linker artifact."
)
@click.option(
    "--fit-report",
    default=None,
    type=ExpandedPath(exists=True),
    help="linker_fit.clustering_report.json.gz from a completed fit.",
)
@click.option(
    "--assignments-parquet",
    default=None,
    type=ExpandedPath(exists=True),
    help="Mention frame with 'entity' and 'cluster' columns.",
)
@click.option(
    "--collision-threshold",
    default=0.5,
    show_default=True,
    help="Shared cluster mass at or above which a pair counts as colliding.",
)
def main(
    kb_csv_path: str,
    report_dir: str,
    model_path: str | None,
    fit_report: str | None,
    assignments_parquet: str | None,
    collision_threshold: float,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sources = [model_path, fit_report, assignments_parquet]
    if sum(s is not None for s in sources) != 1:
        raise click.ClickException(
            "pass exactly one of --model-path / --fit-report / --assignments-parquet"
        )

    kb = pd.read_csv(kb_csv_path)
    labels = {
        str(r["entity_id"]): str(r["label"])
        for _, r in kb.iterrows()
        if not pd.isna(r["label"])
    }
    pairs = load_inverse_pairs(kb)
    logger.info("KB inverse pairs with both members labelled: %d", len(pairs))

    if model_path is not None:
        assignments = _assignments_from_linker(model_path)
    elif fit_report is not None:
        assignments = _assignments_from_fit_report(fit_report)
    else:
        assignments = pd.read_parquet(
            assignments_parquet, columns=["entity", "cluster"]
        )
    assignments = filter_emergent_assignments(assignments)
    logger.info("Emergent mention rows: %d", len(assignments))

    rows, summary = diagnose(
        assignments, pairs, labels, collision_threshold=collision_threshold
    )
    logger.info("Summary: %s", json.dumps(summary, indent=1))

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("mass_overlap", ascending=False).to_csv(
        out / "direction_pairs.csv", index=False
    )
    (out / "direction_summary.json").write_text(json.dumps(summary, indent=1))
    logger.info("Wrote %d pair rows to %s", len(rows), out)


if __name__ == "__main__":
    main()
