"""Cluster-identity stability across bootstrap draws.

Consumes ``sample_cluster_labels.parquet`` — written by
``pelinker-dim-selection --persist-labels`` — and reports whether the clusters are the
same objects from draw to draw, not merely the same *number* of objects. A stable cluster
count with churning membership is indistinguishable from a stable clustering if only the
count is reported, which is what the submitted evaluation did.

Outputs, under ``--report-dir``:

- ``stability_summary.json`` — cluster-count spread, mean pairwise Jaccard, mean
  cross-draw ARI, and the fraction of reference clusters that persist above the floor;
- ``stability_pairs.csv`` — one row per draw pair;
- ``stability_per_cluster.csv`` — per reference-draw cluster, mean best-match Jaccard and
  match rate across the other draws.

Rows are identified by ``(pmid, itext, a_abs, b_abs)`` where available, so a document
that mentions a predicate twice contributes two distinct rows. Numbers are measurement
output: write them outside this repository.

Usage:

    uv run python run/analysis/cluster_stability.py \
        --labels-parquet <report>/sample_cluster_labels.parquet \
        --report-dir <workdir>/eval-runs/stability
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd

from pelinker.clustering.stability import analyze_stability
from pelinker.core.paths import ExpandedPath

logger = logging.getLogger(__name__)

_ROW_KEY_COLUMNS = ("pmid", "itext", "a_abs", "b_abs")


def row_keys(frame: pd.DataFrame) -> pd.Series:
    """Stable per-mention identity, falling back to whatever columns exist."""
    present = [c for c in _ROW_KEY_COLUMNS if c in frame.columns]
    if not present:
        raise click.ClickException(
            f"labels parquet needs at least one of {list(_ROW_KEY_COLUMNS)} to identify "
            "rows across draws"
        )
    keys = frame[present[0]].astype(str)
    for col in present[1:]:
        keys = keys + ":" + frame[col].astype(str)
    return keys


def draws_from_labels(
    frame: pd.DataFrame,
    *,
    pca_components: int | None,
    umap_dim: int | None,
) -> list[dict[str, int]]:
    """Split the persisted labels into one ``row_key -> cluster`` mapping per draw."""
    work = frame
    for col, value in (("pca_components", pca_components), ("umap_dim", umap_dim)):
        if value is not None and col in work.columns:
            work = work.loc[work[col] == value]
    if work.empty:
        raise click.ClickException("no rows left after filtering on the requested cell")

    # A single (pca, umap) cell must be compared at a time: clusters from different
    # dimensionalities are not the same objects and matching them would be meaningless.
    for col in ("pca_components", "umap_dim"):
        if col in work.columns and work[col].nunique() > 1:
            raise click.ClickException(
                f"labels span several {col} values {sorted(work[col].unique())}; pass "
                f"--{col.replace('_', '-')} to pick one cell"
            )

    work = work.assign(_row_key=row_keys(work))
    draws: list[dict[str, int]] = []
    for sample_idx, group in work.groupby("sample_idx", sort=True):
        draws.append(dict(zip(group["_row_key"], group["cluster"].astype(int))))
        logger.info("draw %s: %d rows", sample_idx, len(group))
    return draws


@click.command()
@click.option("--labels-parquet", required=True, type=ExpandedPath(exists=True))
@click.option("--report-dir", required=True, type=ExpandedPath())
@click.option(
    "--jaccard-floor",
    default=0.5,
    show_default=True,
    help="Overlap at or above which two clusters count as the same cluster.",
)
@click.option("--pca-components", default=None, type=int)
@click.option("--umap-dim", default=None, type=int)
@click.option("--reference-index", default=0, show_default=True, type=int)
def main(
    labels_parquet: str,
    report_dir: str,
    jaccard_floor: float,
    pca_components: int | None,
    umap_dim: int | None,
    reference_index: int,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frame = pd.read_parquet(labels_parquet)
    logger.info("Loaded %d label rows from %s", len(frame), labels_parquet)
    draws = draws_from_labels(frame, pca_components=pca_components, umap_dim=umap_dim)
    if len(draws) < 2:
        raise click.ClickException(
            f"stability needs at least two draws, found {len(draws)} — re-run the "
            "selection with --persist-labels and n_sample >= 2"
        )

    report = analyze_stability(
        draws, jaccard_floor=jaccard_floor, reference_index=reference_index
    )
    summary = report.to_jsonable()
    logger.info("Summary: %s", json.dumps(summary, indent=1))

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stability_summary.json").write_text(json.dumps(summary, indent=1))
    pd.DataFrame(list(report.pairs)).to_csv(out / "stability_pairs.csv", index=False)
    pd.DataFrame(
        [
            {"cluster_id": cid, **stats}
            for cid, stats in sorted(report.per_cluster.items())
        ]
    ).to_csv(out / "stability_per_cluster.csv", index=False)
    logger.info("Wrote stability artifacts to %s", out)


if __name__ == "__main__":
    main()
