"""Sample-size sweep: measure how the chosen ``min_cluster_size`` moves with N.

One "rung" is a full inner search (``n_sample`` stratified draws → pooled
``min_cluster_size``) at a single ``clustering_sample_rows`` value. Rungs reuse the
production path exactly — :func:`~pelinker.sampling.draw_selection_sample`,
:func:`~pelinker.selection.evaluate_selection_sample`,
:func:`~pelinker.analysis.pooled_min_cluster_size_from_metrics_dfs` — so the measured
exponent describes the real selector, not a reimplementation of it.

The mention frame is loaded and filtered **once**; only the subsample size varies
between rungs, so N is the sole moving part.
"""

from __future__ import annotations

import gc
import json
import pathlib
from dataclasses import replace
from typing import Literal, cast

import pandas as pd
from rich.console import Console
from rich.table import Table

from pelinker.analysis import pooled_min_cluster_size_from_metrics_dfs
from pelinker.config import ClusteringOptimizationConfig, TransformConfig
from pelinker.grid_export import grid_export_rows_from_report
from pelinker.model_selection.artifacts import (
    merge_new_frames_into_per_sample_grid_csv,
)
from pelinker.model_selection.fusion import clustering_optimization_config_for_run
from pelinker.onto import NEGATIVE_LABEL
from pelinker.ops import parse_model_filename
from pelinker.reporting import (
    CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME,
    ModelSelectionReport,
)
from pelinker.sampling import draw_selection_sample
from pelinker.scaling import (
    MIN_RUNGS_FOR_FIT,
    ScaleCurve,
    ScaleRung,
    fit_scale_curve,
)
from pelinker.selection import evaluate_selection_sample, load_selection_frame

SCALE_CURVE_JSON_BASENAME = "scale_curve.json"
SCALE_CURVE_FIGURE_BASENAME = "scale_curve"
SCALE_CURVE_SCHEMA = "pelinker.scale_curve.v1"

DEFAULT_RUNGS: tuple[int, ...] = (10_000, 25_000, 50_000, 100_000)


def load_scale_curve(path: pathlib.Path | str) -> ScaleCurve:
    """Read a ``scale_curve.json`` written by :func:`run_scale_curve`.

    Raises:
        ValueError: on an unrecognized schema, so a stale or hand-edited file fails
            loudly rather than silently driving a hyperparameter.
    """
    p = pathlib.Path(path).expanduser()
    payload = json.loads(p.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema != SCALE_CURVE_SCHEMA:
        raise ValueError(f"{p}: expected schema {SCALE_CURVE_SCHEMA!r}, got {schema!r}")
    return ScaleCurve.from_jsonable(payload["curve"])


def parse_rungs(rungs: tuple[int, ...] | str) -> tuple[int, ...]:
    """Parse ``"10000,25000,50000"`` into a sorted tuple of distinct sample sizes."""
    if isinstance(rungs, str):
        values = [int(part.strip()) for part in rungs.split(",") if part.strip()]
    else:
        values = [int(v) for v in rungs]
    if any(v < 1 for v in values):
        raise ValueError("rung sample sizes must be >= 1")
    unique = tuple(sorted(set(values)))
    if len(unique) < MIN_RUNGS_FOR_FIT:
        raise ValueError(
            f"need at least {MIN_RUNGS_FOR_FIT} distinct rungs to fit a scale curve, "
            f"got {len(unique)}: {unique}"
        )
    return unique


def _evaluate_rung(
    base_frame: pd.DataFrame,
    *,
    sample_rows: int,
    base_config: ClusteringOptimizationConfig,
    transform_config: TransformConfig,
    n_sample: int,
    selected_labels: set[str] | None,
    console: Console,
) -> tuple[ScaleRung | None, list[tuple[int, ModelSelectionReport]]]:
    """Run the inner search once at ``sample_rows`` and summarize it as a rung."""
    config = replace(base_config, clustering_sample_rows=sample_rows)
    all_metrics_dfs: list[pd.DataFrame] = []
    reports: list[tuple[int, ModelSelectionReport]] = []

    for sample_idx in range(n_sample):
        try:
            sample_frame = draw_selection_sample(
                base_frame, config, sample_index=sample_idx
            )
            report = evaluate_selection_sample(
                sample_frame,
                transform_config,
                optimization_config=config,
                selected_labels=selected_labels,
                all_metrics_dfs=all_metrics_dfs,
            )
        except Exception as exc:
            console.print(
                f"[yellow]Skipping failed sample[/yellow] rung={sample_rows} "
                f"sample {sample_idx + 1}/{n_sample}: {exc}"
            )
            report = None
        if report is not None:
            reports.append((sample_idx, report))
        gc.collect()

    if not reports:
        console.print(
            f"[red]Rung {sample_rows}: all {n_sample} bootstrap samples failed[/red]"
        )
        return None, []

    pooled_mcs, score_at = pooled_min_cluster_size_from_metrics_dfs(
        all_metrics_dfs, config
    )

    realized = [
        r.n_rows_realized for _idx, r in reports if r.n_rows_realized is not None
    ]
    if not realized:
        console.print(
            f"[red]Rung {sample_rows}: no report carried n_rows_realized[/red]"
        )
        return None, reports
    # Draws at one rung differ only by seed, so their realized sizes agree except for
    # stratification rounding; the mean is the honest single number for the rung.
    n_rows_realized = int(round(sum(realized) / len(realized)))

    n_clusters_mean = float(
        sum(r.n_clusters_emergent for _idx, r in reports) / len(reports)
    )

    rung = ScaleRung(
        n_rows_realized=n_rows_realized,
        chosen_min_cluster_size=int(pooled_mcs),
        score_at_chosen=float(score_at),
        n_clusters_mean=n_clusters_mean,
        n_sample=len(reports),
        grid_min_scale=config.resolved_min_scale(),
        grid_max_scale=config.max_scale,
    )
    return rung, reports


def _render_rung_table(console: Console, curve: ScaleCurve) -> None:
    table = Table(title="Scale curve rungs", show_lines=False)
    table.add_column("N (realized)", justify="right")
    table.add_column("chosen MCS", justify="right")
    table.add_column("score", justify="right")
    table.add_column("n_clusters", justify="right")
    table.add_column("draws", justify="right")
    table.add_column("pinned", justify="center")
    for rung in curve.rungs:
        table.add_row(
            f"{rung.n_rows_realized:,}",
            str(rung.chosen_min_cluster_size),
            f"{rung.score_at_chosen:.4f}",
            f"{rung.n_clusters_mean:.1f}",
            str(rung.n_sample),
            "[red]yes[/red]" if rung.is_pinned else "no",
        )
    console.print(table)


def run_scale_curve(
    input_parquet: pathlib.Path,
    report_path: pathlib.Path,
    *,
    rungs: tuple[int, ...] | str = DEFAULT_RUNGS,
    pca_components: int = 100,
    umap_dim: int = 8,
    umap_n_neighbors: int | None = None,
    cluster_viz_method: str = "pca",
    min_class_size: int = 20,
    seed: int = 13,
    pca_seed: int = 13,
    umap_seed: int | None = None,
    batch_size: int = 1000,
    n_sample: int = 3,
    prefix: str = "res",
    model: str | None = None,
    layer: str | None = None,
    max_scale: int = 60,
    min_scale: int | None = None,
    clustering_grid_step: int = 5,
    negative_label: str = NEGATIVE_LABEL,
    screener_kind: str = "lda",
    drop_rare_entities: bool = False,
    min_mentions_per_entity: int = 20,
    max_mentions_per_entity: int | None = None,
    max_mentions_negative: int | None = None,
    mention_cap_seed: int = 13,
) -> ScaleCurve | None:
    """Sweep ``clustering_sample_rows`` and fit ``log(MCS*) ~ a + b·log(N)``.

    Writes ``scale_curve.json`` (consumable by ``pelinker-fit scale_curve_path=…``),
    a log-log figure, and per-sample grid rows under ``report_path``.

    Returns the fitted curve, or ``None`` when too few rungs survived to fit one.
    """
    console = Console(force_terminal=True, width=120, legacy_windows=False)
    input_parquet = input_parquet.expanduser()
    if not input_parquet.exists():
        console.print(f"[red]Input parquet not found: {input_parquet}[/red]")
        return None

    try:
        rung_sizes = parse_rungs(rungs)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return None

    resolved_model, resolved_layer = model or "", layer or ""
    if not resolved_model or not resolved_layer:
        parsed = parse_model_filename(input_parquet.name, prefix)
        if parsed is None:
            console.print(
                f"[red]Cannot parse model/layer from {input_parquet.name!r}; "
                f"pass --model and --layer[/red]"
            )
            return None
        parsed_model, parsed_layer = parsed
        # parse_model_filename yields an int for a numeric layer; layers are strings
        # everywhere else (they may be comma-specs like "1,2").
        resolved_model = resolved_model or str(parsed_model)
        resolved_layer = resolved_layer or str(parsed_layer)

    report_path = report_path.expanduser()
    report_path.mkdir(parents=True, exist_ok=True)
    detail_path = report_path / CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME

    base_config = clustering_optimization_config_for_run(
        min_class_size=min_class_size,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
        seed=seed,
        clustering_sample_rows=None,
        batch_size=batch_size,
        negative_label=negative_label,
        screener_kind=screener_kind,
        drop_rare_entities=drop_rare_entities,
        min_mentions_per_entity=min_mentions_per_entity,
        max_mentions_per_entity=max_mentions_per_entity,
        max_mentions_negative=max_mentions_negative,
        mention_cap_seed=mention_cap_seed,
    )
    viz_method = cluster_viz_method.lower()
    if viz_method not in ("pca", "umap"):
        console.print(
            f"[red]cluster_viz_method must be 'pca' or 'umap', got {cluster_viz_method!r}[/red]"
        )
        return None
    transform_config = TransformConfig(
        pca_components=pca_components,
        umap_components=umap_dim,
        umap_n_neighbors=umap_n_neighbors,
        cluster_viz_components=min(3, umap_dim),
        cluster_viz_method=cast(Literal["pca", "umap"], viz_method),
        pca_seed=pca_seed,
        umap_seed=umap_seed,
    )

    console.print(f"[cyan]Loading mention frame from {input_parquet}[/cyan]")
    base_frame = load_selection_frame(
        file_path=input_parquet,
        config=base_config,
        show_embedding_read_progress=True,
    )
    if base_frame is None or len(base_frame) == 0:
        console.print("[red]No mention rows after load filters[/red]")
        return None

    n_available = len(base_frame)
    console.print(f"[green]Loaded {n_available:,} mention rows[/green]")

    # A rung above the frame size would silently collapse onto the full frame and
    # duplicate an existing rung, which fit_scale_curve then discards as leverage-free.
    usable = tuple(r for r in rung_sizes if r < n_available)
    if len(usable) < len(rung_sizes):
        dropped = tuple(r for r in rung_sizes if r >= n_available)
        console.print(
            f"[yellow]Dropping {len(dropped)} rung(s) at or above the frame size "
            f"({n_available:,}): {dropped}[/yellow]"
        )
    if len(usable) < MIN_RUNGS_FOR_FIT:
        console.print(
            f"[red]Only {len(usable)} usable rung(s) below {n_available:,} rows; "
            f"need {MIN_RUNGS_FOR_FIT}. Use a larger parquet or smaller rungs.[/red]"
        )
        return None

    collected: list[ScaleRung] = []
    for rung_size in usable:
        console.print(f"[cyan]Rung {rung_size:,} rows — {n_sample} draw(s)[/cyan]")
        rung, reports = _evaluate_rung(
            base_frame,
            sample_rows=rung_size,
            base_config=base_config,
            transform_config=transform_config,
            n_sample=n_sample,
            selected_labels=None,
            console=console,
        )
        if rung is None:
            continue
        collected.append(rung)
        merge_new_frames_into_per_sample_grid_csv(
            detail_path,
            [
                grid_export_rows_from_report(
                    report,
                    model=resolved_model,
                    # Keep rungs distinguishable inside the shared grid CSV schema.
                    layer=f"{resolved_layer}:rows{rung.n_rows_realized}",
                    sample_idx=sample_idx,
                    chosen_min_cluster_size=rung.chosen_min_cluster_size,
                )
                for sample_idx, report in reports
            ],
        )
        console.print(
            f"[green]Rung {rung.n_rows_realized:,}: "
            f"chosen min_cluster_size = {rung.chosen_min_cluster_size}[/green]"
        )

    if len(collected) < MIN_RUNGS_FOR_FIT:
        console.print(
            f"[red]Only {len(collected)} rung(s) completed; need "
            f"{MIN_RUNGS_FOR_FIT} to fit a curve.[/red]"
        )
        return None

    curve = fit_scale_curve(collected)
    _render_rung_table(console, curve)

    payload = {
        "schema": SCALE_CURVE_SCHEMA,
        "model": resolved_model,
        "layer": resolved_layer,
        "input_parquet": str(input_parquet.resolve()),
        "n_rows_available": int(n_available),
        "pca_components": int(pca_components),
        "umap_dim": int(umap_dim),
        "umap_n_neighbors": (
            None if umap_n_neighbors is None else int(umap_n_neighbors)
        ),
        "n_sample": int(n_sample),
        "curve": curve.to_jsonable(),
    }
    json_path = report_path / SCALE_CURVE_JSON_BASENAME
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]✓[/green] Scale curve written to [cyan]{json_path}[/cyan]")

    console.print(
        f"[bold]log(MCS) = {curve.log_intercept:.4f} + "
        f"{curve.log_slope:.4f}·log(N)[/bold]  (R² = {curve.r_squared:.4f})"
    )
    _warn_on_weak_fit(console, curve)

    # Imported here so the pure-math and orchestration paths stay importable without
    # matplotlib; the CLI is the only caller that needs a figure.
    from pelinker.plotting import plot_scale_curve

    plot_scale_curve(curve, report_path / SCALE_CURVE_FIGURE_BASENAME)
    return curve


def _warn_on_weak_fit(console: Console, curve: ScaleCurve) -> None:
    """Say plainly when the measured curve should not be extrapolated from."""
    pinned = curve.pinned_rungs
    if pinned:
        sizes = ", ".join(f"{r.n_rows_realized:,}" for r in pinned)
        console.print(
            f"[yellow]Warning: {len(pinned)} rung(s) chose a min_cluster_size on a grid "
            f"bound (N = {sizes}). The grid, not the sample size, decided those points — "
            f"widen --min-scale / --max-scale and re-run before trusting the slope."
            "[/yellow]"
        )
    if curve.r_squared < 0.8:
        console.print(
            f"[yellow]Warning: R² = {curve.r_squared:.3f} is low. The chosen "
            f"min_cluster_size is not a clean power law in N here; treat the "
            f"extrapolation as a weak prior, not a recommendation.[/yellow]"
        )
    if abs(curve.log_slope) < 0.05:
        console.print(
            "[cyan]Note: the fitted exponent is ~0, i.e. the chosen min_cluster_size is "
            "effectively scale-invariant over the measured range — transferring the "
            "absolute value between sample sizes is defensible here.[/cyan]"
        )
