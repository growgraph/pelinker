"""Dim-selection run orchestration (PCA × UMAP grid search)."""

from __future__ import annotations

import gc
import pathlib

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from pelinker.search.grid_solver import pooled_grid_solve_from_metrics_dfs
from pelinker.search.dim_selection.checkpoint import (
    DEFAULT_CHECKPOINT_NAME,
    compute_run_fingerprint,
    fingerprint_config_from_cli,
    load_checkpoint,
    mark_cell_done,
    new_checkpoint,
    record_failure,
    save_checkpoint_atomic,
)
from pelinker.search.dim_selection.grids import (
    DEFAULT_PCA_GRID,
    DEFAULT_UMAP_GRID,
    cell_key,
    cluster_viz_components_for_umap,
    coarse_cells,
    parse_cell_key,
    parse_int_grid,
    pick_winner_row,
    refine_cells,
)
from pelinker.core.onto import NEGATIVE_LABEL
from pelinker.search.dim_selection.summary import (
    render_dim_selection_summary,
    results_dataframe_from_summaries,
)
from pelinker.search.grid_export import grid_export_rows_from_report
from pelinker.search.model_selection.artifacts import (
    merge_new_frames_into_fine_metadata_jsonl,
    merge_new_frames_into_per_sample_grid_csv,
    merge_new_frames_into_screener_eval_jsonl,
    per_datapoint_scores_df,
)
from pelinker.search.model_selection.fine_metadata import clustering_metadata_df
from pelinker.search.model_selection.fusion import (
    clustering_optimization_config_for_run,
)
from pelinker.data.tables import parse_model_filename
from pelinker.plotting import plot_metrics, plot_metrics_with_error_bars
from pelinker.reports.paths import (
    CLUSTERING_SEARCH_FINE_METADATA_BASENAME,
    CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME,
    FINE_SCREENER_EVAL_BASENAME,
)
from pelinker.reports.schema import ModelSelectionReport
from pelinker.reports.summary import summarize_clustering_reports_for_search
from pelinker.search.sampling import draw_selection_sample
from pelinker.search.selected_hyperparameters import (
    SelectedHyperparameters,
    write_selected_hyperparameters,
)
from pelinker.search.selection import evaluate_selection_sample, load_selection_frame
from pelinker.clustering.transform import TransformConfig


def _write_handoff(
    payload: dict,
    report_path: pathlib.Path,
    console: Console,
    *,
    model: str,
    layer: str,
    manifold_kind: str,
    umap_n_neighbors: int | None,
    run_fingerprint: str,
) -> None:
    """Emit ``selected_hyperparameters.json`` so ``pelinker-fit`` can read the winner."""
    chosen = payload.get("chosen")
    if not chosen:
        console.print(
            "[yellow]No winning cell; skipping selected_hyperparameters.json[/yellow]"
        )
        return
    best_size = int(round(float(chosen.get("best_size") or 0.0)))
    if best_size < 2:
        console.print(
            f"[yellow]Chosen min_cluster_size {best_size} < 2; "
            f"skipping selected_hyperparameters.json[/yellow]"
        )
        return
    selected = SelectedHyperparameters(
        source="dim_selection",
        model=model,
        layer=layer,
        pca_components=int(chosen["pca_components"]),
        umap_dim=int(chosen["umap_dim"]),
        min_cluster_size=best_size,
        manifold_kind=manifold_kind,
        n_rows_realized=chosen.get("n_rows_realized"),
        umap_n_neighbors=umap_n_neighbors,
        run_fingerprint=run_fingerprint,
        outer_score=chosen.get("outer_score"),
    )
    out = write_selected_hyperparameters(selected, report_path)
    console.print(
        f"[green]✓[/green] Selected hyperparameters written to [cyan]{out}[/cyan]"
    )


def _export_layer(layer: str, pca_components: int, umap_dim: int) -> str:
    """Unique layer label for shared grid/fine-metadata artifacts."""
    return f"{layer}:pca{int(pca_components)}:umap{int(umap_dim)}"


def _summary_flat_for_cell(
    summary_row,
    *,
    pca_components: int,
    umap_dim: int,
) -> dict[str, str | float | int | None]:
    flat = dict(summary_row.to_flat_dict())
    flat["pca_components"] = int(pca_components)
    flat["umap_dim"] = int(umap_dim)
    return flat


def _load_selected_labels(
    path: pathlib.Path | None, console: Console
) -> set[str] | None:
    if path is None:
        return None
    path = path.expanduser()
    if not path.exists():
        console.print(f"[red]Selected labels KB file not found: {path}[/red]")
        raise FileNotFoundError(str(path))
    console.print(f"[cyan]Loading selected labels KB from {path}[/cyan]")
    df_selected = pd.read_csv(path)
    if "label" not in df_selected.columns:
        raise ValueError(
            f"Selected labels KB file must have a 'label' column. "
            f"Found columns: {list(df_selected.columns)}"
        )
    labels = set(df_selected["label"].dropna().astype(str))
    console.print(f"[green]Loaded {len(labels)} labels from selected labels KB[/green]")
    return labels


def _resolve_model_layer(
    input_parquet: pathlib.Path,
    *,
    prefix: str,
    model: str | None,
    layer: str | None,
) -> tuple[str, str]:
    parsed_model, parsed_layer = parse_model_filename(input_parquet.name, prefix)
    resolved_model = model if model is not None else parsed_model
    resolved_layer = (
        layer
        if layer is not None
        else (str(parsed_layer) if parsed_layer is not None else None)
    )
    if resolved_model is None or resolved_layer is None:
        raise ValueError(
            f"Could not resolve model/layer from {input_parquet.name!r}; "
            "pass --model and --layer explicitly."
        )
    return str(resolved_model), str(resolved_layer)


def run_dim_selection(
    input_parquet: pathlib.Path,
    report_path: pathlib.Path,
    *,
    pca_grid: tuple[int, ...] | str = DEFAULT_PCA_GRID,
    umap_grid: tuple[int, ...] | str = DEFAULT_UMAP_GRID,
    refine: bool = True,
    cluster_viz_method: str = "pca",
    manifold_kind: str = "umap",
    umap_n_neighbors: int | None = None,
    min_class_size: int = 20,
    seed: int = 13,
    pca_seed: int = 13,
    umap_seed: int | None = None,
    clustering_sample_rows: int | None = None,
    batch_size: int = 1000,
    n_sample: int = 3,
    prefix: str = "res",
    model: str | None = None,
    layer: str | None = None,
    selected_labels_kb_path: pathlib.Path | None = None,
    max_scale: int = 60,
    min_scale: int | None = None,
    clustering_grid_step: int = 5,
    resume: bool = True,
    checkpoint_path: pathlib.Path | None = None,
    negative_label: str = NEGATIVE_LABEL,
    screener_kind: str = "lda",
    drop_rare_entities: bool = False,
    min_mentions_per_entity: int = 20,
    max_mentions_per_entity: int | None = None,
    max_mentions_negative: int | None = None,
    mention_cap_seed: int = 13,
) -> None:
    """
    Search ``(pca_components, umap_dim)`` for one embedding parquet.

    **Inner** ``min_cluster_size`` (MCS) uses ``grid_objective=dbcv_ari_geomean``.
    **Outer** cell ranking uses the same DBCV+ARI pooling across candidate cells
    (``outer_score``); ``best_score`` remains mean DBCV for heatmaps.
    """
    console = Console(force_terminal=True, width=120, legacy_windows=False)
    input_parquet = input_parquet.expanduser()
    if not input_parquet.exists():
        console.print(f"[red]Input parquet not found: {input_parquet}[/red]")
        return

    try:
        selected_labels = _load_selected_labels(selected_labels_kb_path, console)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return

    try:
        resolved_model, resolved_layer = _resolve_model_layer(
            input_parquet, prefix=prefix, model=model, layer=layer
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    pca_vals = parse_int_grid(pca_grid, name="pca")
    umap_vals = parse_int_grid(umap_grid, name="umap")

    report_path = report_path.expanduser()
    report_path.mkdir(parents=True, exist_ok=True)
    detail_path = report_path / CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME
    fine_metadata_path = report_path / CLUSTERING_SEARCH_FINE_METADATA_BASENAME
    fine_screener_eval_path = report_path / FINE_SCREENER_EVAL_BASENAME
    if not resume:
        for artifact in (detail_path, fine_metadata_path, fine_screener_eval_path):
            try:
                if artifact.exists():
                    artifact.unlink()
            except OSError:
                pass

    fp_payload = fingerprint_config_from_cli(
        input_parquet=input_parquet,
        model=resolved_model,
        layer=resolved_layer,
        pca_grid=pca_vals,
        umap_grid=umap_vals,
        refine=refine,
        cluster_viz_method=cluster_viz_method.lower(),
        min_class_size=min_class_size,
        seed=seed,
        pca_seed=pca_seed,
        umap_seed=umap_seed,
        clustering_sample_rows=clustering_sample_rows,
        batch_size=batch_size,
        n_sample=n_sample,
        selected_labels_kb_path=selected_labels_kb_path,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
        negative_label=negative_label,
        screener_kind=screener_kind,
        drop_rare_entities=drop_rare_entities,
        min_mentions_per_entity=min_mentions_per_entity,
        max_mentions_per_entity=max_mentions_per_entity,
        max_mentions_negative=max_mentions_negative,
        mention_cap_seed=mention_cap_seed,
    )
    run_fingerprint = compute_run_fingerprint(fp_payload)

    ckpt_path = (
        checkpoint_path.expanduser()
        if checkpoint_path is not None
        else report_path / DEFAULT_CHECKPOINT_NAME
    )

    resumed_from_checkpoint = bool(resume and ckpt_path.exists())
    if resumed_from_checkpoint:
        ckpt = load_checkpoint(ckpt_path)
        if ckpt.run_fingerprint != run_fingerprint:
            console.print(
                "[red]Checkpoint run fingerprint does not match current CLI parameters.[/red]\n"
                f"Checkpoint: {ckpt.run_fingerprint}\n"
                f"Current:    {run_fingerprint}\n"
                "Use the same inputs, or pass --no-resume to reinitialize the checkpoint."
            )
            return
        console.print(
            f"[green]Resuming from checkpoint[/green] [cyan]{ckpt_path}[/cyan]"
        )
    else:
        ckpt = new_checkpoint(run_fingerprint)
        if resume:
            console.print(
                f"[cyan]No checkpoint at[/cyan] [yellow]{ckpt_path}[/yellow][cyan]; "
                f"starting new run (writing checkpoint to[/cyan] "
                f"[green]{ckpt_path}[/green][cyan]).[/cyan]"
            )
        else:
            console.print(
                f"[cyan]New run (--no-resume); checkpoint reinitialized at[/cyan] "
                f"[green]{ckpt_path}[/green]"
            )
    save_checkpoint_atomic(ckpt_path, ckpt)

    completed = set(ckpt.completed_cells)
    optimization_config = clustering_optimization_config_for_run(
        min_class_size=min_class_size,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
        seed=seed,
        clustering_sample_rows=clustering_sample_rows,
        batch_size=batch_size,
        negative_label=negative_label,
        screener_kind=screener_kind,
        drop_rare_entities=drop_rare_entities,
        min_mentions_per_entity=min_mentions_per_entity,
        max_mentions_per_entity=max_mentions_per_entity,
        max_mentions_negative=max_mentions_negative,
        mention_cap_seed=mention_cap_seed,
    )

    console.print(
        f"[cyan]Loading[/cyan] [green]{input_parquet}[/green] "
        f"([cyan]{resolved_model}[/cyan]/[yellow]{resolved_layer}[/yellow])"
    )
    try:
        base_frame = load_selection_frame(
            file_path=input_parquet,
            config=optimization_config,
            selected_labels=selected_labels,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load selection frame: {exc}[/red]")
        return

    def evaluate_cell(
        pca_k: int, umap_d: int
    ) -> dict[str, str | float | int | None] | None:
        key = cell_key(pca_k, umap_d)
        if key in completed:
            return (
                dict(ckpt.summaries_by_key[key])
                if key in ckpt.summaries_by_key
                else None
            )

        transform_config = TransformConfig(
            pca_components=pca_k,
            umap_components=umap_d,
            umap_n_neighbors=umap_n_neighbors,
            cluster_viz_components=cluster_viz_components_for_umap(umap_d),
            cluster_viz_method=cluster_viz_method.lower(),
            manifold_kind=manifold_kind,  # type: ignore[arg-type]
            pca_seed=pca_seed,
            umap_seed=umap_seed,
        )
        export_layer = _export_layer(resolved_layer, pca_k, umap_d)
        file_metrics: list[pd.DataFrame] = []
        file_reports: list[ModelSelectionReport] = []
        all_metrics_dfs: list[pd.DataFrame] = []
        grid_report_samples: list[tuple[int, ModelSelectionReport]] = []
        fine_frames: list[pd.DataFrame] = []
        screener_frames: list[pd.DataFrame] = []

        for sample_idx in range(n_sample):
            try:
                sample_frame = draw_selection_sample(
                    base_frame,
                    optimization_config,
                    sample_index=sample_idx,
                )
                report = evaluate_selection_sample(
                    sample_frame,
                    transform_config,
                    optimization_config=optimization_config,
                    all_metrics_dfs=all_metrics_dfs,
                )
            except Exception as exc:
                console.print(
                    f"[yellow]Skipping failed sample[/yellow] {key} "
                    f"sample {sample_idx + 1}: {exc}"
                )
                report = None

            if report is not None:
                file_metrics.append(report.metrics_df)
                file_reports.append(report)
                grid_report_samples.append((sample_idx, report))
                fine_frames.append(
                    clustering_metadata_df(
                        report,
                        model=resolved_model,
                        layer=export_layer,
                        sample_idx=sample_idx,
                    )
                )
                if report.screener_oos_datapoints is not None:
                    screener_frames.append(
                        per_datapoint_scores_df(
                            report.screener_oos_datapoints,
                            combo_key=key,
                            model=resolved_model,
                            layer=export_layer,
                            sample_idx=sample_idx,
                        )
                    )
            gc.collect()

        if not file_reports:
            record_failure(
                ckpt,
                ckpt_path,
                cell_key=key,
                message="all bootstrap samples failed",
            )
            return None

        grid_solved = pooled_grid_solve_from_metrics_dfs(
            all_metrics_dfs,
            optimization_config,
        )
        pooled_mcs = grid_solved.chosen_min_cluster_size
        grid_batch = [
            grid_export_rows_from_report(
                report,
                model=resolved_model,
                layer=export_layer,
                sample_idx=sample_idx,
                chosen_min_cluster_size=pooled_mcs,
            ).assign(pca_components=pca_k, umap_dim=umap_d)
            for sample_idx, report in grid_report_samples
        ]
        merge_new_frames_into_per_sample_grid_csv(detail_path, grid_batch)
        merge_new_frames_into_fine_metadata_jsonl(fine_metadata_path, fine_frames)
        if screener_frames:
            merge_new_frames_into_screener_eval_jsonl(
                fine_screener_eval_path, screener_frames
            )

        if len(file_metrics) > 1:
            plot_metrics_with_error_bars(
                file_metrics,
                report_path / f"{resolved_model}_{export_layer}_error_bars.png",
                chosen_min_cluster_size=float(pooled_mcs),
                grid_solve=grid_solved,
            )
        else:
            plot_metrics(
                file_metrics[0],
                report_path / f"{resolved_model}_{export_layer}.png",
            )

        summary_row = summarize_clustering_reports_for_search(
            file_reports,
            model=resolved_model,
            layer=resolved_layer,
            pooled_min_cluster_size=pooled_mcs,
        )
        flat = _summary_flat_for_cell(
            summary_row, pca_components=pca_k, umap_dim=umap_d
        )
        mark_cell_done(ckpt, ckpt_path, cell_key=key, summary_flat=flat)
        completed.add(key)
        return flat

    # --- coarse phase ---
    coarse = coarse_cells(pca_vals, umap_vals)
    ckpt.stages["coarse"] = "in_progress"
    save_checkpoint_atomic(ckpt_path, ckpt)

    pending_coarse = [c for c in coarse if cell_key(*c) not in completed]
    console.print(
        f"[bold]Coarse grid[/bold]: {len(coarse)} cells "
        f"({len(pending_coarse)} pending), n_sample={n_sample}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task(
            "[cyan]Coarse PCA×UMAP…",
            total=len(coarse),
            completed=len(coarse) - len(pending_coarse),
        )
        for pca_k, umap_d in coarse:
            key = cell_key(pca_k, umap_d)
            if key in completed:
                progress.advance(task)
                continue
            progress.update(
                task,
                description=f"[cyan]Coarse[/cyan] pca={pca_k} umap={umap_d}",
            )
            evaluate_cell(pca_k, umap_d)
            progress.advance(task)

    ckpt.stages["coarse"] = "complete"
    save_checkpoint_atomic(ckpt_path, ckpt)

    # --- refine phase ---
    if refine:
        ckpt.stages["refine"] = "in_progress"
        save_checkpoint_atomic(ckpt_path, ckpt)
        coarse_summaries = [
            dict(ckpt.summaries_by_key[cell_key(p, u)])
            for p, u in coarse
            if cell_key(p, u) in ckpt.summaries_by_key
        ]
        if coarse_summaries:
            coarse_df = results_dataframe_from_summaries(coarse_summaries)
            winner = pick_winner_row(coarse_df)
            best_pca = int(winner["pca_components"])
            best_umap = int(winner["umap_dim"])
            already = {parse_cell_key(k) for k in completed if k.startswith("pca")}
            refine_list = refine_cells(best_pca, best_umap, already=already)
            console.print(
                f"[bold]Refine[/bold] around pca={best_pca} umap={best_umap}: "
                f"{len(refine_list)} new cells"
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                refresh_per_second=4,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Refine PCA×UMAP…",
                    total=max(len(refine_list), 1),
                )
                if not refine_list:
                    progress.advance(task)
                for pca_k, umap_d in refine_list:
                    progress.update(
                        task,
                        description=f"[cyan]Refine[/cyan] pca={pca_k} umap={umap_d}",
                    )
                    evaluate_cell(pca_k, umap_d)
                    progress.advance(task)
        ckpt.stages["refine"] = "complete"
    else:
        ckpt.stages["refine"] = "skipped"
    save_checkpoint_atomic(ckpt_path, ckpt)

    summaries = [
        dict(row)
        for _k, row in sorted(ckpt.summaries_by_key.items(), key=lambda item: item[0])
    ]
    df_results = results_dataframe_from_summaries(summaries)
    payload = render_dim_selection_summary(
        df_results,
        report_path,
        model=resolved_model,
        layer=resolved_layer,
        n_sample=n_sample,
        refine=refine,
        grid_csv_path=detail_path,
    )
    _write_handoff(
        payload,
        report_path,
        console,
        model=resolved_model,
        layer=resolved_layer,
        manifold_kind=manifold_kind,
        umap_n_neighbors=umap_n_neighbors,
        run_fingerprint=run_fingerprint,
    )

    table = Table(title="Dim selection results (outer DBCV+ARI at pooled MCS)")
    table.add_column("pca")
    table.add_column("umap")
    table.add_column("outer")
    table.add_column("dbcv")
    table.add_column("ari")
    table.add_column("best_size")
    if not df_results.empty:
        from pelinker.clustering.ranking import (
            OUTER_SCORE_COL,
            attach_outer_scores,
        )

        show = attach_outer_scores(df_results).sort_values(
            by=[OUTER_SCORE_COL, "outer_score_std", "pca_components", "umap_dim"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        for _, row in show.iterrows():
            ari_val = row.get("ari")
            ari_str = (
                "—" if ari_val is None or pd.isna(ari_val) else f"{float(ari_val):.3f}"
            )
            table.add_row(
                str(int(row["pca_components"])),
                str(int(row["umap_dim"])),
                f"{float(row[OUTER_SCORE_COL]):.3f}",
                f"{float(row['best_score']):.3f}",
                ari_str,
                f"{float(row['best_size']):.0f}",
            )
    console.print(table)

    chosen = payload.get("chosen")
    if chosen is not None:
        console.print(
            f"\n[bold green]Chosen[/bold green] "
            f"pca_components={chosen['pca_components']} "
            f"umap_dim={chosen['umap_dim']} "
            f"(outer DBCV+ARI={chosen.get('outer_score', chosen['best_score']):.3f}, "
            f"DBCV={chosen['best_score']:.3f})"
        )
    console.print(
        "[dim]MCS = min_cluster_size. "
        "Inner MCS: DBCV+ARI (dbcv_ari_geomean, paired 1-SE). "
        "Outer ranking: DBCV+ARI (geomean, per-cell). "
        "best_score remains mean DBCV.[/dim]"
    )
