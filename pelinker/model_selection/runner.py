"""Model-selection run orchestration (embedding grid search)."""

from __future__ import annotations

import gc
import math
import pathlib

import numpy as np
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

from pelinker.analysis import pooled_min_cluster_size_from_metrics_dfs
from pelinker.clustering_fusion_ranking import (
    singleton_items_by_dbcv_score,
    top_k_fusion_candidates_by_dbcv_proxy,
)
from pelinker.grid_export import grid_export_rows_from_report
from pelinker.model_selection.artifacts import (
    mark_combination_done,
    merge_new_frames_into_fine_metadata_jsonl,
    merge_new_frames_into_per_sample_grid_csv,
    merge_new_frames_into_screener_eval_jsonl,
    per_datapoint_scores_df,
    read_optional_csv,
    read_optional_jsonl_gzip,
    record_failure,
    results_from_checkpoint,
    singleton_score_by_model_layer_from_checkpoint,
    dedupe_per_sample_grid,
)
from pelinker.model_selection.fine_metadata import clustering_metadata_df
from pelinker.model_selection.fusion import (
    clustering_optimization_config_for_run,
    combo_key_for_results_row,
    materialize_best_report,
    parse_fusion_members,
    path_by_model_layer,
    recompute_leaderboard_from_results,
)
from pelinker.model_selection.summary import render_model_selection_summary_figures
from pelinker.model_selection_checkpoint import (
    DEFAULT_CHECKPOINT_NAME,
    RunMode,
    combination_key_from_members,
    compute_run_fingerprint,
    fingerprint_config_from_cli,
    load_checkpoint,
    new_checkpoint,
    reconcile_fusion_checkpoint_params,
    save_checkpoint_atomic,
    utc_now_iso,
)
from pelinker.ops import parse_model_filename
from pelinker.plotting import (
    plot_cluster_viz,
    plot_metrics,
    plot_metrics_with_error_bars,
)
from pelinker.reporting import (
    CLUSTERING_SEARCH_FINE_METADATA_BASENAME,
    FINE_SCREENER_EVAL_BASENAME,
    CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME,
    MODEL_SELECTION_RUN_REPORT_SCHEMA,
    ClusteringSearchSummaryRow,
    ModelSelectionReport,
    ModelSelectionRunReport,
    clustering_search_summary_row_from_flat_dict,
    model_selection_run_report_path,
    summarize_clustering_reports_for_search,
    write_model_selection_run_report_json,
)
from pelinker.sampling import draw_selection_sample
from pelinker.selection import (
    evaluate_selection_sample,
    load_selection_frame,
)
from pelinker.transform import TransformConfig


def json_ready_flat_row(row: ClusteringSearchSummaryRow) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in row.to_flat_dict().items():
        if isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def run_model_selection(
    input_dir: pathlib.Path,
    report_path: pathlib.Path,
    umap_dim: int,
    pca_components: int,
    cluster_viz_method: str,
    min_class_size: int,
    seed: int,
    pca_seed: int,
    umap_seed: int | None,
    clustering_sample_rows: int | None,
    batch_size: int,
    n_sample: int,
    prefix: str,
    selected_labels_kb_path: pathlib.Path | None,
    max_scale: int,
    min_scale: int | None,
    clustering_grid_step: int,
    fusion_pairs: int,
    fusion_triples: int,
    resume: bool,
    checkpoint_path: pathlib.Path | None,
    mode: RunMode,
    negative_label: str,
    screener_kind: str,
    drop_rare_entities: bool,
    min_mentions_per_entity: int,
    max_mentions_per_entity: int | None,
    max_mentions_negative: int | None,
    mention_cap_seed: int,
) -> None:
    """
    Process multiple parquet files and compute optimal cluster sizes.

    Files should follow the pattern: <prefix>_<model>_<layer>.parquet

    After scoring each (model, layer) alone (mean DBCV as ``best_score``), optionally
    evaluates fused embeddings: pairs/triples with the highest sum of singleton DBCV
    scores (see ``fusion_pairs`` / ``fusion_triples``), then clusters the
    concatenated mention-level vectors via :func:`~pelinker.selection.load_selection_frame`
    and per-bootstrap :func:`~pelinker.selection.evaluate_selection_sample`.

    Checkpointing is on by default (``resume``): progress is saved under ``report_path``.
    Use ``resume=False`` to discard the on-disk checkpoint and start from an empty state.
    """
    console = Console(force_terminal=True, width=120, legacy_windows=False)
    input_dir = input_dir.expanduser()

    selected_labels: set[str] | None = None
    if selected_labels_kb_path is not None:
        selected_labels_kb_path = selected_labels_kb_path.expanduser()
        if not selected_labels_kb_path.exists():
            console.print(
                f"[red]Selected labels KB file not found: {selected_labels_kb_path}[/red]"
            )
            return

        console.print(
            f"[cyan]Loading selected labels KB from {selected_labels_kb_path}[/cyan]"
        )
        try:
            df_selected = pd.read_csv(selected_labels_kb_path)
            if "label" not in df_selected.columns:
                console.print(
                    f"[red]Selected labels KB file must have a 'label' column. Found columns: {list(df_selected.columns)}[/red]"
                )
                return
            selected_labels = set(df_selected["label"].dropna().astype(str))
            console.print(
                f"[green]Loaded {len(selected_labels)} labels from selected labels KB[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error loading selected labels KB: {e}[/red]")
            return

    report_path = report_path.expanduser()
    report_path.mkdir(parents=True, exist_ok=True)
    detail_path = report_path / CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME
    fine_metadata_path = report_path / CLUSTERING_SEARCH_FINE_METADATA_BASENAME
    fine_screener_eval_path = report_path / FINE_SCREENER_EVAL_BASENAME
    run_report_json_path = model_selection_run_report_path(report_path)
    if not resume:
        for artifact in (detail_path, fine_metadata_path, fine_screener_eval_path):
            try:
                if artifact.exists():
                    artifact.unlink()
            except OSError:
                pass

    fp_payload = fingerprint_config_from_cli(
        input_dir=input_dir,
        umap_dim=umap_dim,
        pca_components=pca_components,
        cluster_viz_method=cluster_viz_method.lower(),
        min_class_size=min_class_size,
        seed=seed,
        pca_seed=pca_seed,
        umap_seed=umap_seed,
        clustering_sample_rows=clustering_sample_rows,
        batch_size=batch_size,
        prefix=prefix,
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
                f"starting new run (writing checkpoint to[/cyan] [green]{ckpt_path}[/green][cyan]).[/cyan]"
            )
        else:
            console.print(
                f"[cyan]New run (--no-resume); checkpoint reinitialized at[/cyan] "
                f"[green]{ckpt_path}[/green]"
            )

    n_fusion_cleared = reconcile_fusion_checkpoint_params(
        ckpt,
        fusion_pairs=fusion_pairs,
        fusion_triples=fusion_triples,
    )
    if n_fusion_cleared > 0:
        console.print(
            f"[yellow]Fusion settings changed relative to the checkpoint; "
            f"dropped {n_fusion_cleared} cached fusion row(s). Singletons are unchanged.[/yellow]"
        )
    save_checkpoint_atomic(ckpt_path, ckpt)

    completed = set(ckpt.completed_combinations)
    results: list[ClusteringSearchSummaryRow] = results_from_checkpoint(ckpt)
    (
        best_overall_score,
        best_overall_model,
        best_overall_layer,
        best_per_model,
    ) = recompute_leaderboard_from_results(results)

    transform_config = TransformConfig(
        pca_components=pca_components,
        umap_components=umap_dim,
        cluster_viz_method=cluster_viz_method.lower(),
        pca_seed=pca_seed,
        umap_seed=umap_seed,
    )

    parquet_files = sorted(input_dir.glob(f"{prefix}*.parquet"))
    if not parquet_files:
        console.print(
            f"[red]No parquet files found matching pattern '{prefix}*.parquet' in {input_dir}[/red]"
        )
        return

    valid_files: list[tuple[pathlib.Path, str, str]] = []
    for file_path in parquet_files:
        if not file_path.exists():
            continue
        model, layer = parse_model_filename(file_path.name, prefix)
        if model is None or layer is None:
            continue
        # Layer as str matches checkpoint keys from "1:model/layer" and score_by_ml lookups.
        valid_files.append((file_path, model, str(layer)))

    if not valid_files:
        console.print("[red]No valid files to process[/red]")
        return

    path_by_ml = path_by_model_layer(valid_files)
    metrics_by_file: dict[tuple[str, str], list[pd.DataFrame]] = {}
    best_report: ModelSelectionReport | None = None
    detailed_grid_frames: list[pd.DataFrame] = []
    fine_metadata_frames: list[pd.DataFrame] = []

    run_single = mode in ("single", "all")
    run_fusion2 = mode in ("fusion2", "all")
    run_fusion3 = mode in ("fusion3", "all")

    if mode in ("fusion2", "fusion3"):
        if not singleton_score_by_model_layer_from_checkpoint(ckpt):
            console.print(
                "[red]Fusion mode requires singleton scores in the checkpoint.[/red] "
                "Run with ``--mode single`` (or ``all``) first, "
                "or ensure summaries for ``1:...`` combinations exist."
            )
            return

    if mode in ("fusion2", "fusion3"):
        expected_singletons = {
            combination_key_from_members([(m, layer)]) for _fp, m, layer in valid_files
        }
        missing = [k for k in sorted(expected_singletons) if k not in completed]
        if missing:
            console.print(
                f"[yellow]Warning:[/yellow] {len(missing)} singleton combination(s) "
                "are not marked complete in the checkpoint; fusion proxy scores may be incomplete."
            )

    # --- single-embedding combinations (arity 1) ---
    if run_single:
        ckpt.stages["single"] = "in_progress"
        save_checkpoint_atomic(ckpt_path, ckpt)

        total_tasks = len(valid_files) * n_sample
        done_tasks = sum(
            n_sample
            for _fp, m, layer in valid_files
            if combination_key_from_members([(m, layer)]) in completed
        )
        initial_total = total_tasks

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
                f"[cyan]Singletons: {len(valid_files)} files × {n_sample} samples...",
                total=initial_total,
                completed=done_tasks,
            )

            for file_path, model, layer in valid_files:
                comb_key = combination_key_from_members([(model, layer)])
                if comb_key in completed:
                    progress.advance(task, advance=n_sample)
                    continue

                file_metrics: list[pd.DataFrame] = []
                file_reports: list[ModelSelectionReport] = []
                all_metrics_dfs: list[pd.DataFrame] = []
                grid_report_samples: list[tuple[int, ModelSelectionReport]] = []
                combo_screener_frames: list[pd.DataFrame] = []

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

                status_parts_base = [
                    f"[cyan]{model}[/cyan]/[yellow]{layer}[/yellow]",
                ]
                try:
                    base_frame = load_selection_frame(
                        file_path=file_path,
                        config=optimization_config,
                        selected_labels=selected_labels,
                        embedding_read_status=lambda m, sp=status_parts_base: (
                            progress.update(
                                task,
                                description=" | ".join([*sp, m]),
                            )
                        ),
                    )
                except Exception as e:
                    base_frame = None
                    console.print(
                        "[yellow]Skipping combo (load failed)[/yellow] "
                        f"{model}/{layer}: {e}"
                    )

                if base_frame is None:
                    progress.advance(task, advance=n_sample)
                    continue

                for sample_idx in range(n_sample):
                    status_parts = [
                        *status_parts_base,
                        f"sample {sample_idx + 1}/{n_sample}",
                    ]
                    if best_overall_score is not None:
                        status_parts.append(
                            f"[green]Best: {best_overall_score:.3f}[/green] "
                            f"([cyan]{best_overall_model}[/cyan]/[yellow]{best_overall_layer}[/yellow])"
                        )
                    if model in best_per_model:
                        status_parts.append(
                            f"[magenta]{model}: {best_per_model[model]:.3f}[/magenta]"
                        )
                    progress.update(task, description=" | ".join(status_parts))

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
                    except Exception as e:
                        report = None
                        console.print(
                            "[yellow]Skipping failed sample[/yellow] "
                            f"{model}/{layer} sample {sample_idx + 1}: {e}"
                        )

                    if report is not None:
                        file_metrics.append(report.metrics_df)
                        file_reports.append(report)
                        grid_report_samples.append((sample_idx, report))
                        fine_metadata_frames.append(
                            clustering_metadata_df(
                                report,
                                model=model,
                                layer=layer,
                                sample_idx=sample_idx,
                            )
                        )
                        if report.screener_oos_datapoints is not None:
                            combo_screener_frames.append(
                                per_datapoint_scores_df(
                                    report.screener_oos_datapoints,
                                    combo_key=comb_key,
                                    model=model,
                                    layer=layer,
                                    sample_idx=sample_idx,
                                )
                            )
                        if (
                            best_report is None
                            or report.best_score > best_report.best_score
                        ):
                            best_report = report

                    progress.advance(task)
                    gc.collect()

                if file_reports:
                    pooled_mcs, _ = pooled_min_cluster_size_from_metrics_dfs(
                        all_metrics_dfs,
                        optimization_config,
                    )
                    grid_batch = [
                        grid_export_rows_from_report(
                            report,
                            model=model,
                            layer=layer,
                            sample_idx=sample_idx,
                            chosen_min_cluster_size=pooled_mcs,
                        )
                        for sample_idx, report in grid_report_samples
                    ]
                    detailed_grid_frames.extend(grid_batch)

                    metrics_by_file[(model, layer)] = file_metrics
                    summary_row = summarize_clustering_reports_for_search(
                        file_reports,
                        model=model,
                        layer=layer,
                        pooled_min_cluster_size=pooled_mcs,
                    )

                    if len(file_metrics) > 1:
                        plot_metrics_with_error_bars(
                            file_metrics,
                            report_path / f"{model}_{layer}_error_bars.png",
                            chosen_min_cluster_size=float(pooled_mcs),
                        )
                    else:
                        plot_metrics(
                            file_metrics[0], report_path / f"{model}_{layer}.png"
                        )

                    mark_combination_done(
                        ckpt,
                        ckpt_path,
                        combination_key=comb_key,
                        summary=summary_row,
                        singleton_score_key=comb_key,
                    )
                    n_new = len(file_reports)
                    merge_new_frames_into_per_sample_grid_csv(
                        detail_path,
                        grid_batch,
                    )
                    merge_new_frames_into_fine_metadata_jsonl(
                        fine_metadata_path,
                        fine_metadata_frames[-n_new:],
                    )
                    if combo_screener_frames:
                        merge_new_frames_into_screener_eval_jsonl(
                            fine_screener_eval_path,
                            combo_screener_frames,
                        )
                    completed.add(comb_key)
                    results = results_from_checkpoint(ckpt)
                    (
                        best_overall_score,
                        best_overall_model,
                        best_overall_layer,
                        best_per_model,
                    ) = recompute_leaderboard_from_results(results)
                else:
                    record_failure(
                        ckpt,
                        ckpt_path,
                        combination_key=comb_key,
                        message="All samples failed for this combination",
                    )

        ckpt.stages["single"] = "complete"
        save_checkpoint_atomic(ckpt_path, ckpt)

    # --- fusion combinations ---
    fusion_jobs: list[tuple[int, int]] = []
    if run_fusion2 and fusion_pairs > 0:
        fusion_jobs.append((2, fusion_pairs))
    if run_fusion3 and fusion_triples > 0:
        fusion_jobs.append((3, fusion_triples))

    if fusion_pairs == 0:
        ckpt.stages["fusion2"] = "skipped"
    if fusion_triples == 0:
        ckpt.stages["fusion3"] = "skipped"
    if not fusion_jobs:
        save_checkpoint_atomic(ckpt_path, ckpt)

    score_by_ml = singleton_score_by_model_layer_from_checkpoint(ckpt)
    singleton_items = singleton_items_by_dbcv_score(valid_files, score_by_ml)

    fusion_batches: list[
        tuple[
            int,
            int,
            list[tuple[list[pathlib.Path], list[str], list[str], float]],
        ]
    ] = []
    for order, top_k in fusion_jobs:
        cand = top_k_fusion_candidates_by_dbcv_proxy(singleton_items, order, top_k)
        if not cand:
            continue
        fusion_batches.append((order, top_k, cand))

    handled_fusion_orders: set[int] = set()
    for order, _top_k, candidates in fusion_batches:
        handled_fusion_orders.add(order)
        model_label = f"fusion{order}"
        stage_name = "fusion2" if order == 2 else "fusion3"
        fusion_task_total = 0
        for paths, models, layers, _sum_proxy in candidates:
            ckey = combination_key_from_members(
                list(zip(models, layers, strict=True)),
            )
            if ckey not in completed:
                fusion_task_total += n_sample

        if fusion_task_total > 0:
            console.print(
                "[cyan]Fused embeddings:[/cyan] evaluating remaining combinations "
                f"× {n_sample} sample(s)..."
            )

        ckpt.stages[stage_name] = "in_progress"
        save_checkpoint_atomic(ckpt_path, ckpt)

        if fusion_task_total > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                refresh_per_second=4,
            ) as fusion_progress:
                ftask = fusion_progress.add_task(
                    "[cyan]Fusion clustering...",
                    total=fusion_task_total,
                )
                for paths, models, layers, sum_proxy in candidates:
                    ordered = sorted(
                        zip(paths, models, layers, strict=True),
                        key=lambda t: (t[1], t[2]),
                    )
                    ordered_paths = [t[0] for t in ordered]
                    o_models = [t[1] for t in ordered]
                    o_layers = [t[2] for t in ordered]
                    layer_label = "+".join(
                        f"{m}/{lyr}" for m, lyr in zip(o_models, o_layers, strict=True)
                    )
                    comb_key = combination_key_from_members(
                        list(zip(o_models, o_layers, strict=True)),
                    )

                    if comb_key in completed:
                        fusion_progress.advance(ftask, advance=n_sample)
                        continue

                    fusion_metrics: list[pd.DataFrame] = []
                    fusion_reports: list[ModelSelectionReport] = []
                    fusion_all_metrics_dfs: list[pd.DataFrame] = []
                    fusion_grid_report_samples: list[
                        tuple[int, ModelSelectionReport]
                    ] = []
                    fusion_combo_screener_frames: list[pd.DataFrame] = []

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

                    fusion_status_base = (
                        f"[cyan]{model_label}[/cyan] "
                        f"[yellow]{layer_label}[/yellow] "
                        f"(mean singles≈{sum_proxy / len(paths):.3f})"
                    )
                    try:
                        fusion_base_frame = load_selection_frame(
                            file_paths=ordered_paths,
                            config=optimization_config,
                            selected_labels=selected_labels,
                            embedding_read_status=lambda m, fs=fusion_status_base: (
                                fusion_progress.update(
                                    ftask,
                                    description=f"{fs} | [dim]{m}[/dim]",
                                )
                            ),
                        )
                    except Exception as e:
                        fusion_base_frame = None
                        console.print(
                            "[yellow]Skipping fusion combo (load failed)[/yellow] "
                            f"{layer_label}: {e}"
                        )

                    if fusion_base_frame is None:
                        fusion_progress.advance(ftask, advance=n_sample)
                        continue

                    for sample_idx in range(n_sample):
                        fusion_status = (
                            f"{fusion_status_base} sample {sample_idx + 1}/{n_sample}"
                        )
                        fusion_progress.update(ftask, description=fusion_status)
                        try:
                            sample_frame = draw_selection_sample(
                                fusion_base_frame,
                                optimization_config,
                                sample_index=sample_idx,
                            )
                            report = evaluate_selection_sample(
                                sample_frame,
                                transform_config,
                                optimization_config=optimization_config,
                                all_metrics_dfs=fusion_all_metrics_dfs,
                            )
                        except Exception as e:
                            report = None
                            console.print(
                                "[yellow]Skipping failed fusion sample[/yellow] "
                                f"{layer_label} sample {sample_idx + 1}: {e}"
                            )
                        if report is not None:
                            fusion_metrics.append(report.metrics_df)
                            fusion_reports.append(report)
                            fusion_grid_report_samples.append((sample_idx, report))
                            fine_metadata_frames.append(
                                clustering_metadata_df(
                                    report,
                                    model=model_label,
                                    layer=layer_label,
                                    sample_idx=sample_idx,
                                )
                            )
                            if report.screener_oos_datapoints is not None:
                                fusion_combo_screener_frames.append(
                                    per_datapoint_scores_df(
                                        report.screener_oos_datapoints,
                                        combo_key=comb_key,
                                        model=model_label,
                                        layer=layer_label,
                                        sample_idx=sample_idx,
                                    )
                                )
                            if (
                                best_report is None
                                or report.best_score > best_report.best_score
                            ):
                                best_report = report
                        fusion_progress.advance(ftask)
                        gc.collect()

                    if fusion_reports:
                        pooled_mcs, _ = pooled_min_cluster_size_from_metrics_dfs(
                            fusion_all_metrics_dfs,
                            optimization_config,
                        )
                        fusion_grid_batch = [
                            grid_export_rows_from_report(
                                report,
                                model=model_label,
                                layer=layer_label,
                                sample_idx=sample_idx,
                                chosen_min_cluster_size=pooled_mcs,
                            )
                            for sample_idx, report in fusion_grid_report_samples
                        ]
                        detailed_grid_frames.extend(fusion_grid_batch)

                        fusion_summary = summarize_clustering_reports_for_search(
                            fusion_reports,
                            model=model_label,
                            layer=layer_label,
                            pooled_min_cluster_size=pooled_mcs,
                        )
                        metrics_by_file[(model_label, layer_label)] = fusion_metrics

                        safe = layer_label.replace("/", "_").replace("+", "__")
                        out_metric = report_path / f"{model_label}_{safe}.png"
                        if len(fusion_metrics) > 1:
                            plot_metrics_with_error_bars(
                                fusion_metrics,
                                out_metric,
                                chosen_min_cluster_size=float(pooled_mcs),
                            )
                        else:
                            plot_metrics(fusion_metrics[0], out_metric)

                        mark_combination_done(
                            ckpt,
                            ckpt_path,
                            combination_key=comb_key,
                            summary=fusion_summary,
                            singleton_score_key=None,
                        )
                        n_new = len(fusion_reports)
                        merge_new_frames_into_per_sample_grid_csv(
                            detail_path,
                            fusion_grid_batch,
                        )
                        merge_new_frames_into_fine_metadata_jsonl(
                            fine_metadata_path,
                            fine_metadata_frames[-n_new:],
                        )
                        if fusion_combo_screener_frames:
                            merge_new_frames_into_screener_eval_jsonl(
                                fine_screener_eval_path,
                                fusion_combo_screener_frames,
                            )
                        completed.add(comb_key)
                        results = results_from_checkpoint(ckpt)
                    else:
                        record_failure(
                            ckpt,
                            ckpt_path,
                            combination_key=comb_key,
                            message="All fusion samples failed for this combination",
                        )

        ckpt.stages[stage_name] = "complete"
        save_checkpoint_atomic(ckpt_path, ckpt)

    for order, _top_k in fusion_jobs:
        if order in handled_fusion_orders:
            continue
        stage_name = "fusion2" if order == 2 else "fusion3"
        ckpt.stages[stage_name] = "complete"
    if fusion_jobs:
        save_checkpoint_atomic(ckpt_path, ckpt)

    results = results_from_checkpoint(ckpt)
    if not results:
        console.print("[red]No results to save[/red]")
        return

    df_results = pd.DataFrame([r.to_flat_dict() for r in results])
    df_results.insert(
        0,
        "combo_key",
        [combo_key_for_results_row(row) for _, row in df_results.iterrows()],
    )
    df_results = df_results.sort_values(["model", "layer"])

    df_grid_detail = read_optional_csv(detail_path)
    if df_grid_detail is not None and not df_grid_detail.empty:
        df_grid_detail = dedupe_per_sample_grid(df_grid_detail)
        tmp_grid = detail_path.with_suffix(detail_path.suffix + ".tmp")
        df_grid_detail.to_csv(tmp_grid, index=False)
        tmp_grid.replace(detail_path)

    summary_figures = render_model_selection_summary_figures(
        report_path,
        checkpoint_path=ckpt_path,
        grid_csv_path=detail_path,
        fine_screener_eval_path=fine_screener_eval_path,
    )
    for fig_path in summary_figures.written_paths:
        console.print(
            f"[green]✓[/green] Summary figure saved to: [cyan]{fig_path}[/cyan]"
        )
    for note in summary_figures.skipped_messages:
        console.print(f"[yellow]{note}[/yellow]")

    console.print("\n[bold green]Results Summary[/bold green]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Layer", style="yellow")
    table.add_column("Best Size", justify="right", style="green")
    table.add_column("Clusters", justify="right", style="bright_blue")
    table.add_column("Properties", justify="right", style="magenta")
    table.add_column("Best Score", justify="right", style="blue")
    table.add_column("Scr AUC", justify="right", style="white")
    table.add_column("Comb AUC", justify="right", style="white")

    for _, row in df_results.iterrows():
        if n_sample > 1:
            best_size_str = f"{int(row['best_size'])} ± {row['best_size_std']:.1f}"
            clusters_str = (
                f"{row['n_clusters_emergent']:.1f} ± "
                f"{row['n_clusters_emergent_std']:.1f}"
            )
            properties_str = (
                f"{int(row['number_properties'])} ± {row['number_properties_std']:.1f}"
            )
            best_score_str = f"{row['best_score']:.3f} ± {row['best_score_std']:.3f}"
            sa = row.get("screener_auc_mean")
            ca = row.get("combined_auc_mean")
            scr_auc_str = (
                f"{float(sa):.3f} ± {float(row.get('screener_auc_std') or 0):.3f}"
                if sa is not None and not (isinstance(sa, float) and math.isnan(sa))
                else "—"
            )
            comb_auc_str = (
                f"{float(ca):.3f} ± {float(row.get('combined_auc_std') or 0):.3f}"
                if ca is not None and not (isinstance(ca, float) and math.isnan(ca))
                else "—"
            )
        else:
            best_size_str = str(int(row["best_size"]))
            clusters_str = str(int(round(row["n_clusters_emergent"])))
            properties_str = str(int(row["number_properties"]))
            best_score_str = f"{row['best_score']:.3f}"
            sa = row.get("screener_auc_mean")
            ca = row.get("combined_auc_mean")
            scr_auc_str = (
                f"{float(sa):.3f}"
                if sa is not None and not (isinstance(sa, float) and math.isnan(sa))
                else "—"
            )
            comb_auc_str = (
                f"{float(ca):.3f}"
                if ca is not None and not (isinstance(ca, float) and math.isnan(ca))
                else "—"
            )

        table.add_row(
            str(row["model"]),
            str(row["layer"]),
            best_size_str,
            clusters_str,
            properties_str,
            best_score_str,
            scr_auc_str,
            comb_auc_str,
        )

    console.print(table)
    if df_grid_detail is not None and not df_grid_detail.empty:
        console.print(
            f"[green]✓[/green] Per-sample grid (all min_cluster_size values) saved to: "
            f"[cyan]{detail_path}[/cyan]"
        )
    fm: pd.DataFrame | None = None
    if fine_metadata_path.exists():
        fm = read_optional_jsonl_gzip(fine_metadata_path)
        if fm is not None:
            console.print(
                "[green]✓[/green] Fine clustering metadata (gzip JSON Lines) saved to: "
                f"[cyan]{fine_metadata_path}[/cyan]"
            )
    if fine_screener_eval_path.exists():
        se = read_optional_jsonl_gzip(fine_screener_eval_path)
        if se is not None:
            console.print(
                "[green]✓[/green] Fine screener eval datapoints saved to: "
                f"[cyan]{fine_screener_eval_path}[/cyan]"
            )

    top_idx = df_results["best_score"].idxmax()
    top_row = df_results.loc[top_idx]
    top_summary = clustering_search_summary_row_from_flat_dict(
        {str(k): top_row[k] for k in top_row.index}
    )
    console.print(
        f"\n[bold green]Best mean DBCV (best_score): {float(top_row['best_score']):.3f}[/bold green] "
        f"([cyan]{top_row['model']}[/cyan]/[yellow]{top_row['layer']}[/yellow])"
    )

    if best_report is None:
        viz_config = clustering_optimization_config_for_run(
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
            "[cyan]Materializing best clustering report for UMAP (not held in memory)...[/cyan]"
        )
        best_report = materialize_best_report(
            top_summary,
            valid_files=valid_files,
            path_by_ml=path_by_ml,
            transform_config=transform_config,
            optimization_config=viz_config,
            selected_labels=selected_labels,
        )

    if best_report is not None:
        cluster_viz_path = report_path / "cluster_viz_best.html"
        console.print(
            "[green]✓[/green] Generating cluster-space visualization for best model..."
        )
        cluster_viz_df = pd.DataFrame(
            best_report.cluster_viz,
            columns=[
                f"cviz_{j:02d}" for j in range(int(best_report.cluster_viz.shape[1]))
            ],
            index=best_report.assignments.index,
        )
        assign_cols = (
            ["entity", "cluster"]
            if "entity" in best_report.assignments.columns
            else ["cluster"]
        )
        plot_df = pd.concat(
            [
                best_report.assignments[assign_cols].rename(
                    columns={"cluster": "class"}
                ),
                cluster_viz_df,
            ],
            axis=1,
        )
        plot_cluster_viz(
            plot_df,
            output_path=str(cluster_viz_path),
            viz_method=best_report.cluster_viz_method,
        )
        console.print(
            "[green]✓[/green] Cluster visualization saved to: "
            f"[cyan]{cluster_viz_path}[/cyan]"
        )

    checkpoint_payload = {
        "path": str(ckpt_path),
        "stages": dict(ckpt.stages),
        "completed_combinations_count": len(ckpt.completed_combinations),
        "failure_count": len(ckpt.failures),
        "resumed": resumed_from_checkpoint,
    }
    best_overall_payload: dict[str, object] | None = None
    if best_overall_score is not None:
        best_overall_payload = {
            "model": best_overall_model,
            "layer": best_overall_layer,
            "best_score": float(best_overall_score),
        }
        br_sel = df_results.loc[
            (df_results["model"].astype(str) == str(best_overall_model))
            & (df_results["layer"].astype(str) == str(best_overall_layer))
        ]
        bk = (
            br_sel.iloc[0]
            if len(br_sel) > 0
            else df_results.loc[df_results["best_score"].idxmax()]
        )
        sbk_mean = bk.get("screener_auc_mean")
        cb_mean = bk.get("combined_auc_mean")
        ob_mean = bk.get("oov_auc_mean")
        if sbk_mean is not None and not (
            isinstance(sbk_mean, float) and math.isnan(sbk_mean)
        ):
            best_overall_payload["screener_auc"] = float(sbk_mean)
        if cb_mean is not None and not (
            isinstance(cb_mean, float) and math.isnan(cb_mean)
        ):
            best_overall_payload["combined_auc"] = float(cb_mean)
        if ob_mean is not None and not (
            isinstance(ob_mean, float) and math.isnan(ob_mean)
        ):
            best_overall_payload["oov_auc"] = float(ob_mean)
        s_kind = bk.get("screener_best_kind")
        if isinstance(s_kind, str) and s_kind:
            best_overall_payload["screener_best_kind"] = s_kind
        o_kind = bk.get("oov_winner_kind")
        if isinstance(o_kind, str) and o_kind:
            best_overall_payload["oov_winner_kind"] = o_kind

    run_report = ModelSelectionRunReport(
        schema=MODEL_SELECTION_RUN_REPORT_SCHEMA,
        generated_at=utc_now_iso(),
        run_fingerprint=run_fingerprint,
        run_config=fp_payload,
        checkpoint=checkpoint_payload,
        combinations=[
            {
                "combination_key": combination_key_from_members(
                    parse_fusion_members(r.layer)
                    if r.model.startswith("fusion")
                    else [(r.model, r.layer)]
                ),
                "model": r.model,
                "layer": r.layer,
                "summary": json_ready_flat_row(r),
            }
            for r in results
        ],
        failures=[
            {
                "combination_key": f.combination_key,
                "error": f.error,
                "at": f.at,
            }
            for f in ckpt.failures
        ],
        best_overall=best_overall_payload,
        best_per_model={k: float(v) for k, v in sorted(best_per_model.items())},
    )
    write_model_selection_run_report_json(run_report_json_path, run_report)
    console.print(
        f"\n[green]✓[/green] Standardized run report saved to: [cyan]{run_report_json_path}[/cyan]"
    )
