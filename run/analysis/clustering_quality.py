import click
import gc
import pathlib
import sys

import pandas as pd
from numpy.random import RandomState
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from pelinker.analysis import (
    estimate_model_clustering,
    metrics_df_with_grid_sample_columns,
    pooled_min_cluster_size_from_metrics_dfs,
)
from pelinker.clustering_fusion_ranking import (
    singleton_items_by_dbcv_score,
    top_k_fusion_candidates_by_dbcv_proxy,
)
from pelinker.clustering_quality_checkpoint import (
    ClusteringQualityCheckpoint,
    DEFAULT_CHECKPOINT_NAME,
    FailureRecord,
    RunMode,
    combination_key_from_members,
    compute_run_fingerprint,
    fingerprint_config_from_cli,
    load_checkpoint,
    model_layer_from_singleton_key,
    new_checkpoint,
    reconcile_fusion_checkpoint_params,
    save_checkpoint_atomic,
    score_by_model_layer_from_checkpoint,
    utc_now_iso,
)
from pelinker.config import ClusteringOptimizationConfig
from pelinker.ops import parse_model_filename
from pelinker.plotting import (
    GRID_COL_CHOSEN_MIN_CLUSTER_SIZE,
    GRID_COL_SAMPLE_BEST_DBCV,
    GRID_COL_SAMPLE_ARI,
    plot_dbcv_vs_ari_from_grid,
    plot_heatmap,
    plot_metrics,
    plot_metrics_with_error_bars,
    plot_umap_viz,
)
from pelinker.reporting import (
    ClusteringReport,
    ClusteringSearchSummaryRow,
    clustering_search_summary_row_from_flat_dict,
    summarize_clustering_reports_for_search,
)
from pelinker.transform import TransformConfig


def _path_by_model_layer(
    valid_files: list[tuple[pathlib.Path, str, str]],
) -> dict[tuple[str, str], pathlib.Path]:
    return {(m, layer): fp for fp, m, layer in valid_files}


def _parse_fusion_members(layer_label: str) -> list[tuple[str, str]]:
    members: list[tuple[str, str]] = []
    for part in layer_label.split("+"):
        p = part.strip()
        model, _, layer = p.partition("/")
        members.append((model, layer))
    return sorted(members, key=lambda t: (t[0], t[1]))


def _ordered_paths_for_fusion(
    path_by_ml: dict[tuple[str, str], pathlib.Path],
    members: list[tuple[str, str]],
) -> list[pathlib.Path]:
    return [path_by_ml[t] for t in members]


def _update_leaderboard_fixed(
    summary_row: ClusteringSearchSummaryRow,
    *,
    best_overall_score: float | None,
    best_overall_model: str | None,
    best_overall_layer: str | None,
    best_per_model: dict[str, float],
) -> tuple[float | None, str | None, str | None, dict[str, float]]:
    mean_dbcv = summary_row.dbcv.mean
    model, layer = summary_row.model, summary_row.layer
    if not model.startswith("fusion"):
        if best_overall_score is None or mean_dbcv > best_overall_score:
            best_overall_score = mean_dbcv
            best_overall_model = model
            best_overall_layer = layer
        if model not in best_per_model or mean_dbcv > best_per_model[model]:
            best_per_model[model] = mean_dbcv
    return best_overall_score, best_overall_model, best_overall_layer, best_per_model


def _recompute_leaderboard_from_results(
    results: list[ClusteringSearchSummaryRow],
) -> tuple[float | None, str | None, str | None, dict[str, float]]:
    best_overall_score = None
    best_overall_model = None
    best_overall_layer = None
    best_per_model: dict[str, float] = {}
    for r in results:
        best_overall_score, best_overall_model, best_overall_layer, best_per_model = (
            _update_leaderboard_fixed(
                r,
                best_overall_score=best_overall_score,
                best_overall_model=best_overall_model,
                best_overall_layer=best_overall_layer,
                best_per_model=best_per_model,
            )
        )
    return best_overall_score, best_overall_model, best_overall_layer, best_per_model


def _materialize_best_report(
    top: ClusteringSearchSummaryRow,
    *,
    valid_files: list[tuple[pathlib.Path, str, str]],
    path_by_ml: dict[tuple[str, str], pathlib.Path],
    transform_config: TransformConfig,
    optimization_config: ClusteringOptimizationConfig,
    selected_labels: set[str] | None,
) -> ClusteringReport | None:
    if top.model.startswith("fusion"):
        members = _parse_fusion_members(top.layer)
        try:
            ordered_paths = _ordered_paths_for_fusion(path_by_ml, members)
        except KeyError:
            return None
        return estimate_model_clustering(
            transform_config=transform_config,
            optimization_config=optimization_config,
            file_paths=ordered_paths,
            selected_labels=selected_labels,
            all_metrics_dfs=None,
            show_embedding_read_progress=sys.stdout.isatty(),
        )
    key = (top.model, top.layer)
    path = path_by_ml.get(key)
    if path is None:
        return None
    return estimate_model_clustering(
        transform_config=transform_config,
        optimization_config=optimization_config,
        file_path=path,
        selected_labels=selected_labels,
        all_metrics_dfs=None,
        show_embedding_read_progress=sys.stdout.isatty(),
    )


def _fine_clustering_metadata_df(
    report: ClusteringReport,
    *,
    model: str,
    layer: str,
    sample_idx: int,
) -> pd.DataFrame:
    """Per-sample clustering assignments for downstream analysis."""
    cols = ["model", "layer", "sample_idx", "entity", "cluster"]
    optional_cols = ["pmid", "mention"]
    present_optional = [c for c in optional_cols if c in report.assignments.columns]
    keep = [
        c
        for c in ["entity", "cluster", *present_optional]
        if c in report.assignments.columns
    ]
    if "entity" not in keep and "property" in report.assignments.columns:
        keep = ["entity" if c == "property" else c for c in keep]
        out0 = report.assignments.rename(columns={"property": "entity"})
    else:
        out0 = report.assignments
    if "entity" not in keep or "cluster" not in keep:
        return pd.DataFrame(columns=cols + present_optional)
    out = out0[keep].copy()
    out.insert(0, "sample_idx", sample_idx)
    out.insert(0, "layer", layer)
    out.insert(0, "model", model)
    return out


def _per_sample_grid_column_order() -> list[str]:
    return [
        "model",
        "layer",
        "sample_idx",
        GRID_COL_CHOSEN_MIN_CLUSTER_SIZE,
        GRID_COL_SAMPLE_BEST_DBCV,
        GRID_COL_SAMPLE_ARI,
        "min_cluster_size",
        "icm",
        "n_clusters",
        "dbcv",
        "ari",
    ]


def _dedupe_per_sample_grid(df: pd.DataFrame) -> pd.DataFrame:
    grid_cols = _per_sample_grid_column_order()
    ordered = [c for c in grid_cols if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    out = df[ordered + tail]
    dup_subset = [c for c in grid_cols if c in out.columns]
    if dup_subset:
        out = out.drop_duplicates(subset=dup_subset, keep="last")
    return out


def _read_optional_csv(path: pathlib.Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _merge_new_frames_into_per_sample_grid_csv(
    detail_path: pathlib.Path,
    new_frames: list[pd.DataFrame],
) -> None:
    """Append grid rows to ``results_grid_per_sample.csv`` (merge + dedupe, atomic replace)."""
    if not new_frames:
        return
    new_df = pd.concat(new_frames, ignore_index=True)
    if new_df.empty:
        return
    prior = _read_optional_csv(detail_path)
    if prior is not None and not prior.empty:
        merged = pd.concat([prior, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = _dedupe_per_sample_grid(merged)
    tmp = detail_path.with_suffix(detail_path.suffix + ".tmp")
    merged.to_csv(tmp, index=False)
    tmp.replace(detail_path)


def _fine_metadata_dedupe_subset(df: pd.DataFrame) -> list[str]:
    wanted = [
        "model",
        "layer",
        "sample_idx",
        "pmid",
        "mention",
        "entity",
        "cluster",
    ]
    return [c for c in wanted if c in df.columns]


def _dedupe_fine_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = _fine_metadata_dedupe_subset(df)
    if cols:
        return df.drop_duplicates(subset=cols, keep="last")
    return df


def _merge_new_frames_into_fine_metadata_pickle(
    fine_metadata_path: pathlib.Path,
    new_frames: list[pd.DataFrame],
) -> None:
    if not new_frames:
        return
    new_df = pd.concat(new_frames, ignore_index=True)
    if new_df.empty:
        return
    prior_frames: list[pd.DataFrame] = []
    if fine_metadata_path.exists():
        try:
            prior = pd.read_pickle(fine_metadata_path, compression="gzip")
            if prior is not None and not prior.empty:
                prior_frames.append(prior)
        except Exception:
            pass
    merged = pd.concat(prior_frames + [new_df], ignore_index=True)
    merged = _dedupe_fine_metadata_df(merged)
    tmp = fine_metadata_path.with_name(fine_metadata_path.name + ".tmp")
    merged.to_pickle(tmp, compression="gzip")
    tmp.replace(fine_metadata_path)


def _mark_combination_done(
    ckpt: ClusteringQualityCheckpoint,
    ckpt_path: pathlib.Path,
    *,
    combination_key: str,
    summary: ClusteringSearchSummaryRow,
    singleton_score_key: str | None,
) -> None:
    if combination_key not in ckpt.completed_combinations:
        ckpt.completed_combinations.append(combination_key)
    flat = summary.to_flat_dict()
    ckpt.summaries_by_key[combination_key] = dict(flat)
    if singleton_score_key is not None:
        ckpt.singleton_scores_by_key[singleton_score_key] = float(summary.dbcv.mean)
    save_checkpoint_atomic(ckpt_path, ckpt)


def _record_failure(
    ckpt: ClusteringQualityCheckpoint,
    ckpt_path: pathlib.Path,
    *,
    combination_key: str,
    message: str,
) -> None:
    ckpt.failures.append(
        FailureRecord(combination_key=combination_key, error=message, at=utc_now_iso())
    )
    save_checkpoint_atomic(ckpt_path, ckpt)


def _singleton_score_by_model_layer_from_checkpoint(
    ckpt: ClusteringQualityCheckpoint,
) -> dict[tuple[str, str], float]:
    """Mean DBCV per (model, layer) for fusion proxy (singletons only)."""
    out = score_by_model_layer_from_checkpoint(ckpt.singleton_scores_by_key)
    if out:
        return out
    for key, row in ckpt.summaries_by_key.items():
        if not key.startswith("1:"):
            continue
        ml = model_layer_from_singleton_key(key)
        score = row.get("best_score")
        if score is not None:
            out[ml] = float(score)
    return out


def _results_from_checkpoint(
    ckpt: ClusteringQualityCheckpoint,
) -> list[ClusteringSearchSummaryRow]:
    return [
        clustering_search_summary_row_from_flat_dict(dict(row))
        for _k, row in sorted(ckpt.summaries_by_key.items(), key=lambda item: item[0])
    ]


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Directory containing parquet files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Output directory for all results (default: input_dir)",
)
@click.option(
    "--umap-dim",
    type=click.INT,
    default=8,
    help="UMAP dimensionality for clustering (range: 3-5)",
)
@click.option(
    "--pca-components",
    type=click.INT,
    default=100,
    help="Number of PCA components for dimensionality reduction",
)
@click.option(
    "--min-class-size",
    type=click.INT,
    default=20,
    help="Minimum class size for filtering",
)
@click.option(
    "--seed",
    type=click.INT,
    default=13,
    help="Random seed",
)
@click.option(
    "--frac",
    type=click.FLOAT,
    default=0.1,
    help="Fraction of dataset to sample",
)
@click.option(
    "--n-embedding-batches",
    type=click.INT,
    default=None,
    help=(
        "Max parquet read batches per file (see --batch-size for rows per batch); "
        "omit to read all batches"
    ),
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=1000,
    help="Rows per batch when reading mention-level embedding parquet files",
)
@click.option(
    "--prefix",
    type=click.STRING,
    default="res",
    help="Optional prefix for input embedding files to differentiate between models",
)
@click.option(
    "--n-sample",
    type=click.INT,
    default=1,
    help="Number of samples/runs per (model, layer) combination",
)
@click.option(
    "--selected-labels-kb-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Optional path to selected labels KB CSV file. If provided, clustering will only use labels from this KB.",
)
@click.option(
    "--max-scale",
    type=click.INT,
    default=60,
    show_default=True,
    help="Exclusive upper bound for grid evaluation of min_cluster_size (numpy.arange end).",
)
@click.option(
    "--min-scale",
    type=click.INT,
    default=None,
    help=(
        "Inclusive lower bound for min_cluster_size on the grid. "
        "Default: max(1, min_class_size // 2) (legacy: half of --min-class-size)."
    ),
)
@click.option(
    "--clustering-grid-step",
    type=click.INT,
    default=5,
    show_default=True,
    help="Step between consecutive min_cluster_size values on the optimization grid.",
)
@click.option(
    "--fusion-pairs",
    type=click.INT,
    default=5,
    show_default=True,
    help=(
        "After scoring single embeddings (DBCV), evaluate fused pairs: "
        "pick this many distinct pairs with highest sum of singleton DBCV. 0 disables."
    ),
)
@click.option(
    "--fusion-triples",
    type=click.INT,
    default=0,
    show_default=True,
    help=("Same as --fusion-pairs but for three-way fusions (costly). 0 disables."),
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help=(
        "If the checkpoint file exists and matches the run fingerprint, skip completed work. "
        "If the file is missing, start fresh and create it. Use --no-resume to ignore an "
        "existing checkpoint and reinitialize (overwrites on save)."
    ),
)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help=f"Checkpoint JSON path (default: <output-dir>/{DEFAULT_CHECKPOINT_NAME})",
)
@click.option(
    "--mode",
    type=click.Choice(["single", "fusion2", "fusion3", "all"]),
    default="all",
    show_default=True,
    help=(
        "single: only single-embedding combinations; fusion2/fusion3: only that fusion order "
        "(requires prior singleton scores in checkpoint); all: singletons then enabled fusions."
    ),
)
def main(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    umap_dim: int,
    pca_components: int,
    min_class_size: int,
    seed: int,
    frac: float,
    n_embedding_batches: int | None,
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
):
    """
    Process multiple parquet files and compute optimal cluster sizes.

    Files should follow the pattern: <prefix>_<model>_<layer>.parquet

    After scoring each (model, layer) alone (mean DBCV as ``best_score``), optionally
    evaluates fused embeddings: pairs/triples with the highest sum of singleton DBCV
    scores (see ``--fusion-pairs`` / ``--fusion-triples``), then clusters the
    concatenated mention-level vectors via ``estimate_model_clustering(..., file_paths=...)``.

    Checkpointing is on by default (``--resume``): progress is saved under the output directory.
    Use ``--no-resume`` to discard the on-disk checkpoint and start from an empty state.
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

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.csv"
    detail_path = output_dir / "results_grid_per_sample.csv"
    fine_metadata_path = output_dir / "fine_clustering_metadata.pkl.gz"
    if not resume:
        for artifact in (detail_path, fine_metadata_path):
            try:
                if artifact.exists():
                    artifact.unlink()
            except OSError:
                pass

    fp_payload = fingerprint_config_from_cli(
        input_dir=input_dir,
        umap_dim=umap_dim,
        pca_components=pca_components,
        min_class_size=min_class_size,
        seed=seed,
        frac=frac,
        n_embedding_batches=n_embedding_batches,
        batch_size=batch_size,
        prefix=prefix,
        n_sample=n_sample,
        selected_labels_kb_path=selected_labels_kb_path,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
    )
    run_fingerprint = compute_run_fingerprint(fp_payload)

    ckpt_path = (
        checkpoint_path.expanduser()
        if checkpoint_path is not None
        else output_dir / DEFAULT_CHECKPOINT_NAME
    )

    if resume and ckpt_path.exists():
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
    results: list[ClusteringSearchSummaryRow] = _results_from_checkpoint(ckpt)
    (
        best_overall_score,
        best_overall_model,
        best_overall_layer,
        best_per_model,
    ) = _recompute_leaderboard_from_results(results)

    transform_config = TransformConfig(
        pca_components=pca_components,
        umap_components=umap_dim,
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

    path_by_ml = _path_by_model_layer(valid_files)
    metrics_by_file: dict[tuple[str, str], list[pd.DataFrame]] = {}
    best_report: ClusteringReport | None = None
    detailed_grid_frames: list[pd.DataFrame] = []
    fine_metadata_frames: list[pd.DataFrame] = []

    run_single = mode in ("single", "all")
    run_fusion2 = mode in ("fusion2", "all")
    run_fusion3 = mode in ("fusion3", "all")

    if mode in ("fusion2", "fusion3"):
        if not _singleton_score_by_model_layer_from_checkpoint(ckpt):
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
                file_reports: list[ClusteringReport] = []
                all_metrics_dfs: list[pd.DataFrame] = []
                grid_batch: list[pd.DataFrame] = []

                optimization_config = ClusteringOptimizationConfig(
                    min_class_size=min_class_size,
                    max_scale=max_scale,
                    min_scale=min_scale,
                    clustering_grid_step=clustering_grid_step,
                    rns=RandomState(seed=seed),
                    frac=frac,
                    n_embedding_batches=n_embedding_batches,
                    batch_size=batch_size,
                    optimization_method="mean",
                )

                for sample_idx in range(n_sample):
                    status_parts = [
                        f"[cyan]{model}[/cyan]/[yellow]{layer}[/yellow]",
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
                        report = estimate_model_clustering(
                            transform_config=transform_config,
                            optimization_config=optimization_config,
                            file_path=file_path,
                            selected_labels=selected_labels,
                            all_metrics_dfs=all_metrics_dfs,
                            embedding_read_status=lambda m, sp=status_parts: (
                                progress.update(
                                    task,
                                    description=" | ".join([*sp, m]),
                                )
                            ),
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
                        grid_batch.append(
                            metrics_df_with_grid_sample_columns(
                                report,
                                model=model,
                                layer=layer,
                                sample_idx=sample_idx,
                            )
                        )
                        fine_metadata_frames.append(
                            _fine_clustering_metadata_df(
                                report,
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
                    for _gf in grid_batch:
                        _gf[GRID_COL_CHOSEN_MIN_CLUSTER_SIZE] = pooled_mcs
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
                            output_dir / f"{model}_{layer}_error_bars.png",
                            chosen_min_cluster_size=float(pooled_mcs),
                        )
                    else:
                        plot_metrics(
                            file_metrics[0], output_dir / f"{model}_{layer}.png"
                        )

                    _mark_combination_done(
                        ckpt,
                        ckpt_path,
                        combination_key=comb_key,
                        summary=summary_row,
                        singleton_score_key=comb_key,
                    )
                    n_new = len(file_reports)
                    _merge_new_frames_into_per_sample_grid_csv(
                        detail_path,
                        grid_batch,
                    )
                    _merge_new_frames_into_fine_metadata_pickle(
                        fine_metadata_path,
                        fine_metadata_frames[-n_new:],
                    )
                    completed.add(comb_key)
                    results = _results_from_checkpoint(ckpt)
                    (
                        best_overall_score,
                        best_overall_model,
                        best_overall_layer,
                        best_per_model,
                    ) = _recompute_leaderboard_from_results(results)
                else:
                    _record_failure(
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

    score_by_ml = _singleton_score_by_model_layer_from_checkpoint(ckpt)
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
                    fusion_reports: list[ClusteringReport] = []
                    fusion_all_metrics_dfs: list[pd.DataFrame] = []
                    fusion_grid_batch: list[pd.DataFrame] = []

                    optimization_config = ClusteringOptimizationConfig(
                        min_class_size=min_class_size,
                        max_scale=max_scale,
                        min_scale=min_scale,
                        clustering_grid_step=clustering_grid_step,
                        rns=RandomState(seed=seed),
                        frac=frac,
                        n_embedding_batches=n_embedding_batches,
                        batch_size=batch_size,
                        optimization_method="mean",
                    )

                    for sample_idx in range(n_sample):
                        fusion_status = (
                            f"[cyan]{model_label}[/cyan] "
                            f"[yellow]{layer_label}[/yellow] "
                            f"(mean singles≈{sum_proxy / len(paths):.3f}) "
                            f"sample {sample_idx + 1}/{n_sample}"
                        )
                        fusion_progress.update(ftask, description=fusion_status)
                        try:
                            report = estimate_model_clustering(
                                transform_config=transform_config,
                                optimization_config=optimization_config,
                                file_paths=ordered_paths,
                                selected_labels=selected_labels,
                                all_metrics_dfs=fusion_all_metrics_dfs,
                                embedding_read_status=lambda m, fs=fusion_status: (
                                    fusion_progress.update(
                                        ftask,
                                        description=f"{fs} | [dim]{m}[/dim]",
                                    )
                                ),
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
                            fusion_grid_batch.append(
                                metrics_df_with_grid_sample_columns(
                                    report,
                                    model=model_label,
                                    layer=layer_label,
                                    sample_idx=sample_idx,
                                )
                            )
                            fine_metadata_frames.append(
                                _fine_clustering_metadata_df(
                                    report,
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
                        for _gf in fusion_grid_batch:
                            _gf[GRID_COL_CHOSEN_MIN_CLUSTER_SIZE] = pooled_mcs
                        detailed_grid_frames.extend(fusion_grid_batch)

                        fusion_summary = summarize_clustering_reports_for_search(
                            fusion_reports,
                            model=model_label,
                            layer=layer_label,
                            pooled_min_cluster_size=pooled_mcs,
                        )
                        metrics_by_file[(model_label, layer_label)] = fusion_metrics

                        safe = layer_label.replace("/", "_").replace("+", "__")
                        out_metric = output_dir / f"{model_label}_{safe}.png"
                        if len(fusion_metrics) > 1:
                            plot_metrics_with_error_bars(
                                fusion_metrics,
                                out_metric,
                                chosen_min_cluster_size=float(pooled_mcs),
                            )
                        else:
                            plot_metrics(fusion_metrics[0], out_metric)

                        _mark_combination_done(
                            ckpt,
                            ckpt_path,
                            combination_key=comb_key,
                            summary=fusion_summary,
                            singleton_score_key=None,
                        )
                        n_new = len(fusion_reports)
                        _merge_new_frames_into_per_sample_grid_csv(
                            detail_path,
                            fusion_grid_batch,
                        )
                        _merge_new_frames_into_fine_metadata_pickle(
                            fine_metadata_path,
                            fine_metadata_frames[-n_new:],
                        )
                        completed.add(comb_key)
                        results = _results_from_checkpoint(ckpt)
                    else:
                        _record_failure(
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

    results = _results_from_checkpoint(ckpt)
    if not results:
        console.print("[red]No results to save[/red]")
        return

    df_results = pd.DataFrame([r.to_flat_dict() for r in results])
    df_results = df_results.sort_values(["model", "layer"])
    df_results.to_csv(output_path, index=False)

    df_grid_detail = _read_optional_csv(detail_path)
    if df_grid_detail is not None and not df_grid_detail.empty:
        df_grid_detail = _dedupe_per_sample_grid(df_grid_detail)
        tmp_grid = detail_path.with_suffix(detail_path.suffix + ".tmp")
        df_grid_detail.to_csv(tmp_grid, index=False)
        tmp_grid.replace(detail_path)
        scatter_path = output_dir / "model.dbcv_vs_ari.png"
        if plot_dbcv_vs_ari_from_grid(df_grid_detail, scatter_path):
            console.print(
                f"[green]✓[/green] DBCV vs ARI scatter saved to: "
                f"[cyan]{scatter_path}[/cyan]"
            )

    df_heatmap = df_results[~df_results["model"].isin(["fusion2", "fusion3"])].copy()

    heatmap_path = output_dir / "model.perf.heatmap.png"
    if len(df_heatmap) > 0:
        plot_heatmap(
            df_heatmap, heatmap_path, metric="best_score", metric_label="Best Score"
        )
    else:
        console.print(
            "[yellow]Skipping score heatmap: no single-embedding rows[/yellow]"
        )

    if (
        len(df_heatmap) > 0
        and "ari" in df_heatmap.columns
        and df_heatmap["ari"].notna().any()
    ):
        ari_heatmap_path = output_dir / "model.ari.heatmap.png"
        plot_heatmap(
            df_heatmap,
            ari_heatmap_path,
            metric="ari",
            metric_label="ARI",
        )

    console.print("\n[bold green]Results Summary[/bold green]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Layer", style="yellow")
    table.add_column("Best Size", justify="right", style="green")
    table.add_column("Clusters", justify="right", style="bright_blue")
    table.add_column("Properties", justify="right", style="magenta")
    table.add_column("Best Score", justify="right", style="blue")

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
        else:
            best_size_str = str(int(row["best_size"]))
            clusters_str = str(int(round(row["n_clusters_emergent"])))
            properties_str = str(int(row["number_properties"]))
            best_score_str = f"{row['best_score']:.3f}"

        table.add_row(
            str(row["model"]),
            str(row["layer"]),
            best_size_str,
            clusters_str,
            properties_str,
            best_score_str,
        )

    console.print(table)
    console.print(f"\n[green]✓[/green] Results saved to: [cyan]{output_path}[/cyan]")
    if df_grid_detail is not None and not df_grid_detail.empty:
        console.print(
            f"[green]✓[/green] Per-sample grid (all min_cluster_size values) saved to: "
            f"[cyan]{detail_path}[/cyan]"
        )
    if fine_metadata_path.exists():
        try:
            fm = pd.read_pickle(fine_metadata_path, compression="gzip")
        except Exception:
            fm = None
        if fm is not None and not fm.empty:
            console.print(
                f"[green]✓[/green] Fine clustering metadata (gzipped pickle) saved to: "
                f"[cyan]{fine_metadata_path}[/cyan]"
            )
    if len(df_heatmap) > 0:
        console.print(f"[green]✓[/green] Heatmap saved to: [cyan]{heatmap_path}[/cyan]")

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
        viz_config = ClusteringOptimizationConfig(
            min_class_size=min_class_size,
            max_scale=max_scale,
            min_scale=min_scale,
            clustering_grid_step=clustering_grid_step,
            rns=RandomState(seed=seed),
            frac=frac,
            n_embedding_batches=n_embedding_batches,
            batch_size=batch_size,
            optimization_method="mean",
        )
        console.print(
            "[cyan]Materializing best clustering report for UMAP (not held in memory)...[/cyan]"
        )
        best_report = _materialize_best_report(
            top_summary,
            valid_files=valid_files,
            path_by_ml=path_by_ml,
            transform_config=transform_config,
            optimization_config=viz_config,
            selected_labels=selected_labels,
        )

    if best_report is not None:
        umap_viz_path = output_dir / "umap_best.html"
        console.print(
            "[green]✓[/green] Generating UMAP visualization for best model..."
        )
        umap_viz_df = pd.DataFrame(
            best_report.umap_visualization,
            columns=[
                f"uviz_{j:02d}"
                for j in range(int(best_report.umap_visualization.shape[1]))
            ],
            index=best_report.assignments.index,
        )
        plot_df = pd.concat(
            [
                best_report.assignments[["cluster"]].rename(
                    columns={"cluster": "class"}
                ),
                umap_viz_df,
            ],
            axis=1,
        )
        plot_umap_viz(plot_df, output_path=str(umap_viz_path))
        console.print(
            f"[green]✓[/green] UMAP visualization saved to: [cyan]{umap_viz_path}[/cyan]"
        )


if __name__ == "__main__":
    main()
