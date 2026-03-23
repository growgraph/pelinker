import click
import pathlib
import numpy as np

import pandas as pd
from numpy.random import RandomState
from pelinker.config import ClusteringOptimizationConfig
from pelinker.plotting import (
    plot_metrics_with_error_bars,
    plot_heatmap,
    plot_umap_viz,
    plot_metrics,
)
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from pelinker.analysis import (
    estimate_model_clustering,
)
from pelinker.ops import parse_model_filename
from pelinker.transform import TransformConfig


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
    "--head",
    type=click.INT,
    default=None,
    help="Number of batches to take (None for all)",
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=1000,
    help="Batch size for reading files",
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
    default=120,
    show_default=True,
    help="Maximum value for grid evaluation of min_cluster_size",
)
def main(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    umap_dim: int,
    pca_components: int,
    min_class_size: int,
    seed: int,
    frac: float,
    head: int,
    batch_size: int,
    n_sample: int,
    prefix: str,
    selected_labels_kb_path: pathlib.Path | None,
    max_scale: int,
):
    """
    Process multiple parquet files and compute optimal cluster sizes.

    Files should follow the pattern: <prefix>_<model>_<layer>.parquet
    """
    # Configure console to work better in PyCharm and other IDEs
    # Use legacy_windows=False and force_terminal to ensure progress bars work
    console = Console(force_terminal=True, width=120, legacy_windows=False)
    input_dir = input_dir.expanduser()

    # Load selected labels KB if provided
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
            # Extract labels from the selected labels KB
            # The file should have a 'label' column
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

    # Set up output directory
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create transform config from CLI options
    transform_config = TransformConfig(
        pca_components=pca_components,
        umap_components=umap_dim,
    )

    # Find all parquet files matching the pattern
    parquet_files = sorted(input_dir.glob(f"{prefix}*.parquet"))

    if not parquet_files:
        console.print(
            f"[red]No parquet files found matching pattern '{prefix}*.parquet' in {input_dir}[/red]"
        )
        return

    # Filter valid files first
    valid_files = []
    for file_path in parquet_files:
        if not file_path.exists():
            continue

        model, layer = parse_model_filename(file_path.name, prefix)
        if model is None or layer is None:
            continue

        valid_files.append((file_path, model, layer))

    if not valid_files:
        console.print("[red]No valid files to process[/red]")
        return

    # Process each file with progress bar
    results = []
    best_overall_score = None
    best_overall_model = None
    best_overall_layer = None
    best_per_model = {}  # Track best score per model
    metrics_by_file = {}  # Store metrics for each (model, layer) combination
    best_report = None  # Track the absolute best report for UMAP visualization

    total_tasks = len(valid_files) * n_sample
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=4,  # Refresh rate for better PyCharm compatibility
    ) as progress:
        task = progress.add_task(
            f"[cyan]Processing {len(valid_files)} files × {n_sample} samples...",
            total=total_tasks,
        )

        for file_path, model, layer in valid_files:
            # Accumulate metrics across runs for this file
            file_metrics = []
            file_reports = []
            all_metrics_dfs = []  # Accumulate metrics across samples for aggregation

            optimization_config = ClusteringOptimizationConfig(
                min_class_size=min_class_size,
                max_scale=max_scale,
                rns=RandomState(seed=seed),
                frac=frac,
                head=head,
                batch_size=batch_size,
                optimization_method="mean",
            )

            for sample_idx in range(n_sample):
                # Update progress bar description with current status
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

                report = estimate_model_clustering(
                    file_path=file_path,
                    transform_config=transform_config,
                    optimization_config=optimization_config,
                    selected_labels=selected_labels,
                    all_metrics_dfs=all_metrics_dfs,
                )

                if report is not None:
                    # Collect metrics for plotting (function already updated all_metrics_dfs)
                    file_metrics.append(report.metrics_df)
                    file_reports.append(report)

                    # Ensure all_metrics_dfs is initialized for next iteration
                    if all_metrics_dfs is None:
                        all_metrics_dfs = [report.metrics_df]

                    # Track the absolute best report (highest score)
                    if (
                        best_report is None
                        or report.best_score > best_report.best_score
                    ):
                        best_report = report

                progress.advance(task)

            # Process accumulated results for this file
            if file_reports:
                # Store metrics for plotting
                metrics_by_file[(model, layer)] = file_metrics

                # Aggregate results across runs (take mean and std)
                best_sizes = [r.best_size for r in file_reports]
                best_scores = [r.best_score for r in file_reports]
                number_properties_list = [r.number_properties for r in file_reports]
                hungarian_accuracies = [
                    r.hungarian_accuracy
                    for r in file_reports
                    if r.hungarian_accuracy is not None
                ]

                avg_best_size = np.mean(best_sizes)
                std_best_size = np.std(best_sizes) if len(best_sizes) > 1 else 0.0

                avg_best_score = np.mean(best_scores)
                std_best_score = np.std(best_scores) if len(best_scores) > 1 else 0.0

                avg_number_properties = np.mean(number_properties_list)
                std_number_properties = (
                    np.std(number_properties_list)
                    if len(number_properties_list) > 1
                    else 0.0
                )

                avg_hungarian_accuracy = (
                    np.mean(hungarian_accuracies) if hungarian_accuracies else None
                )
                std_hungarian_accuracy = (
                    np.std(hungarian_accuracies)
                    if len(hungarian_accuracies) > 1
                    else 0.0
                )

                results.append(
                    {
                        "model": model,
                        "layer": layer,
                        "best_size": avg_best_size,
                        "best_size_std": std_best_size,
                        "number_properties": avg_number_properties,
                        "number_properties_std": std_number_properties,
                        "best_score": avg_best_score,
                        "best_score_std": std_best_score,
                        "hungarian_accuracy": avg_hungarian_accuracy,
                        "hungarian_accuracy_std": std_hungarian_accuracy,
                    }
                )

                # Update best overall (using mean score)
                if best_overall_score is None or avg_best_score > best_overall_score:
                    best_overall_score = avg_best_score
                    best_overall_model = model
                    best_overall_layer = layer

                # Update best per model
                if (
                    model not in best_per_model
                    or avg_best_score > best_per_model[model]
                ):
                    best_per_model[model] = avg_best_score

                if len(file_metrics) > 1:
                    # Use lineplot with error bars for multiple samples
                    plot_metrics_with_error_bars(
                        file_metrics, output_dir / f"{model}_{layer}_error_bars.png"
                    )
                else:
                    # Use original plot for single sample
                    plot_metrics(file_metrics[0], output_dir / f"{model}_{layer}.png")

    # Create results dataframe
    if not results:
        console.print("[red]No results to save[/red]")
        return

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(["model", "layer"])

    # Save results
    output_path = output_dir / "results.csv"
    df_results.to_csv(output_path, index=False)

    # Create heatmaps
    heatmap_path = output_dir / "model.perf.heatmap.png"
    plot_heatmap(
        df_results, heatmap_path, metric="best_score", metric_label="Best Score"
    )

    # Create Hungarian accuracy heatmap if available
    if (
        "hungarian_accuracy" in df_results.columns
        and df_results["hungarian_accuracy"].notna().any()
    ):
        hungarian_heatmap_path = output_dir / "model.hungarian_accuracy.heatmap.png"
        plot_heatmap(
            df_results,
            hungarian_heatmap_path,
            metric="hungarian_accuracy",
            metric_label="Hungarian Accuracy",
        )

    console.print("\n[bold green]Results Summary[/bold green]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Layer", style="yellow")
    table.add_column("Best Size", justify="right", style="green")
    table.add_column("Properties", justify="right", style="magenta")
    table.add_column("Best Score", justify="right", style="blue")

    for _, row in df_results.iterrows():
        # Format with std if n_sample > 1
        if n_sample > 1:
            best_size_str = f"{int(row['best_size'])} ± {row['best_size_std']:.1f}"
            properties_str = (
                f"{int(row['number_properties'])} ± {row['number_properties_std']:.1f}"
            )
            best_score_str = f"{row['best_score']:.3f} ± {row['best_score_std']:.3f}"
        else:
            best_size_str = str(int(row["best_size"]))
            properties_str = str(int(row["number_properties"]))
            best_score_str = f"{row['best_score']:.3f}"

        table.add_row(
            str(row["model"]),
            str(row["layer"]),
            best_size_str,
            properties_str,
            best_score_str,
        )

    console.print(table)
    console.print(f"\n[green]✓[/green] Results saved to: [cyan]{output_path}[/cyan]")
    console.print(f"[green]✓[/green] Heatmap saved to: [cyan]{heatmap_path}[/cyan]")

    if best_overall_score is not None:
        console.print(
            f"\n[bold green]Best overall score: {best_overall_score:.3f}[/bold green] "
            f"([cyan]{best_overall_model}[/cyan]/[yellow]{best_overall_layer}[/yellow])"
        )

    # Generate UMAP visualization for the best model
    if best_report is not None:
        umap_viz_path = output_dir / "umap_best.html"
        console.print(
            "[green]✓[/green] Generating UMAP visualization for best model..."
        )
        plot_umap_viz(best_report.df, output_path=str(umap_viz_path))
        console.print(
            f"[green]✓[/green] UMAP visualization saved to: [cyan]{umap_viz_path}[/cyan]"
        )


if __name__ == "__main__":
    main()
