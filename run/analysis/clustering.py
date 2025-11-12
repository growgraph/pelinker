import click
import pathlib
import re
import numpy as np
from dataclasses import dataclass

import torch

import pandas as pd
from numpy.random import RandomState
from sklearn.cluster import HDBSCAN

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
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
    compute_optimal_min_cluster_size,
    plot_metrics,
    umap_it,
    cosine_similarity_std,
    # plot_umap_viz
)
from pelinker.io import read_batches


@dataclass
class ClusteringReport:
    """Report containing clustering analysis results."""

    best_size: int
    best_score: float
    number_properties: int
    metrics_df: (
        pd.DataFrame
    )  # DataFrame with columns: min_cluster_size, icm, n_clusters, silhouette
    dfr2: pd.DataFrame  # DataFrame with UMAP embeddings and cluster labels


def estimate_model(
    file_path: pathlib.Path,
    rns: RandomState,
    umap_dim: int = 15,
    min_class_size: int = 20,
    tol: float = 0.05,
    frac: float = 0.1,
    head: int | None = None,
    batch_size: int = 1000,
) -> ClusteringReport | None:
    """
    Estimate optimal cluster size for a single model/layer file.

    Args:
        file_path: Path to parquet file
        rns: RandomState object for reproducible sampling
        umap_dim: UMAP dimensionality
        min_class_size: Minimum class size for filtering
        tol: Tolerance for optimization
        frac: Fraction of dataset to sample
        head: Number of batches to take (None for all)
        batch_size: Batch size for reading

    Returns:
        ClusteringReport or None: Report containing clustering results, or None if processing failed
        ClusteringReport or None: Report containing clustering results, or None if processing failed
    """
    # Simple check: if file doesn't exist (handles broken symlinks), skip it
    if not file_path.exists():
        return None
    agg = []

    try:
        for i, batch in enumerate(
            read_batches(file_path.as_posix(), batch_size=batch_size)
        ):
            sample = batch.sample(frac=frac, random_state=rns)
            agg += [sample]
            if head is not None and i >= head - 1:
                break
    except Exception:
        return None

    if not agg:
        return None

    dfr = pd.concat(agg)

    umap_columns = [f"u_{j:02d}" for j in range(umap_dim)]

    # trim rare mentions
    mention_count = dfr["property"].value_counts()
    low_count_properties = mention_count[
        ~(mention_count >= min_class_size)
    ].index.to_list()

    dfr = dfr.loc[~dfr["property"].isin(low_count_properties)].copy()

    if len(dfr) == 0:
        return None

    # Get number of unique properties before UMAP
    number_properties = dfr["property"].nunique()

    dfr2 = umap_it(dfr, umap_dim=umap_dim)

    sizes = list(np.arange(int(0.5 * min_class_size), int(5 * min_class_size), 5)) + [
        int(2 * number_properties)
    ]

    metrics = []
    for size in sizes:
        clusterer = HDBSCAN(min_cluster_size=size, metric="cosine")
        labels = clusterer.fit_predict(dfr2[umap_columns])
        dfr2_temp = dfr2.copy()
        dfr2_temp["class"] = pd.DataFrame(
            labels, columns=["class"], index=dfr2_temp.index
        )

        ic = []
        for ix, group in dfr2_temp.groupby("class"):
            if ix == -1:  # Skip noise points
                continue
            tgroup = torch.from_numpy(group[umap_columns].values)
            st = cosine_similarity_std(tgroup)
            ic += [st]

        icm = np.mean(ic) if ic else np.nan

        # Only compute silhouette score if we have valid clusters (at least 2 clusters, excluding noise)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters > 2:
            score = silhouette_score(dfr2[umap_columns], labels)
            metrics += [(size, icm, n_clusters, score)]

    dfm = pd.DataFrame(
        metrics, columns=["min_cluster_size", "icm", "n_clusters", "silhouette"]
    )

    # Find optimal cluster size
    bounds = [(int(0.5 * min_class_size), int(5 * min_class_size))]
    best_size, best_score, dfr2_final = compute_optimal_min_cluster_size(
        dfr2, umap_columns, tol=tol, bounds=bounds
    )

    return ClusteringReport(
        best_size=best_size,
        best_score=best_score,
        number_properties=number_properties,
        metrics_df=dfm,
        dfr2=dfr2_final,
    )


def parse_filename(filename: str):
    """
    Parse filename like 'res_bert_1.parquet' to extract model and layer.

    Args:
        filename: Filename to parse

    Returns:
        tuple: (model, layer) or (None, None) if pattern doesn't match
    """
    # Pattern: res_<model>_<layer>.parquet
    pattern = r"res_([^_]+)_(\d+)\.parquet"
    match = re.match(pattern, filename)
    if match:
        model = match.group(1)
        layer = int(match.group(2))
        return model, layer
    return None, None


def plot_metrics_with_error_bars(
    metrics_list: list[pd.DataFrame],
    output_path: pathlib.Path,
):
    """
    Plot metrics across multiple runs with error bars using seaborn lineplot.

    Args:
        metrics_list: List of DataFrames, each with columns: min_cluster_size, icm, n_clusters, silhouette
        output_path: Path to save the figure
    """
    # Combine all metrics DataFrames, adding a run_id column
    combined_metrics = []
    for run_id, df in enumerate(metrics_list):
        df_copy = df.copy()
        df_copy["run_id"] = run_id
        combined_metrics.append(df_copy)

    df_combined = pd.concat(combined_metrics, ignore_index=True)

    # Filter out trivial points where n_clusters <= 1
    df_combined = df_combined[df_combined["n_clusters"] > 1].copy()

    if len(df_combined) == 0:
        print(
            f"Warning: No valid data points after filtering (n_clusters > 1) for {output_path}"
        )
        return

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color palette for different plots
    colors = ["#2E86AB", "#A23B72", "#F18F01"]  # Blue, Purple, Orange

    # Plot silhouette score with error bars
    sns.lineplot(
        data=df_combined,
        x="min_cluster_size",
        y="silhouette",
        ax=axes[0],
        errorbar="sd",  # Standard deviation error bars
        marker="o",
        color=colors[0],
        linewidth=2,
        markersize=8,
        err_kws={"alpha": 0.3, "linewidth": 1.5},
    )
    axes[0].set_xlabel("min_cluster_size", fontsize=12, fontweight="bold")
    axes[0].set_ylabel(
        "Silhouette score", fontsize=12, fontweight="bold", color=colors[0]
    )
    axes[0].set_title(
        "Silhouette Score vs. min_cluster_size", fontsize=13, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].tick_params(axis="y", labelcolor=colors[0])
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # # Plot ICM with error bars (log scale)
    # sns.lineplot(
    #     data=df_combined,
    #     x="min_cluster_size",
    #     y="icm",
    #     ax=axes[1],
    #     errorbar="sd",
    #     marker="s",
    #     color=colors[1],
    #     linewidth=2,
    #     markersize=8,
    #     err_kws={"alpha": 0.3, "linewidth": 1.5},
    # )
    # # axes[1].set_yscale("log")
    # axes[1].set_xlabel("min_cluster_size", fontsize=12, fontweight="bold")
    # axes[1].set_ylabel(
    #     "ICM (log scale)", fontsize=12, fontweight="bold", color=colors[1]
    # )
    # axes[1].set_title("ICM vs. min_cluster_size", fontsize=13, fontweight="bold")
    # axes[1].grid(True, alpha=0.3, linestyle="--")
    # axes[1].tick_params(axis="y", labelcolor=colors[1])
    # axes[1].spines["top"].set_visible(False)
    # axes[1].spines["right"].set_visible(False)

    # Plot n_clusters with error bars (log scale)
    sns.lineplot(
        data=df_combined,
        x="min_cluster_size",
        y="n_clusters",
        ax=axes[1],
        errorbar="sd",
        marker="^",
        color=colors[1],
        linewidth=2,
        markersize=8,
        err_kws={"alpha": 0.3, "linewidth": 1.5},
    )
    # axes[2].set_yscale("log")
    axes[1].set_xlabel("min_cluster_size", fontsize=12, fontweight="bold")
    axes[1].set_ylabel(
        "n_clusters (log scale)", fontsize=12, fontweight="bold", color=colors[2]
    )
    axes[1].set_title(
        "Number of Clusters vs. min_cluster_size", fontsize=13, fontweight="bold"
    )
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].tick_params(axis="y", labelcolor=colors[2])
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(df_results: pd.DataFrame, output_path: pathlib.Path):
    """
    Create a heatmap with model (rows) and layer (columns).
    Color represents best_score, text shows best_size.

    Args:
        df_results: DataFrame with columns: model, layer, best_size, best_score
        output_path: Path to save the heatmap figure
    """
    # Create pivot tables
    score_pivot = df_results.pivot(index="model", columns="layer", values="best_score")
    size_pivot = df_results.pivot(index="model", columns="layer", values="best_size")

    # Create figure
    fig, ax = plt.subplots(
        figsize=(
            max(8, len(score_pivot.columns) * 0.8),
            max(6, len(score_pivot.index) * 0.6),
        )
    )

    # Create heatmap with best_score as color
    sns.heatmap(
        score_pivot,
        annot=False,  # We'll add custom annotations
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Best Score"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )

    # Add best_size as text annotations
    # Calculate mean score for text color threshold
    valid_scores = score_pivot.values[~pd.isna(score_pivot.values)]
    mean_score = valid_scores.mean() if len(valid_scores) > 0 else 0

    for i in range(len(score_pivot.index)):
        for j in range(len(score_pivot.columns)):
            score_val = score_pivot.iloc[i, j]
            size_val = size_pivot.iloc[i, j]

            if not pd.isna(score_val) and not pd.isna(size_val):
                # Use white text for darker cells (lower scores), black for lighter cells
                text_color = "white" if score_val < mean_score else "black"
                # Add text annotation with best_size
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{int(size_val)}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=9,
                )

    ax.set_title("Clustering Results: Best Score (color) and Best Size (text)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Directory containing parquet files",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Output CSV path for results (default: input_dir/results.csv)",
)
@click.option(
    "--heatmap-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Output path for heatmap figure (default: input_dir/heatmap.png)",
)
@click.option(
    "--umap-dim",
    type=click.INT,
    default=15,
    help="UMAP dimensionality",
)
@click.option(
    "--min-class-size",
    type=click.INT,
    default=20,
    help="Minimum class size for filtering",
)
@click.option(
    "--tol",
    type=click.FLOAT,
    default=0.05,
    help="Tolerance for optimization",
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
    "--n-sample",
    type=click.INT,
    default=1,
    help="Number of samples/runs per (model, layer) combination",
)
def main(
    input_dir: pathlib.Path,
    output_path: pathlib.Path,
    heatmap_path: pathlib.Path,
    umap_dim: int,
    min_class_size: int,
    tol: float,
    seed: int,
    frac: float,
    head: int,
    batch_size: int,
    n_sample: int,
):
    """
    Process multiple parquet files and compute optimal cluster sizes.

    Files should follow the pattern: res_<model>_<layer>.parquet
    """
    # Configure console to work better in PyCharm and other IDEs
    # Use legacy_windows=False and force_terminal to ensure progress bars work
    console = Console(force_terminal=True, width=120, legacy_windows=False)
    input_dir = input_dir.expanduser()

    # Plot metrics with error bars for each (model, layer) combination
    figs_dir = pathlib.Path("figs")
    figs_dir.mkdir(exist_ok=True)

    # Find all parquet files matching the pattern
    parquet_files = sorted(input_dir.glob("res_*.parquet"))

    if not parquet_files:
        console.print(
            f"[red]No parquet files found matching pattern 'res_*.parquet' in {input_dir}[/red]"
        )
        return

    # Filter valid files first
    valid_files = []
    for file_path in parquet_files:
        if not file_path.exists():
            continue

        model, layer = parse_filename(file_path.name)
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

            rns = RandomState(seed=seed)

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

                report = estimate_model(
                    file_path=file_path,
                    rns=rns,
                    umap_dim=umap_dim,
                    min_class_size=min_class_size,
                    tol=tol,
                    frac=frac,
                    head=head,
                    batch_size=batch_size,
                )

                if report is not None:
                    file_metrics.append(report.metrics_df)
                    file_reports.append(report)

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
                        file_metrics, figs_dir / f"{model}_{layer}_error_bars.png"
                    )
                else:
                    # Use original plot for single sample
                    plot_metrics(file_metrics[0], figs_dir / f"{model}_{layer}.png")

    # Create results dataframe
    if not results:
        console.print("[red]No results to save[/red]")
        return

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(["model", "layer"])

    # Save results
    if output_path is None:
        output_path = figs_dir / "results.csv"
    else:
        output_path = output_path.expanduser()

    df_results.to_csv(output_path, index=False)

    # Create heatmap
    if heatmap_path is None:
        heatmap_path = figs_dir / "heatmap.png"
    else:
        heatmap_path = heatmap_path.expanduser()

    plot_heatmap(df_results, heatmap_path)

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
        from pelinker.analysis import plot_umap_viz

        umap_viz_path = figs_dir / "umap_best.html"
        console.print(
            "[green]✓[/green] Generating UMAP visualization for best model..."
        )
        plot_umap_viz(best_report.dfr2, output_path=str(umap_viz_path))
        console.print(
            f"[green]✓[/green] UMAP visualization saved to: [cyan]{umap_viz_path}[/cyan]"
        )


if __name__ == "__main__":
    main()
