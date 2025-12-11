import pathlib

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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
