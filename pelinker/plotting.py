import pathlib

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import express as px, graph_objects as go


def plot_metrics_with_error_bars(
    metrics_list: list[pd.DataFrame],
    output_path: pathlib.Path,
):
    """
    Plot metrics across multiple runs with error bars using seaborn lineplot.

    Args:
        metrics_list: List of DataFrames, each with columns: min_cluster_size, icm, n_clusters, dbcv
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

    # Plot DBCV score with error bars
    sns.lineplot(
        data=df_combined,
        x="min_cluster_size",
        y="dbcv",
        ax=axes[0],
        errorbar="sd",  # Standard deviation error bars
        marker="o",
        color=colors[0],
        linewidth=2,
        markersize=8,
        err_kws={"alpha": 0.3, "linewidth": 1.5},
    )
    axes[0].set_xlabel("min_cluster_size", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("DBCV Score", fontsize=12, fontweight="bold", color=colors[0])
    axes[0].set_title("DBCV Score vs. min_cluster_size", fontsize=13, fontweight="bold")
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
    axes[1].set_xlabel("min_cluster_size", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("n clusters", fontsize=12, fontweight="bold", color=colors[2])
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


def plot_heatmap(
    df_results: pd.DataFrame,
    output_path: pathlib.Path,
    metric: str = "best_score",
    metric_label: str | None = None,
):
    """
    Create a heatmap with model (rows) and layer (columns).
    Color represents the specified metric, text shows best_size and metric name.

    Args:
        df_results: DataFrame with columns: model, layer, best_size, and the metric column
        output_path: Path to save the heatmap figure
        metric: Column name for the metric to display as color (default: "best_score")
        metric_label: Label for the metric (default: uses metric column name)
    """
    if metric_label is None:
        metric_label = metric.replace("_", " ").title()

    # Create pivot tables
    score_pivot = df_results.pivot(index="model", columns="layer", values=metric)
    size_pivot = df_results.pivot(index="model", columns="layer", values="best_size")

    # Create figure
    fig, ax = plt.subplots(
        figsize=(
            max(8, len(score_pivot.columns) * 0.8),
            max(6, len(score_pivot.index) * 0.6),
        )
    )

    # Create heatmap with metric as color
    # Use RdBu_r (Red-Blue reversed) for clear visual distinction: red=high, blue=low
    sns.heatmap(
        score_pivot,
        annot=False,  # We'll add custom annotations
        fmt=".3f",
        cmap="RdBu_r",
        center=None,  # Center colormap at the median for better contrast
        cbar_kws={"label": metric_label, "shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        square=False,
    )

    # Add best_size and metric name as text annotations
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
                # Format metric value based on its magnitude
                if abs(score_val) < 0.01:
                    metric_str = f"{score_val:.2e}"
                elif abs(score_val) < 1:
                    metric_str = f"{score_val:.3f}"
                else:
                    metric_str = f"{score_val:.2f}"
                # Add text annotation with best_size and metric value
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{int(size_val)}\n{metric_str}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=8,
                    linespacing=1.2,
                )

    ax.set_title(f"Clustering Results: {metric_label} (color) and Best Size (text)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_umap_viz(df, output_path="umap.html"):
    df["show_label"] = df["property"]
    show_rate = max(len(df) // 20, 1)
    df.loc[df.index % show_rate != 0, "show_label"] = ""

    # Ensure class is treated as categorical
    df["class"] = df["class"].astype(str)

    # Base scatter plot
    fig = px.scatter_3d(
        df,
        x="uviz_00",
        y="uviz_01",
        z="uviz_02",
        color="class",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        hover_name="property",
        labels={"uviz_00": "Dim 1", "uviz_01": "Dim 2", "uviz_02": "Dim 3"},
    )

    # Add text labels as a separate trace
    df_labels = df[df["show_label"] != ""]
    text_trace = go.Scatter3d(
        x=df_labels["uviz_00"],
        y=df_labels["uviz_01"],
        z=df_labels["uviz_02"],
        mode="text",
        text=df_labels["show_label"],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
        textfont=dict(size=10, color="black"),
    )
    fig.add_trace(text_trace)

    # Update layout
    fig.update_layout(
        title="3D Scatter Plot of Embeddings",
        scene=dict(
            xaxis_title="uviz_00",
            yaxis_title="uviz_01",
            zaxis_title="uviz_02",
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=30),
    )

    fig.write_html(output_path)


def plot_metrics(df: pd.DataFrame, fname):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("min_cluster_size")
    ax1.set_ylabel("DBCV Score", color=color1)
    ax1.plot(
        df["min_cluster_size"],
        df["dbcv"],
        marker="o",
        color=color1,
        label="DBCV",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Add a second y-axis for icm
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    # ax2.set_yscale("log")
    ax2.set_ylabel("ICM", color=color2)
    ax2.plot(
        df["min_cluster_size"],
        df["icm"],
        marker="s",
        linestyle="--",
        color=color2,
        label="ICM",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add a second y-axis for icm
    ax3 = ax1.twinx()
    color2 = "tab:green"
    # ax3.set_yscale("log")
    ax3.set_ylabel("n_clusters", color=color2)
    ax3.plot(
        df["min_cluster_size"],
        df["n_clusters"],
        marker="s",
        linestyle="--",
        color=color2,
        label="ICM",
    )

    ax3.tick_params(axis="y", labelcolor=color2)

    # Titles and layout
    plt.title("Clustering metrics vs. min_cluster_size (HDBSCAN)")
    fig.tight_layout()

    plt.savefig(fname, bbox_inches="tight", dpi=300)
