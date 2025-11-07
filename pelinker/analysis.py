import numpy as np
import pandas as pd
import torch
import umap
from matplotlib import pyplot as plt
from plotly import express as px, graph_objects as go
from scipy.optimize import differential_evolution
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from torch.nn import functional as F


def compute_optimal_min_cluster_size(dfr2, umap_columns):
    def objective(size):
        size = int(size[0])
        clusterer = HDBSCAN(min_cluster_size=size, metric="cosine")
        labels = clusterer.fit_predict(dfr2[umap_columns])

        if len(set(labels)) <= 1 or len(set(labels)) == len(labels):
            return -1  # invalid clustering → low score

        score = silhouette_score(dfr2[umap_columns], labels)
        print(f"size = {size}, score = {score:.3f}")
        return -score

    bounds = [(5, 200)]  # min_cluster_size range

    result = differential_evolution(objective, bounds, maxiter=20, seed=42)
    best_size = int(round(result.x[0]))

    clusterer = HDBSCAN(min_cluster_size=best_size, metric="cosine")
    labels = clusterer.fit_predict(dfr2[umap_columns])
    dfr2["class"] = pd.DataFrame(labels, columns=["class"], index=dfr2.index)
    return best_size, dfr2


def plot_metrics(df: pd.DataFrame, fname="figs/cl.silhouette.png"):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("min_cluster_size")
    ax1.set_ylabel("Silhouette score", color=color1)
    ax1.plot(
        df["min_cluster_size"],
        df["silhouette"],
        marker="o",
        color=color1,
        label="Silhouette",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Add a second y-axis for icm
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_yscale("log")
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
    ax3.set_yscale("log")
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


def plot_umap_viz(df):
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

    fig.write_html("umap.html")


def umap_it(df, umap_dim=15):
    embedding_vectors = np.stack(df["embed"].values)
    reduced = umap.UMAP(n_components=umap_dim, metric="cosine").fit_transform(
        embedding_vectors
    )
    df_reduced = pd.DataFrame(
        reduced, index=df.index, columns=[f"u_{j:02d}" for j in range(umap_dim)]
    )
    reduced_viz = umap.UMAP(n_components=3, metric="cosine").fit_transform(reduced)
    df_reduced_viz = pd.DataFrame(
        reduced_viz, index=df.index, columns=[f"uviz_{j:02d}" for j in range(3)]
    )
    df = pd.concat([df, df_reduced, df_reduced_viz], axis=1)
    return df


def cosine_similarity_std(tensor):
    """
    Calculate the standard deviation of pairwise cosine similarities
    for a tensor of shape (n_b, dim_emb).

    Args:
        tensor: torch.Tensor of shape (n_b, dim_emb)

    Returns:
        torch.Tensor: scalar tensor containing the standard deviation
    """

    # Normalize the embeddings to unit vectors
    normalized = F.normalize(tensor, p=2, dim=1)

    # Compute pairwise cosine similarities
    cos_sim_matrix = torch.mm(normalized, normalized.t())

    # Get upper triangular part (excluding diagonal) to avoid duplicates and self-similarity
    triu_indices = torch.triu_indices(
        cos_sim_matrix.size(0), cos_sim_matrix.size(1), offset=1
    )
    cos_similarities = cos_sim_matrix[triu_indices[0], triu_indices[1]]

    # Calculate standard deviation
    std_dev = torch.std(cos_similarities)

    return std_dev
