import plotly.graph_objects as go
import click
import pathlib
import numpy as np

import torch
import torch.nn.functional as F


import plotly.express as px
import pandas as pd
import pyarrow.feather as pf
from numpy.random import RandomState
import umap
from sklearn.cluster import HDBSCAN
from scipy.optimize import differential_evolution

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


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


def read_feather_mmap_batches(file_path, batch_size=1000):
    """
    Read feather file using memory mapping for large files.
    """
    table = pf.read_table(file_path)
    total_rows = len(table)

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_table = table.slice(start_idx, end_idx - start_idx)
        yield batch_table.to_pandas()


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


@click.command()
@click.option(
    "--input-path",
    type=click.Path(path_type=pathlib.Path),
    default="~/data/pelinker/mag_data/bio_2M_res.feather",
    help="input df",
)
@click.option(
    "--seed",
    type=click.INT,
    default=13,
    help="seed to sample the dataset",
)
@click.option(
    "--head",
    type=click.INT,
    default=100,
    help="number of batches to take",
)
@click.option(
    "--frac",
    type=click.FLOAT,
    default=0.1,
    help="fraction of dataset to consider",
)
def main(input_path: pathlib.Path, seed, frac, head, umap_dim=15, batch_size=1000):
    input_path = input_path.expanduser()
    rns = RandomState(seed=seed)
    agg = []
    for i, batch in enumerate(
        read_feather_mmap_batches(input_path.as_posix(), batch_size=batch_size)
    ):
        # vc = batch[batch.columns[0]].value_counts()
        sample = batch.sample(frac=frac, random_state=rns)
        agg += [sample]
        print(f"Batch {i+1}: {len(batch)} rows, {len(batch.columns)} columns")
        if head is not None and i > head - 1:
            break

    dfr = pd.concat(agg)

    umap_columns = [f"u_{j:02d}" for j in range(umap_dim)]
    # trim rare mentions
    mention_count = dfr["property"].value_counts()
    low_count_mentions = mention_count[~(mention_count > 50)].index.to_list()
    dfr = dfr.loc[~dfr["property"].isin(low_count_mentions)].copy()

    dfr2 = umap_it(dfr, umap_dim=umap_dim)

    sizes = list(np.arange(20, 60, 5)) + [60, 75, 100, 150, 200]

    metrics = []
    for size in sizes:
        clusterer = HDBSCAN(min_cluster_size=size, metric="cosine")
        labels = clusterer.fit_predict(dfr2[umap_columns])
        dfr2["class"] = pd.DataFrame(labels, columns=["class"], index=dfr2.index)

        ic = []
        for ix, group in dfr2.groupby("class"):
            tgroup = torch.from_numpy(group[umap_columns].values)
            st = cosine_similarity_std(tgroup)
            ic += [st]

        icm = np.mean(ic)

        if len(set(labels)) > 1:
            score = silhouette_score(dfr2[umap_columns], labels)
            print(
                f"Min c size: {size}, silhouette score: {score:.3f}, icm = {icm:.3g}, nclusters = {len(set(labels))}"
            )
            metrics += [(size, icm, len(set(labels)), score)]

    dfm = pd.DataFrame(
        metrics, columns=["min_cluster_size", "n_clusters", "icm", "silhouette"]
    )

    plot_metrics(dfm)

    best_size, dfr2 = compute_optimal_min_cluster_size(dfr2, umap_columns)

    print(f"Best min_cluster_size: {best_size}. n_cluster = {dfr2['class'].nunique()}")

    print(dfr2.groupby("class").apply(lambda x: x["property"].value_counts()))
    print(dfr2.groupby("property").apply(lambda x: x["class"].value_counts()))
    print("end")

    # plot_umap_viz(dfr2)


if __name__ == "__main__":
    main()
