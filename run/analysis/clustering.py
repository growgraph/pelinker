import plotly.graph_objects as go
import click
import pathlib
import numpy as np

import plotly.express as px
import pandas as pd
import pyarrow.feather as pf
from numpy.random import RandomState
import umap
from sklearn.cluster import HDBSCAN


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
        vc = batch[batch.columns[0]].value_counts()
        sample = batch.sample(frac=frac, random_state=rns)
        agg += [sample]
        print(vc)
        print(f"Batch {i+1}: {len(batch)} rows, {len(batch.columns)} columns")
        if head is not None and i > head - 1:
            break

    dfr = pd.concat(agg)
    dfr2 = umap_it(dfr, umap_dim=umap_dim)
    clusterer = HDBSCAN(min_cluster_size=int(dfr2.shape[0] / 100), metric="cosine")
    labels = clusterer.fit_predict(dfr2[dfr2.columns[-umap_dim:]])
    dfr2["class"] = pd.DataFrame(labels, columns=["class"], index=dfr2.index)
    print(dfr2.groupby("class").apply(lambda x: x["property"].value_counts()))
    print(dfr2.groupby("property").apply(lambda x: x["class"].value_counts()))
    print("end")
    plot_umap_viz(dfr2)


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


if __name__ == "__main__":
    main()
