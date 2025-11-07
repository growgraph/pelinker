import click
import pathlib
import numpy as np

import torch

import pandas as pd
from numpy.random import RandomState
from sklearn.cluster import HDBSCAN

from sklearn.metrics import silhouette_score

from pelinker.analysis import (
    compute_optimal_min_cluster_size,
    plot_metrics,
    umap_it,
    cosine_similarity_std,
)
from pelinker.io import read_batches


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
        read_batches(input_path.as_posix(), batch_size=batch_size)
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
