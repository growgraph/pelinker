# pylint: disable=E1120

import click
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import torch
import seaborn as sns
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer

from pelinker.util import text_to_tokens_embeddings


@click.command()
@click.option("--text-path", type=click.Path())
@click.option("--model-type", type=click.STRING, default="scibert")
def run(text_path, model_type):
    fig_path = "./figs"

    df0 = pd.read_csv("data/derived/properties.synthesis.csv")

    if model_type == "scibert":
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
    else:
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    df = df0.copy()

    # id_label_dict = dict(df[["property", "label"]].values)
    mask_desc_notnull = df["description"].notnull()
    # id_description_dict = dict(
    #     df.loc[mask_desc_notnull, ["property", "description"]].values
    # )

    ixlabel_id_dict = df["property"].reset_index(drop=True).to_dict()
    id_ixlabel_dict = {v: k for k, v in ixlabel_id_dict.items()}

    ixdesc_id = list(
        df.loc[mask_desc_notnull, "property"].reset_index(drop=True).items()
    )

    df.loc[mask_desc_notnull, "property"].reset_index(drop=True)

    tt_labels_layered, labels_spans = text_to_tokens_embeddings(
        df["label"].values.tolist(), tokenizer, model
    )

    tt_descs_layered, desc_spans = text_to_tokens_embeddings(
        df.loc[mask_desc_notnull, "description"].values.tolist(),
        tokenizer,
        model,
    )

    layers = [[-x] for x in range(1, 8)] + [[-3, -2, -1]]

    metrics = []
    for ls in layers:
        tt_labels = tt_labels_layered[ls].mean(0).mean(dim=1)
        tt_labels = tt_labels / tt_labels.norm(dim=-1).unsqueeze(-1)
        tt_descs = tt_descs_layered[ls].mean(0).mean(dim=1)
        tt_descs = tt_descs / tt_descs.norm(dim=-1).unsqueeze(-1)

        index = faiss.IndexFlatIP(tt_labels.shape[1])
        index.add(tt_labels)

        nb_nn = min([10, tt_labels.shape[0]])
        distance_matrix, nearest_neighbors_matrix = index.search(tt_descs, nb_nn)
        dfd = pd.DataFrame(distance_matrix[:, 0], columns=["dist"])
        dfd["position"] = "top"
        dfd2 = pd.DataFrame(distance_matrix[:, 1:].flatten(), columns=["dist"])
        dfd2["position"] = "1-9"
        dfa = pd.concat([dfd, dfd2])
        acc_score = accuracy_score(
            nearest_neighbors_matrix[:, 0],
            np.array([id_ixlabel_dict[x[1]] for x in ixdesc_id]),
        )

        layers_str = "_".join([str(x) for x in ls])
        dp = sns.displot(
            data=dfa,
            x="dist",
            hue="position",
            stat="density",
            common_norm=False,
            bins=np.arange(0.5, 1.01, 0.01),
        )
        dp.fig.subplots_adjust(top=0.9)
        top_mean = distance_matrix[:, 0].mean()
        below_mean = distance_matrix[:, 1:].flatten().mean()

        top_cand_dist = top_mean - below_mean
        dp.fig.suptitle(
            f"layers: {layers_str}, acc: {acc_score:.3g}, dist: {top_cand_dist:.3g}"
        )

        metrics += [(layers_str, top_cand_dist, acc_score)]

        path = f"{fig_path}/desc2label_cased_{model_type}_{layers_str}.png"
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()

    metrics_df = pd.DataFrame(metrics, columns=["kind", "dist", "acc"])
    print(metrics_df)


if __name__ == "__main__":
    run()
