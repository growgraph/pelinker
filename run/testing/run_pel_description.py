# pylint: disable=E1120

import click
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import seaborn as sns

from pelinker.util import (
    text_to_tokens_embeddings,
    tt_aggregate_normalize,
    load_models,
    compute_distance_ref,
)
from pelinker.preprocess import pre_process_properties


@click.command()
@click.option(
    "--model-type",
    type=click.STRING,
    default=["scibert"],
    multiple=True,
    help="run over BERT flavours",
)
@click.option(
    "--superposition",
    type=click.BOOL,
    default=False,
    is_flag=True,
    help="use a superposition of label and description embeddings, where available",
)
def run(model_type, superposition):
    fig_path = "./figs"
    report_path = "./reports"

    suffix = "_superposition" if superposition else ""

    df0 = pd.read_csv("data/derived/properties.synthesis.csv")

    report = pre_process_properties(df0)
    labels = report.pop("labels")
    descriptions = report.pop("descriptions")
    ixlabel_ixdesc = report.pop("ixlabel_ixdesc")
    property_from_desc = report.pop("property_from_desc")
    id_ixlabel_map = report.pop("id_ixlabel_map")

    layers = (
        [[-x] for x in range(1, 10)]
        + [[-x for x in range(1, 4)]]
        + [[-x for x in range(1, 8)]]
        + [[-1, -2, -8, -9]]
    )

    metrics = []
    df_agg = []

    for mt in model_type:
        tokenizer, model = load_models(mt)

        tt_labels_layered, labels_spans = text_to_tokens_embeddings(
            labels, tokenizer, model
        )

        tt_descs_layered, desc_spans = text_to_tokens_embeddings(
            descriptions,
            tokenizer,
            model,
        )

        for ls in layers:
            tt_labels = tt_aggregate_normalize(tt_labels_layered, ls)
            tt_descs = tt_aggregate_normalize(tt_descs_layered, ls)

            if superposition:
                tt_basis = []
                for j, tt in enumerate(tt_labels):
                    if j in ixlabel_ixdesc:
                        tt_basis += [tt + tt_descs[ixlabel_ixdesc[j]]]
                    else:
                        tt_basis += [tt]

                tt_basis = torch.stack(tt_basis)
                tt_basis = tt_basis / tt_basis.norm(dim=1).unsqueeze(1)
            else:
                tt_basis = tt_labels

            index = faiss.IndexFlatIP(tt_basis.shape[1])
            index.add(tt_basis)

            nb_nn = min([10, tt_basis.shape[0]])
            layers_str = "_".join([str(x) for x in ls])

            m0, dfa = compute_distance_ref(
                index,
                tt_descs,
                nb_nn,
                np.array([id_ixlabel_map[p] for p in property_from_desc]),
            )
            dfa["model_type"] = mt
            dfa["layers"] = layers_str

            df_agg += [dfa]
            metrics += [(mt, layers_str, *m0)]

    df0 = pd.concat(df_agg)
    _ = sns.displot(
        data=df0,
        x="dist",
        hue="position",
        stat="density",
        common_norm=False,
        col="layers",
        row="model_type",
        bins=np.arange(0.2, 1.1, 0.05),
        aspect=0.7,
    )

    path = f"{fig_path}/desc2label_dist{suffix}.png"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    metrics_df = pd.DataFrame(metrics, columns=["model_type", "layers", "dist", "acc"])
    metrics_df.to_csv(f"{report_path}/metrics_desc2label{suffix}.csv")

    min_len = min(metrics_df["layers"].apply(lambda x: len(x)))
    metrics_df2 = metrics_df[
        metrics_df["layers"].apply(lambda x: len(x) == min_len)
    ].copy()
    metrics_df2["layers"] = metrics_df2["layers"].astype(int)

    sns.lineplot(metrics_df2, hue="model_type", x="layers", y="acc")
    path = f"{fig_path}/desc2label_accuracy{suffix}.png"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()

    sns.lineplot(metrics_df2, hue="model_type", x="layers", y="dist")
    path = f"{fig_path}/desc2label_dist_modes{suffix}.png"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()

    print(metrics_df)


if __name__ == "__main__":
    run()
