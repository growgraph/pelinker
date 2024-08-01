# pylint: disable=E1120

import click
import faiss
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from pelinker.util import load_models, encode
from pelinker.preprocess import pre_process_properties


@click.command()
@click.option(
    "--model-type",
    type=click.STRING,
    default=["biobert"],
    multiple=True,
    help="run over BERT flavours",
)
def run(model_type):
    model_type = sorted(model_type)
    fig_path = "./figs"

    df0 = pd.read_csv("data/derived/properties.synthesis.csv")

    report = pre_process_properties(df0)
    labels = report.pop("labels")

    layers = (
        # [[-x] for x in range(1, 6)] +
        # [[-x for x in range(1, 4)]] +
        [[-1]]
        +
        # [[-1, -2, -8, -9]] +
        ["sent"]
    )

    df_agg = []

    for mt in model_type:
        tokenizer, model = load_models(mt, False)
        _, smodel = load_models(mt, True)
        for ls in layers[:]:
            tt_labels = encode(labels, tokenizer, smodel if ls == "sent" else model, ls)
            layers_str = ls if ls == "sent" else "_".join([str(x) for x in ls])

            index = faiss.IndexFlatIP(tt_labels.shape[1])
            nb_nn = min([100, tt_labels.shape[0]])
            index.add(tt_labels)

            distance_matrix, nearest_neighbors_matrix = index.search(tt_labels, nb_nn)
            ds = 1.0 - distance_matrix[:, 1]

            dfa = pd.DataFrame(ds, columns=["delta"])
            dfa["model_type"] = mt
            dfa["layers"] = layers_str
            df_agg += [dfa]
    df0 = pd.concat(df_agg)
    path = f"{fig_path}/discrim.pdf"
    col_wrap = min([4, len(layers)])

    sns.set_style("whitegrid")
    g = sns.displot(
        data=df0,
        x="delta",
        hue="model_type",
        stat="density",
        common_norm=False,
        col="layers",
        col_wrap=col_wrap,
        bins=np.arange(0.0, 0.6, 0.05),
        alpha=0.5,
        facet_kws=dict(legend_out=False),
    )
    sns.move_legend(g, "upper right")
    plt.savefig(path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    run()
