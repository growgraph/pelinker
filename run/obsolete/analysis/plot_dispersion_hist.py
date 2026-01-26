# pylint: disable=E1120

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def run():
    df0 = pd.read_csv("./data/derived/491_within_n_across_disps.csv")
    df0 = df0.rename(columns={"within_verbs": "std"})
    fig_path = "./figs"

    sns.set_style("whitegrid")
    _ = sns.displot(
        data=df0,
        x="std",
        stat="density",
        common_norm=False,
        bins=np.arange(0.1, 0.7, 0.02),
    )
    path = f"{fig_path}/std.verb.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    run()
