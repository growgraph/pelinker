# pylint: disable=E1120

import click
import faiss
import pandas as pd
import torch


from pelinker.util import (
    text_to_tokens_embeddings,
    tt_aggregate_normalize,
    load_models,
)
from importlib.resources import files
import joblib


@click.command()
@click.option(
    "--model-type",
    type=click.STRING,
    default="pubmedbert",
    help="run over BERT flavours",
)
@click.option(
    "--superposition",
    type=click.BOOL,
    default=False,
    help="use a superposition of label and description embeddings, where available",
)
@click.option(
    "--layers",
    type=click.INT,
    default=[-1],
    multiple=True,
    help="layers to consider",
)
def run(model_type, layers, superposition):
    suffix = ".superposition" if superposition else ""

    df0 = pd.read_csv("data/derived/properties.synthesis.csv")

    df = df0.copy()

    mask_desc_exclude = df["description"].isnull() | df["description"].apply(
        lambda x: "inverse" in x.lower() if isinstance(x, str) else True
    )

    ixlabel_id_dict = df["property"].reset_index(drop=True).to_dict()
    id_ixlabel_dict = {v: k for k, v in ixlabel_id_dict.items()}

    ixdesc_id = list(
        df.loc[~mask_desc_exclude, "property"].reset_index(drop=True).items()
    )

    ixlabel_ixdesc = {id_ixlabel_dict[label]: ixd for ixd, label in ixdesc_id}

    df.loc[~mask_desc_exclude, "property"].reset_index(drop=True)

    tokenizer, model = load_models(model_type)

    tt_labels_layered, labels_spans = text_to_tokens_embeddings(
        df["label"].values.tolist(), tokenizer, model
    )

    tt_descs_layered, desc_spans = text_to_tokens_embeddings(
        df.loc[~mask_desc_exclude, "description"].values.tolist(),
        tokenizer,
        model,
    )
    layers_str = "_".join([str(x) for x in layers])
    tt_labels = tt_aggregate_normalize(tt_labels_layered, layers)
    tt_descs = tt_aggregate_normalize(tt_descs_layered, layers)

    if superposition:
        tt_basis = []
        for j, tt in enumerate(tt_labels):
            if j in ixlabel_ixdesc:
                tt_basis += [tt + tt_descs[ixlabel_ixdesc[j]]]
            else:
                tt_basis += [tt]

        tt_basis = torch.stack(tt_basis)
        tt_basis = tt_basis / tt_labels.norm(1).unsqueeze(1)
    else:
        tt_basis = tt_labels

    index = faiss.IndexFlatIP(tt_basis.shape[1])
    index.add(tt_basis)

    file_path = files("task_manager.models.store").joinpath(
        f"pelinker.model.{model_type}.{layers_str}{suffix}.gz"
    )
    joblib.dump(index, file_path, compress=3)


if __name__ == "__main__":
    run()
