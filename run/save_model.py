# pylint: disable=E1120

import click
import faiss
import pandas as pd
import torch

from pelinker.model import LinkerModel
from pelinker.util import (
    text_to_tokens_embeddings,
    tt_aggregate_normalize,
    load_models,
)
from importlib.resources import files
from pelinker.preprocess import pre_process_properties
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
    is_flag=True,
    help="use a superposition of label and description embeddings, where available",
)
@click.option(
    "--layers",
    type=click.INT,
    default=[-6, -5, -4, -3, -2, -1],
    multiple=True,
    help="layers to consider",
)
def run(model_type, layers, superposition):
    layers = list(layers)
    suffix = ".superposition" if superposition else ""

    df0 = pd.read_csv("data/derived/properties.synthesis.csv")

    report = pre_process_properties(df0)
    labels = report.pop("labels")
    descriptions = report.pop("descriptions")
    ixlabel_ixdesc = report.pop("ixlabel_ixdesc")
    properties = report.pop("properties")

    tokenizer, model = load_models(model_type)

    tt_labels_layered, labels_spans = text_to_tokens_embeddings(
        labels, tokenizer, model
    )

    tt_descs_layered, desc_spans = text_to_tokens_embeddings(
        descriptions,
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
        tt_basis = tt_basis / tt_basis.norm(dim=1).unsqueeze(1)
    else:
        tt_basis = tt_labels

    index = faiss.IndexFlatIP(tt_basis.shape[1])
    index.add(tt_basis)
    lm = LinkerModel(index=index, vocabulary=properties, ls=layers)

    file_path = files("pelinker.store").joinpath(
        f"pelinker.model.{model_type}.{layers_str}{suffix}.gz"
    )
    joblib.dump(lm, file_path, compress=3)


if __name__ == "__main__":
    run()
