# pylint: disable=E1120

import click
import faiss
import pandas as pd
import torch

from pelinker.model import LinkerModel
from pelinker.util import load_models, encode
from importlib.resources import files
from pelinker.preprocess import pre_process_properties


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
    "--layers-spec",
    type=click.STRING,
    help="`sent` or a string of layers, `1,2,3` would correspond to layers [-1, -2, -3]",
)
def run(model_type, layers_spec, superposition):
    layers = LinkerModel.str2layers(layers_spec)
    sentence = True if layers == "sent" else False

    suffix = ".superposition" if superposition else ""

    df0 = pd.read_csv("data/derived/properties.synthesis.csv")

    report = pre_process_properties(df0)
    labels = report.pop("labels")
    descriptions = report.pop("descriptions")
    ixlabel_ixdesc = report.pop("ixlabel_ixdesc")
    properties = report.pop("properties")
    property_label_map = report.pop("property_label_map")

    tokenizer, model = load_models(model_type, sentence)

    layers_str = LinkerModel.layers2str(layers)
    tt_labels = encode(labels, tokenizer, model, layers)

    if superposition:
        tt_descs = encode(descriptions, tokenizer, model, layers)
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
    lm = LinkerModel(
        index=index, vocabulary=properties, layers=layers, labels_map=property_label_map
    )
    file_spec = files("pelinker.store").joinpath(
        f"pelinker.model.{model_type}.{layers_str}{suffix}"
    )
    lm.dump(file_spec)


if __name__ == "__main__":
    run()
