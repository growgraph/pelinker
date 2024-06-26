# pylint: disable=E1120

import json
import logging.config
import sys

import click
import faiss
import pandas as pd
import spacy

from pelinker.util import (
    MAX_LENGTH,
    load_models,
)
from pelinker.preprocess import pre_process_properties
from pelinker.model import LinkerModel
from pelinker.util import encode


@click.command()
@click.option("--text-path", type=click.Path())
@click.option("--model-type", type=click.STRING, default="scibert")
@click.option("--extra-context", type=click.BOOL, is_flag=True, default=False)
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
    default="1,2,3",
    help="`sent` or a string of layers, `1,2,3` would correspond to layers [-1, -2, -3]",
)
def run(text_path, model_type, superposition, extra_context, layers_spec):
    superposition_sfx = "_superposition" if superposition else ""
    suffix = f"_ctx_{'extra' if extra_context else 'vb'}{superposition_sfx}"
    report_path = "./reports"
    with open(text_path) as json_file:
        json_data = json.load(json_file)
    text = json_data["text"]

    df0 = pd.read_csv("data/derived/properties.synthesis.v2.csv")

    layers = LinkerModel.str2layers(layers_spec)
    sentence = True if layers == "sent" else False

    report = pre_process_properties(df0)
    labels = report.pop("labels")
    properties = report.pop("properties")
    property_label_map = report.pop("property_label_map")

    nlp = spacy.load("en_core_web_sm")

    tokenizer, model = load_models(model_type, sentence)

    layers_str = LinkerModel.layers2str(layers)
    print(f">>> {layers_str} <<<")
    tt_labels = encode(labels, tokenizer, model, layers)

    tt_basis = tt_labels

    index = faiss.IndexFlatIP(tt_basis.shape[1])
    index.add(tt_basis)

    nb_nn = min([10, tt_labels.shape[0]])
    lm = LinkerModel(
        index=index,
        vocabulary=properties,
        layers=layers,
        labels_map=property_label_map,
        nb_nn=nb_nn,
    )

    report = lm.link(
        text, tokenizer, model, nlp, extra_context=extra_context, max_length=MAX_LENGTH
    )

    metrics_df = pd.DataFrame(report["entities"])
    metrics_df.to_csv(f"{report_path}/metrics_{model_type}_{layers_str}{suffix}.v2.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()
