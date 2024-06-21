# pylint: disable=E1120

import json
import logging.config
import sys

import click
import faiss
import pandas as pd
import spacy
import torch

from pelinker.util import (
    text_to_tokens_embeddings,
    MAX_LENGTH,
    load_models,
    tt_aggregate_normalize,
)
from pelinker.preprocess import pre_process_properties
from pelinker.model import LinkerModel


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
def run(text_path, model_type, superposition, extra_context):
    superposition_sfx = "_superposition" if superposition else ""
    suffix = f"_ctx_{'extra' if extra_context else 'vb'}{superposition_sfx}"
    # save_topk = 3
    # roi = ["induce", "associa", "suppress"]
    report_path = "./reports"
    with open(text_path) as json_file:
        json_data = json.load(json_file)
    text = json_data["text"]

    df0 = pd.read_csv("data/derived/properties.synthesis.csv")

    report = pre_process_properties(df0)
    labels = report.pop("labels")
    descriptions = report.pop("descriptions")
    ixlabel_ixdesc = report.pop("ixlabel_ixdesc")
    properties = report.pop("properties")
    property_label_map = report.pop("property_label_map")

    tokenizer, model = load_models(model_type)

    nlp = spacy.load("en_core_web_sm")

    tt_labels_layered, labels_spans = text_to_tokens_embeddings(
        labels, tokenizer, model
    )

    tt_descs_layered, desc_spans = text_to_tokens_embeddings(
        descriptions,
        tokenizer,
        model,
    )

    layers_list = [[-3, -2, -1], [-6, -5, -4, -3, -2, -1], [-1, -2, -8, -9]][:1]

    for layers in layers_list:
        layers_str = LinkerModel.encode_layers(layers)
        print(f">>> {layers_str} <<<")
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

        nb_nn = min([10, tt_labels.shape[0]])
        lm = LinkerModel(
            index=index,
            vocabulary=properties,
            layers=layers,
            labels_map=property_label_map,
            nb_nn=nb_nn,
        )
        report = lm.link(text, tokenizer, model, nlp, MAX_LENGTH, extra_context)

        metrics_df = pd.DataFrame(report["entities"])
        metrics_df.to_csv(
            f"{report_path}/metrics_{model_type}_{layers_str}{suffix}.csv"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()
