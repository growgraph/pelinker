# pylint: disable=E1120

import json
import logging.config
import sys

import click
import pandas as pd
import spacy

from pelinker.util import (
    load_models,
)
from pelinker.onto import MAX_LENGTH
from pelinker.preprocess import pre_process_properties
from pelinker.model import LinkerModel
from pelinker.util import fetch_latest_kb, report2kb, texts_to_vrep
from pelinker.onto import WordGrouping
from pathlib import Path


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
    gt = json_data["ground_truth"]

    path_derived = Path("./data/derived/")

    fname, version = fetch_latest_kb(path_derived)

    try:
        df0 = pd.read_csv(path_derived / fname)
    except Exception as e:
        print(f"kb not found at {path_derived}")
        raise e

    layers = LinkerModel.str2layers(layers_spec)
    sentence = True if layers == "sent" else False

    report = pre_process_properties(df0)
    labels = report.pop("labels")
    properties = report.pop("entity_ids")
    property_label_map = report.pop("property_label_map")

    nlp = spacy.load("en_core_web_trf")

    tokenizer, model = load_models(model_type, sentence)

    layers_str = LinkerModel.layers2str(layers)

    report = texts_to_vrep(
        labels,
        tokenizer,
        model,
        layers_spec=layers,
        word_mode=WordGrouping.SENTENCE,
        nlp=nlp,
    )
    vocabulary, index = report2kb(report)

    nb_nn = min([10, report["tensor"].shape[0]])
    lm = LinkerModel(
        index=index,
        vocabulary=properties,
        layers=layers,
        labels_map=property_label_map,
        nb_nn=nb_nn,
    )
    report = lm.link(
        [text],
        tokenizer,
        model,
        nlp,
        max_length=MAX_LENGTH,
        topk=3,
    )

    df_gt = pd.DataFrame(gt)

    df_pred = pd.DataFrame(report["entities"])
    df_gt.merge(df_pred, how="left", on=["itext", "a"], suffixes=("", "_gt"))
    df_cmp = df_gt.merge(df_pred, how="left", on=["itext", "a"], suffixes=("", "_gt"))

    accuracy = (
        (df_cmp["entity_id"] == df_cmp["entity_id_predicted"]).astype(float).mean()
    )
    accuracy = round(accuracy, 5)
    print(f"{model_type} | {layers_str} | {suffix} : accuracy >> {accuracy} <<")
    report = {"accuracy": accuracy, "prediction": df_cmp.to_dict(orient="records")}
    with open(
        f"{report_path}/metrics_{model_type}_{layers_str}{suffix}.json", "w"
    ) as file:
        json.dump(report, file, indent=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()
