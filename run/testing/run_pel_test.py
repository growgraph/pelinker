# pylint: disable=E1120

import json
import logging.config
import sys

import click
import faiss
import pandas as pd
import spacy
import torch
from transformers import AutoModel, AutoTokenizer

from pelinker.util import process_text, text_to_tokens_embeddings


@click.command()
@click.option("--text-path", type=click.Path())
@click.option("--model-type", type=click.STRING, default="scibert")
def run(text_path, model_type):
    # fig_path = "./figs"
    with open(text_path) as json_file:
        json_data = json.load(json_file)
    text = json_data["text"]

    df0 = pd.read_csv("data/derived/properties.synthesis.csv")

    if model_type == "scibert":
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
    else:
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    nlp = spacy.load("en_core_web_sm")

    # df = df0.loc[df0["label"].apply(lambda x: any(y in x for y in roi))]
    df = df0.copy()

    id_label_dict = dict(df[["property", "label"]].values)
    ixlabel_id_dict = df["property"].reset_index(drop=True).to_dict()
    mask_desc_notnull = df["description"].notnull()

    df.loc[mask_desc_notnull, "property"].reset_index(drop=True)

    tt_labels_layered, labels_spans = text_to_tokens_embeddings(
        df["label"].values.tolist(), tokenizer, model
    )

    layers = [[-5], [-4], [-3], [-2], [-1], [-3, -2, -1]]
    layers = [[-1], [-5, -4, -3, -2, -1]]

    for ls in layers:
        layers_str = "_".join([str(x) for x in ls])
        print(f">>> {layers_str} <<<")
        tt_labels = tt_labels_layered[ls].mean(0).mean(dim=1)
        tt_labels = tt_labels / tt_labels.norm(dim=-1).unsqueeze(-1)

        index = faiss.IndexFlatIP(tt_labels.shape[1])
        index.add(tt_labels)

        nb_nn = min([10, tt_labels.shape[0]])
        # spans list[spans]
        sents, spans, tt_text = process_text(text, tokenizer, model, nlp)
        # tt_text = tt_text / tt_text.norm(dim=-1).unsqueeze(-1)

        tt_text = tt_text[ls].mean(0)

        # roi = ["induce", "associa", "suppress"]
        for s, miti, tt_sent in zip(sents, spans, tt_text):
            tt_words_list = []
            for k, v in miti:
                rr = tt_sent[v].mean(0)
                rr = rr / rr.norm(dim=-1).unsqueeze(-1)
                tt_words_list += [rr]

            tt_words = torch.stack(tt_words_list)

            distance_matrix, nearest_neighbors_matrix = index.search(tt_words, nb_nn)

            for nn, d, miti_item in zip(
                nearest_neighbors_matrix, distance_matrix, miti
            ):
                a, b = miti_item[0]
                clabels = [id_label_dict[ixlabel_id_dict[nnx]] for nnx in nn]

                dif = d[0] - d[1]
                # if any([x in s[a:b] for x in roi]):
                print(f"{a} | {s[a:b]} | {clabels[:2]} | {d[0]:.4f} | {dif:.4g}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()
