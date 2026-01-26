# pylint: disable=E1120

import click
import faiss
import pandas as pd
import pathlib
from pelinker.util import load_models, encode
from pelinker.preprocess import pre_process_properties
from pelinker.util import fetch_latest_kb
from pathlib import Path


@click.command()
@click.option(
    "--rep-string",
    type=click.STRING,
    default="1,2,3",
    help="`sent` or a string of layers, `1,2,3` would correspond to layers [-1, -2, -3]",
)
@click.option(
    "--model-type",
    type=click.STRING,
    default="biobert",
    help="choose BERT flavours",
)
@click.option(
    "--kb-path",
    type=click.Path(path_type=pathlib.Path),
    default="data/derived",
    help="choose BERT flavours",
)
def run(rep_string, model_type, kb_path):
    path_derived = Path(kb_path)

    fname, version = fetch_latest_kb(path_derived)

    try:
        df0 = pd.read_csv(path_derived / fname)
    except Exception as e:
        print(f"kb not found at {path_derived}")
        raise e

    report = pre_process_properties(df0)
    labels = report.pop("labels")
    if rep_string != "sent":
        try:
            layers = rep_string.split(",")
            ls = set([-int(x) for x in layers])
        except:
            raise ValueError(f"{rep_string} could not be parsed into layers")
        sentence = False
    else:
        sentence = True
        ls = rep_string

    # labels = labels[:] + ["queen", "king", "chess", "US Embassy", "Bill Gates"]
    tokenizer, model = load_models(model_type, sentence)
    tt_labels = encode(labels, tokenizer, model, ls)

    index = faiss.IndexFlatIP(tt_labels.shape[1])
    nb_nn = min([100, tt_labels.shape[0]])
    index.add(tt_labels)

    distance_matrix, nearest_neighbors_matrix = index.search(tt_labels, nb_nn)
    ds = distance_matrix[:, 1:].flatten()
    ds.sort()
    k_links = 20

    print(ds[-k_links:])
    thr = ds[-k_links]

    edges = []
    for dd, nn in zip(distance_matrix, nearest_neighbors_matrix):
        m = dd >= thr
        equis = nn[m]
        edges += [tuple(sorted((equis[0], c))) for c in equis[1:]]

    edges = set(edges)
    nlabels = [(labels[a], labels[b]) for a, b in edges]
    print(nlabels)


if __name__ == "__main__":
    run()
