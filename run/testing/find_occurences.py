# pylint: disable=E1120

import json
import logging.config
import sys
import spacy

import click
import pandas as pd

from pelinker.matching import match_pattern, match_pieces
from pelinker.util import fetch_latest_kb, get_vb_spans, split_into_sentences
from pathlib import Path


@click.command()
@click.option("--text-path", type=click.Path())
def run(text_path):
    suffix_length = 1
    with open(text_path) as json_file:
        json_data = json.load(json_file)
    text = json_data["text"]
    path_derived = Path("./data/derived/")

    fname, version = fetch_latest_kb(path_derived)

    try:
        df0 = pd.read_csv(path_derived / fname)
    except Exception as e:
        print(f"kb not found at {path_derived}")
        raise e

    nlp = spacy.load("en_core_web_trf")

    patterns = df0["label"].values[:]

    texts = split_into_sentences(text)

    for text in texts:
        word_bnds = get_vb_spans(nlp=nlp, text=text, extra_context=False)
        for pattern in patterns:
            matches = match_pattern(pattern, text, suffix_length=1)
            if matches:
                vb_match = [
                    (a, b)
                    for a, b in matches
                    if any(min([y, b]) - max([x, a]) > 0 for x, y in word_bnds)
                ]
                if vb_match:
                    wmaps = match_pieces(
                        pattern, text, suffix_length=suffix_length, matches=matches
                    )
                    print(pattern, wmaps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()
