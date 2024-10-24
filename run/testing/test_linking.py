# pylint: disable=E1120

import logging

import faiss
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from pelinker.util import fetch_latest_kb
from pathlib import Path


def vectorize_text(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu()


def main():
    path_derived = Path("./data/derived/")

    fname, version = fetch_latest_kb(path_derived)

    try:
        df = pd.read_csv(path_derived / fname)
    except Exception as e:
        print(f"kb not found at {path_derived}")
        raise e

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    labels = [
        (j, p, vectorize_text(x, tokenizer, model))
        for j, (p, x) in enumerate(df[["entity_ids", "label"]].values)
    ]
    descs = [
        (j, p, vectorize_text(x, tokenizer, model))
        for j, (p, x) in enumerate(
            df.loc[df["description"].notnull(), ["entity_ids", "description"]].values
        )
    ]

    tt_labels = torch.stack([x[-1] for x in labels])
    tt_descs = torch.stack([x[-1] for x in descs])

    tt_labels = tt_labels / tt_labels.norm(dim=1).unsqueeze(1)
    tt_descs = tt_descs / tt_descs.norm(dim=1).unsqueeze(1)

    index = faiss.IndexFlatIP(tt_labels.shape[1])
    nb_nn = 3
    index.add(tt_labels)
    distance_matrix, nearest_neighbors_matrix = index.search(tt_descs, nb_nn)
    nb_nn = 10


if __name__ == "__main__":
    main()
