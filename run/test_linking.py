# pylint: disable=E1120

import logging

import faiss
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def vectorize_text(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu()


def main():
    df = pd.read_csv("../data/derived/properties.synthesis.csv")

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    labels = [
        (j, p, vectorize_text(x, tokenizer, model))
        for j, (p, x) in enumerate(df[["property", "label"]].values)
    ]
    descs = [
        (j, p, vectorize_text(x, tokenizer, model))
        for j, (p, x) in enumerate(
            df.loc[df["description"].notnull(), ["property", "description"]].values
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


if __name__ == "__main__":
    main()
