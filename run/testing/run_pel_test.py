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
    process_text,
    text_to_tokens_embeddings,
    MAX_LENGTH,
    load_models,
    tt_aggregate_normalize,
)

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
    suffix = f"ctx_{'extra' if extra_context else 'vb'}{superposition_sfx}"
    save_topk = 3
    # roi = ["induce", "associa", "suppress"]
    report_path = "./reports"
    with open(text_path) as json_file:
        json_data = json.load(json_file)
    text = json_data["text"]

    df0 = pd.read_csv("data/derived/properties.synthesis.csv")

    tokenizer, model = load_models(model_type)

    nlp = spacy.load("en_core_web_sm")

    # df = df0.loc[df0["label"].apply(lambda x: any(y in x for y in roi))]
    df = df0.copy()

    id_label_dict = dict(df[["property", "label"]].values)
    ixlabel_id_dict = df["property"].reset_index(drop=True).to_dict()

    mask_desc_exclude = df["description"].isnull() | df["description"].apply(
        lambda x: "inverse" in x.lower() if isinstance(x, str) else True
    )

    id_ixlabel_dict = {v: k for k, v in ixlabel_id_dict.items()}

    ixdesc_id = list(
        df.loc[~mask_desc_exclude, "property"].reset_index(drop=True).items()
    )

    ixlabel_ixdesc = {id_ixlabel_dict[label]: ixd for ixd, label in ixdesc_id}

    tt_labels_layered, labels_spans = text_to_tokens_embeddings(
        df["label"].values.tolist(), tokenizer, model
    )

    tt_descs_layered, desc_spans = text_to_tokens_embeddings(
        df.loc[~mask_desc_exclude, "description"].values.tolist(),
        tokenizer,
        model,
    )

    layers = [[-1], [-3, -2, -1], [-6, -5, -4, -3, -2, -1], [-1, -2, -8, -9]]

    for ls in layers:
        layers_str = "_".join([str(x) for x in ls])
        print(f">>> {layers_str} <<<")
        tt_labels = tt_aggregate_normalize(tt_labels_layered, ls)
        tt_descs = tt_aggregate_normalize(tt_descs_layered, ls)

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

        lm = LinkerModel(index=index, vocabulary=ids, ls=layers)

        nb_nn = min([10, tt_labels.shape[0]])

        # spans list[spans]
        sents, spans, tt_text = process_text(
            text,
            tokenizer,
            model,
            nlp,
            max_length=MAX_LENGTH,
            extra_context=extra_context,
        )

        # tt_text = tt_text / tt_text.norm(dim=-1).unsqueeze(-1)

        tt_text = tt_text[ls].mean(0)

        metrics = []
        for s, miti, tt_sent in zip(sents, spans, tt_text):
            tt_words_list = []
            for k, v in miti:
                rr = tt_sent[v].mean(0)
                rr = rr / rr.norm(dim=-1).unsqueeze(-1)
                tt_words_list += [rr]

            tt_words = torch.stack(tt_words_list)

            distance_matrix, nearest_neighbors_matrix = index.search(tt_words, nb_nn)

            for bj, (nn, d, miti_item) in enumerate(
                zip(nearest_neighbors_matrix, distance_matrix, miti)
            ):
                a, b = miti_item[0]
                clabels = [id_label_dict[ixlabel_id_dict[nnx]] for nnx in nn]

                d = d.tolist()

                dif = d[0] - d[1]
                m0 = [bj, a, s[a:b]]
                m0 += [round(dif, 4)]
                for cl, cs in zip(clabels[:save_topk], d[:save_topk]):
                    m0 += [cl, round(cs, 4)]

                metrics += [tuple(m0)]

                print(f"{a} | {s[a:b]} | {clabels[:2]} | {d[0]:.4f} | {dif:.4g}")
        cols = [
            item
            for pair in zip(
                [f"cand_{j}" for j in range(save_topk)],
                [f"score_{j}" for j in range(save_topk)],
            )
            for item in pair
        ]
        metrics_df = pd.DataFrame(
            metrics, columns=["nb", "ntoken", "target", "top2next_separation"] + cols
        )
        metrics_df.to_csv(
            f"{report_path}/metrics_{model_type}_{layers_str}{suffix}.csv"
        )
        print(metrics_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()
