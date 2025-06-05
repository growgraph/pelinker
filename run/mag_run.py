import spacy
import torch
import pickle
import numpy as np
import pandas as pd
import tqdm
import pdb

from pelinker.matching import match_pattern
from pelinker.util import texts_to_vrep
from pelinker.onto import WordGrouping
from pelinker.util import map_spans_to_spans

from transformers import AutoTokenizer, AutoModel


M = AutoModel.from_pretrained(
    "NeuML/pubmedbert-base-embeddings", output_hidden_states=True
)
tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")

nlp = spacy.load("en_core_web_sm")

data_path = "data/jamshid/bio_mag_2M.tsv"
df = pd.read_csv(data_path, 
                 sep="\t", 
                 header=None, 
                 chunksize=2000
        )


props_path = "data/test/uni_props.txt"
with open(props_path, "r") as f:
    all_props = f.read().split("\n")


def run_on_selected_texts():
    
    res = {}
    save_path = "data/jamshid/bio_2M_res.pkl"
    for i,chunk in tqdm.tqdm(enumerate(df), leave=True, position=0):

        chunk_texts = list(chunk[1])
        extract_and_embed_mentions(all_props, chunk_texts, res)
        with open("data/jamshid/log", 'w') as f:
            f.write(str(i))

        if not(i%20):
            with open(save_path, 'wb') as f:
                pickle.dump(res, f)


def extract_and_embed_mentions(props, texts, embeds_dict={}):

    indexes_of_interest_per_pat = []
    for p in props:
        indexes_of_interest_per_pat += [
            [match_pattern(p, x, suffix_length=0) for x in texts]
        ]

    flags = [
        any(True if ixs_ else False for ixs_ in item)
        for item in zip(*indexes_of_interest_per_pat)
    ]

    data = [t for flag, t in zip(flags, texts) if flag]

    batch_size = 40

    data_batched = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    frep = []
    tt_averages = []


    for batch in (pbar := tqdm.tqdm(data_batched)):
        report = texts_to_vrep(
            batch,
            tokenizer,
            M,
            [1, 2],
            word_modes=[WordGrouping.W1],
        )

        for p in props:
            normalized_texts = report["normalized_text"]
            word_groupings = report["word_groupings"]

            for w, r_item in word_groupings.items():
                for jsent, (text, report_sent) in enumerate(zip(normalized_texts, r_item)):
                    if p == props[0] and w == sorted(word_groupings)[0]:
                        tt0 = [t for _, t in report_sent]
                        tt_averages += tt0

                    indexes_of_interest_batched = match_pattern(p, text, suffix_length=0)

                    if not indexes_of_interest_batched:
                        continue

                    report_sent = sorted(report_sent, key=lambda x: x[0]["a"])

                    _, map_ij = map_spans_to_spans(
                        [(x["a"], x["b"]) for x, _ in report_sent],
                        indexes_of_interest_batched,
                    )

                    tts = [
                        torch.stack([t for _, t in report_sent[ja:jb]]).mean(0)
                        for ja, jb in map_ij
                    ]

                    mentions = [
                        " ".join([x["mention"] for x, _ in report_sent[ja:jb]])
                        for (ja, jb) in map_ij
                    ]

                    frep += [(w, jsent, p, m, tt) for m, tt in zip(mentions, tts)]

        pbar.set_description(f"entities added : {len(frep)}")

    uprops = np.unique([x[2] for x in frep])

    for p in uprops:
        new_embeds = torch.stack([x[4] for x in frep if x[2] == p])
        if p in embeds_dict:
            embeds_dict[p] = torch.concat((embeds_dict[p], new_embeds), axis=0)
        else:
            embeds_dict[p] = new_embeds

    return embeds_dict




