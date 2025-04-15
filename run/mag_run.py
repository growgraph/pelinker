import spacy
import torch
import pandas as pd
import tqdm

from pelinker.matching import match_pattern
from pelinker.util import texts_to_vrep
from pelinker.onto import WordGrouping
from pelinker.util import map_spans_to_spans

from transformers import AutoTokenizer, AutoModel


M = AutoModel.from_pretrained("NeuML/pubmedbert-base-embeddings", output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("data/test/sample.tsv", sep='\t', header=None)

props = ['activates', 'enhances', 'upregulates', 'suppress']
texts = df[1]

indexes_of_interest_per_pat = []
for p in props:
    indexes_of_interest_per_pat += [[match_pattern(p, x, suffix_length=0) for x in texts]]

flags = [any(True if ixs_ else False for ixs_ in item)
        for item in zip(*indexes_of_interest_per_pat)
        ]

data = [t for flag, t in zip(flags, texts) if flag]

batch_size = 40

data_batched = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

frep = []
tt_averages = []


for batch in (pbar := tqdm.tqdm(data_batched)):
    report = texts_to_vrep(batch, tokenizer, M, [1,2], word_modes=[WordGrouping.W1],)

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

                tts = [torch.stack([t for _, t in report_sent[ja:jb]]).mean(0)
                        for ja, jb in map_ij
                        ]

                mentions = [" ".join([x["mention"] for x, _ in report_sent[ja:jb]])
                        for (ja, jb) in map_ij]

                frep += [(w, jsent, p, m, tt) for m, tt in zip(mentions, tts)]

    pbar.set_description(f"entities added : {len(frep)}")

tt_all = (torch.stack([x[4] for x in frep]), "all.patterns")
tt_pats = [(torch.stack([x[4] for x in frep if x[2] == p]), p) for p in props]






