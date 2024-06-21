import faiss
from pelinker.util import process_text
import torch


class LinkerModel:
    def __init__(self, index: faiss.IndexFlatIP, vocabulary: list[str], ls, nb_nn=10):
        self.index = index
        self.vocabulary = vocabulary
        self.ls = ls
        self.nb_nn = nb_nn

    def link(self, text, tokenizer, model, nlp, max_length, extra_context):
        sents, spans, tt_text = process_text(
            text,
            tokenizer,
            model,
            nlp,
            max_length=max_length,
            extra_context=extra_context,
        )
        tt_text = tt_text[self.ls].mean(0)

        report = []
        for js, (s, miti, tt_sent) in enumerate(zip(sents, spans, tt_text)):
            tt_words_list = []
            for k, v in miti:
                rr = tt_sent[v].mean(0)
                rr = rr / rr.norm(dim=-1).unsqueeze(-1)
                tt_words_list += [rr]

            tt_words = torch.stack(tt_words_list)

            distance_matrix, nearest_neighbors_matrix = self.index.search(
                tt_words, self.nb_nn
            )

            for bj, (nn, d, miti_item) in enumerate(
                zip(nearest_neighbors_matrix, distance_matrix, miti)
            ):
                a, b = miti_item[0]
                d = d.tolist()

                clabels = [self.vocabulary[nnx] for nnx in nn]

                dif = float(d[0] - d[1])

                report += [
                    {
                        "js": js,
                        "a": a,
                        "b": b,
                        "ent": clabels[0],
                        "conf": float(d[0]),
                        "mention": s[a:b],
                        "nei_dif": dif,
                    }
                ]

        slens = [len(s) + 1 for s in sents[:-1]]
        cumsum = [0]
        for s in slens:
            cumsum += [cumsum[-1] + s]

        report2 = []
        for r in report:
            js = r.pop("js")
            a = r.pop("a")
            b = r.pop("b")
            report2 += [{**{"a": a + cumsum[js], "b": b + cumsum[js]}, **r}]
        sall = " ".join(sents)
        return {"entities": report2, "normalized_text": sall}
