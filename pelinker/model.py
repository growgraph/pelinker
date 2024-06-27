import faiss
from pelinker.util import process_text, encode, get_vb_spans
import torch
import joblib


class LinkerModel:
    def __init__(
        self,
        vocabulary: list[str],
        layers,
        nb_nn=10,
        index: faiss.IndexFlatIP | None = None,
        **kwargs,
    ):
        self.index: faiss.IndexFlatIP | None = index
        self.vocabulary: list[str] = vocabulary
        self.labels_map: dict[str, str] = kwargs.pop("labels_map", dict())
        self.ls = layers
        self.nb_nn = nb_nn

    def link(self, text, tokenizer, model, nlp, max_length, extra_context=False):
        if self.index is None:
            raise TypeError("index not set")

        if self.ls == "sent":
            return self._link_sent(
                text, tokenizer, model, nlp, max_length, extra_context
            )
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

                candidate_entity = [self.vocabulary[nnx] for nnx in nn]

                dif = round(d[0] - d[1], 5)
                item = {
                    "js": js,
                    "a": a,
                    "b": b,
                    "mention": s[a:b],
                    "entity": candidate_entity[0],
                    "score": round(d[0], 4),
                    "dif_to_next": dif,
                }
                if self.labels_map:
                    item["entity_label"] = (
                        self.labels_map[candidate_entity[0]]
                        if candidate_entity[0] in self.labels_map
                        else "NA"
                    )
                report += [item]

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

    def _link_sent(self, text, tokenizer, model, nlp, max_length, extra_context):
        spans = get_vb_spans(nlp, text, extra_context=extra_context)

        vbs = [text[a:b] for a, b in spans]
        tt_relations = encode(vbs, tokenizer, model, self.ls)

        distance_matrix, nearest_neighbors_matrix = self.index.search(
            tt_relations, self.nb_nn
        )

        report = []
        for bj, (nn, d, span) in enumerate(
            zip(nearest_neighbors_matrix, distance_matrix, spans)
        ):
            a, b = span
            d = d.tolist()

            candidate_entity = [self.vocabulary[nnx] for nnx in nn]

            dif = round(d[0] - d[1], 5)
            item = {
                "a": a,
                "b": b,
                "mention": text[a:b],
                "entity": candidate_entity[0],
                "score": round(d[0], 4),
                "dif_to_next": dif,
            }
            if self.labels_map:
                item["entity_label"] = (
                    self.labels_map[candidate_entity[0]]
                    if candidate_entity[0] in self.labels_map
                    else "NA"
                )
            report += [item]
        return {"entities": report, "normalized_text": text}

    @classmethod
    def layers2str(cls, layers):
        if isinstance(layers, str):
            layers_str = layers
        else:
            if any(l0 > 0 for l0 in layers):
                raise ValueError(f" there are positive layers: {layers}")
            alayers = sorted([abs(l0) for l0 in layers])
            layers_str = "".join([str(l0) for l0 in alayers])
        return layers_str

    @classmethod
    def str2layers(cls, layers_spec):
        if "," in layers_spec:
            layers_spec = "".join(layers_spec.split(","))
        if layers_spec.isdigit():
            try:
                layers = list(set([-abs(int(x)) for x in layers_spec]))
            except:
                raise ValueError(f"{layers_spec} could not be parsed into layers")
        else:
            layers = layers_spec
        return layers

    @classmethod
    def filter_report(cls, report, thr_score, thr_dif):
        report["entities"] = [
            r
            for r in report["entities"]
            if r["dif_to_next"] > thr_dif and r["score"] > thr_score
        ]
        return report

    def dump(self, file_spec):
        faiss.write_index(self.index, f"{file_spec}.index")
        self.index = None
        joblib.dump(self, f"{file_spec}.gz", compress=3)

    @classmethod
    def load(cls, file_spec):
        index = faiss.read_index(f"{file_spec}.index")
        pe_model = joblib.load(f"{file_spec}.gz")
        pe_model.index = index
        return pe_model
