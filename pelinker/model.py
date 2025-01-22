import faiss
from pelinker.util import texts_to_vrep
from pelinker.onto import WordGrouping
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
            if r["dif_to_next"] >= thr_dif and r["score"] >= thr_score
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

    def link(self, texts, tokenizer, model, nlp, max_length, topk=None):
        if self.index is None:
            raise TypeError("index not set")

        # if self.ls == "sent":
        #     return self._link_sent(
        #         texts, tokenizer, model, nlp, extra_context, topk=topk
        #     )

        report = texts_to_vrep(
            texts,
            tokenizer,
            model,
            layers_spec=self.ls,
            word_modes=[WordGrouping.VERBAL],
            nlp=nlp,
            max_length=max_length,
        )

        wg_current = report["word_groupings"][WordGrouping.VERBAL]

        tt_list = []
        vocabulary = []
        for sentence in wg_current:
            tt_list += [t for _, t in sentence]
            vocabulary += [item for item, _ in sentence]
        tt = torch.concat(tt_list)

        distance_matrix, nearest_neighbors_matrix = self.index.search(tt, self.nb_nn)

        kb_items = []
        for item, nn, d in zip(vocabulary, nearest_neighbors_matrix, distance_matrix):
            item = self.complement_with_kb_data(item, nn, d, topk=topk)
            kb_items += [item]

        report["entities"] = kb_items
        return report

    def complement_with_kb_data(self, item, nearest_neighbors, distance, topk):
        distance = distance.tolist()

        candidate_entity = [self.vocabulary[nnx] for nnx in nearest_neighbors]

        dif = round(distance[0] - distance[1], 5)
        item = {
            **item,
            **{
                "entity_id_predicted": candidate_entity[0],
                "score": round(distance[0], 4),
                "dif_to_next": dif,
            },
        }
        if topk is not None:
            item["_leading_candidates"] = candidate_entity[1:topk]
            item["_leading_scores"] = [round(x, 4) for x in distance[1:topk]]

        if self.labels_map:
            item["entity_label"] = (
                self.labels_map[candidate_entity[0]]
                if candidate_entity[0] in self.labels_map
                else "NA"
            )
            if topk is not None:
                item["_leading_candidates_labels"] = [
                    self.labels_map[e] if e in self.labels_map else "NA"
                    for e in candidate_entity[1:topk]
                ]

        return item
