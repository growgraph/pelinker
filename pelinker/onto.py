import dataclasses
from dataclasses import field
from enum import Enum
from dataclass_wizard import JSONWizard
import torch
from collections import defaultdict

MAX_LENGTH = 512


class WordGrouping(Enum):
    W1 = 1
    W2 = 2
    W3 = 3
    W4 = 4


class BaseDataclass(JSONWizard, JSONWizard.Meta):
    key_transform_with_dump = "SNAKE"
    skip_defaults = True


@dataclasses.dataclass
class ChunkMapper(BaseDataclass):
    tensor: torch.Tensor  # n_layers x n_batch x n_len x n_emb - tensor where n_batch dim goes over all chunks
    chunks: list[str]  # flat list of chunks
    token_spans_list: list[
        list[tuple[int, int]]
    ]  # for each chunk contains a list of token spans
    it_ic: list[tuple[int, int]]
    cumulative_lens: list[list[int]]
    text_word_spans_list: list[list[tuple[int, int]]] | None = None
    token_word_spans_list: list[list[tuple[int, int]]] | None = None
    mapping_table: list[tuple[int, int, tuple[int, int], tuple[int, int]]] | None = None
    text_chunk_map: defaultdict[int, list] = field(
        default_factory=lambda: defaultdict(list)
    )
    tt_expressions: list[torch.Tensor] = field(
        default_factory=list
    )  # n_expressions [n_len x n_emb]

    def set_token_word_spans(self, word_int_bounds):
        from pelinker.util import map_words_to_tokens_list

        self.token_word_spans_list, self.text_word_spans_list = (
            map_words_to_tokens_list(self.token_spans_list, word_int_bounds)
        )

    def set_mapping_table(self):
        it_ic = sorted(self.it_ic)
        self.mapping_table = []
        self.text_chunk_map = defaultdict(list)

        if self.text_word_spans_list is None:
            pass
        for (ichunk, (itext, ichunk_local)), chsp in zip(
            enumerate(it_ic), self.text_word_spans_list
        ):
            self.text_chunk_map[itext].append(ichunk)
            chunk_offset = self.cumulative_lens[itext][ichunk_local]
            for a, b in chsp:
                self.mapping_table += [
                    (itext, ichunk, (a, b), (a + chunk_offset, b + chunk_offset))
                ]

    def map_chunk_to_text(self, itext, ichunk_local, a=0):
        return a + self.cumulative_lens[itext][ichunk_local]

    def ichunk_to_itext_ichunk_local(self, ichunk: int):
        return self.it_ic[ichunk]

    def ichunk_char_offset(self, ichunk: int):
        itext, ichunk_local = self.ichunk_to_itext_ichunk_local(ichunk)
        return self.cumulative_lens[itext][ichunk_local]


@dataclasses.dataclass
class SimplifiedToken(BaseDataclass):
    ix: int
    ix_end: int
    text: str
    lemma: str
    tag: str


@dataclasses.dataclass
class Expression(BaseDataclass):
    tokens: list[SimplifiedToken]
    itext: int | None = None  # index of document
    ichunk: int | None = None  # index of document chunk
    a: int | None = None  # index of the first character
    b: int | None = None  # index of the last character

    def __post_init__(self):
        self.tokens = sorted(self.tokens, key=lambda x: x.ix)
        self.a = self.tokens[0].ix
        self.b = self.tokens[-1].ix_end


@dataclasses.dataclass
class ExpressionHolder(BaseDataclass):
    tt: torch.Tensor
    expressions: list[Expression]

    def __post_init__(self):
        if len(self.expressions) != self.tt.shape[0]:
            raise ValueError(
                "The number of expressions does not match the shape of the tensor"
            )

    def filter_on_lemmas(
        self, tokens: list[SimplifiedToken]
    ) -> list[tuple[Expression, torch.Tensor]]:
        tokens_lemmatized = " ".join([e.lemma for e in tokens])
        return [
            (e, t)
            for e, t in zip(self.expressions, self.tt)
            if " ".join([t.lemma for t in e.tokens]) == tokens_lemmatized
        ]


@dataclasses.dataclass
class ExpressionHolderBatch(BaseDataclass):
    expression_data: list[ExpressionHolder]
    word_grouping: WordGrouping | None = None


@dataclasses.dataclass
class ReportBatch(BaseDataclass):
    _data: list[ExpressionHolderBatch]
    texts: list[str]
    chunk_mapper: ChunkMapper

    def __post_init__(self):
        for item in self._data:
            if len(self.texts) != len(item.expression_data):
                raise ValueError(
                    "The number of ExpressionHolders does not match the number of texts"
                )

    def __getitem__(self, wg: WordGrouping) -> ExpressionHolderBatch:
        filtered = [item for item in self._data if item.word_grouping == wg]
        if filtered:
            return filtered[0]
        else:
            raise ValueError(f"{wg} not available")

    def get_data_for_grouping(self, wg: WordGrouping):
        return [item for item in self._data if item.word_grouping == wg]

    def available_groupings(self):
        return [item.word_grouping for item in self._data]
