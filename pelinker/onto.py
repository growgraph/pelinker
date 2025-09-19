import dataclasses
from enum import Enum
from dataclass_wizard import JSONWizard
import torch


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
    # n_layers x n_batch x n_len x n_emb
    tt: torch.Tensor
    flattened_chunks: list[str]
    token_bounds: list[list[tuple[int, int]]]
    it_ic: list[tuple[int, int]]
    chunk_cumlens: list[list[int]]
    char_spans: list[list[tuple[int, int]]] | None = None
    token_spans: list[list[tuple[int, int]]] | None = None


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
    ibatch: int | None = None
    itext: int | None = None  # index of text
    a: int | None = None  # index of the first character
    b: int | None = None  # index of the last character

    def __post_init__(self):
        self.tokens = sorted(self.tokens, key=lambda x: x.ix)
        self.a = self.tokens[0].ix
        self.b = self.tokens[-1].ix_end


@dataclasses.dataclass
class ExpressionBatch(BaseDataclass):
    tt: torch.Tensor
    expressions: list[Expression]

    def __post_init__(self):
        if len(self.expressions) != self.tt.shape[0]:
            raise ValueError(
                "The number of expressions does not match the shape of the tensor"
            )


@dataclasses.dataclass
class ExpressionContainer(BaseDataclass):
    batches: list[ExpressionBatch]
