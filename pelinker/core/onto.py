import dataclasses
from enum import Enum
from dataclass_wizard import JSONWizard
import torch
from collections.abc import Iterator

MAX_LENGTH = 512


class WordGrouping(Enum):
    W1 = 1
    W2 = 2
    W3 = 3
    W4 = 4


class BaseDataclass(JSONWizard, JSONWizard.Meta):
    key_transform_with_dump = "SNAKE"
    skip_defaults = True


NEGATIVE_LABEL = "__NEGATIVE__"


@dataclasses.dataclass
class SimplifiedToken(BaseDataclass):
    ix: int
    ix_end: int
    text: str
    lemma: str
    tag: str
    pos: str | None = None  # spaCy ``token.pos_``; ``None`` only for legacy payloads
    is_stop: bool | None = None  # spaCy ``token.is_stop``


@dataclasses.dataclass
class Expression(BaseDataclass):
    tokens: list[SimplifiedToken]
    itext: int | None = None  # index of document
    ichunk: int | None = None  # index of document chunk
    a: int | None = None  # index of the first character
    b: int | None = None  # exclusive end index (Python slice semantics)

    def __post_init__(self):
        self.tokens = sorted(self.tokens, key=lambda x: x.ix)
        self.a = self.tokens[0].ix
        self.b = self.tokens[-1].ix_end


@dataclasses.dataclass
class MentionCandidate(BaseDataclass):
    """Typed mention payload used by :class:`pelinker.model.Linker` predictions."""

    mention: str
    a: int | None
    b: int | None
    a_abs: int | None = None
    b_abs: int | None = None
    itext: int | None = None
    ichunk: int | None = None
    word_grouping: WordGrouping | None = None
    lemma: str = ""


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
        return list(self.iter_on_lemmas(tokens))

    def iter_on_lemmas(
        self, tokens: list[SimplifiedToken]
    ) -> Iterator[tuple[Expression, torch.Tensor]]:
        tokens_lemmatized = " ".join(e.lemma for e in tokens)
        for expression, embedding in zip(self.expressions, self.tt):
            if (
                " ".join(token.lemma for token in expression.tokens)
                == tokens_lemmatized
            ):
                yield expression, embedding


@dataclasses.dataclass
class ExpressionHolderBatch(BaseDataclass):
    expression_data: list[ExpressionHolder]
    word_grouping: WordGrouping | None = None


def _wg_for_property(prop: str) -> WordGrouping | None:
    n = len(prop.split())
    if n in (1, 2, 3, 4):
        return WordGrouping(n)
    return None
