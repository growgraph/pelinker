import dataclasses
import enum

import torch


class WordGrouping(str, enum.Enum):
    VERBAL_STRICT = "verbal_strict"
    VERBAL = "verbal"
    W1 = "1"
    W12 = "12"
    W123 = "123"
    W1234 = "1234"
    SENTENCE = "sentence"


MAX_LENGTH = 512


@dataclasses.dataclass
class ChunkMapper:
    tt: torch.tensor
    flattened_chunks: list[str]
    token_bounds: list[list[tuple[int, int]]]
    it_ic: list[tuple[int, int]]
    chunk_cumlens: list[list[int]]
    char_spans: list[list[tuple[int, int]]] | None = None
    token_spans: list[list[tuple[int, int]]] | None = None
