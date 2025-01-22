import dataclasses
import enum

import torch


class WordGrouping(str, enum.Enum):
    VERBAL_STRICT = "verbal_strict"
    VERBAL = "verbal"
    W1 = "1"
    W2 = "2"
    W3 = "3"
    W4 = "4"
    SENTENCE = "sentence"


MAX_LENGTH = 512


@dataclasses.dataclass
class ChunkMapper:
    # n_layers x n_batch x n_len x n_emb
    tt: torch.tensor
    flattened_chunks: list[str]
    token_bounds: list[list[tuple[int, int]]]
    it_ic: list[tuple[int, int]]
    chunk_cumlens: list[list[int]]
    char_spans: list[list[tuple[int, int]]] | None = None
    token_spans: list[list[tuple[int, int]]] | None = None
