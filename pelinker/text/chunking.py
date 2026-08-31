"""Split long documents into encoder-sized chunks."""

import dataclasses
import re
from collections import defaultdict
from dataclasses import field

import torch

from pelinker.core.onto import BaseDataclass
from pelinker.text.tokenize import map_words_to_tokens_list


def split_text_into_batches(text: str, max_length: int) -> list[str]:
    """Split a single string into chunks no longer than ``max_length`` characters.

    Uses a regex that prefers breaking after whitespace near the limit; a chunk may
    reach ``max_length`` when no earlier break exists.

    Args:
        text: Full document or segment to split.
        max_length: Maximum characters per chunk (typically tokenizer/model limit).

    Returns:
        Non-empty string segments whose concatenation recovers ``text`` (aside from
        regex edge cases on pathological input).
    """
    pattern = (
        r"(.{1," + str(max_length - 1) + r"})(\s|$)|(.{1," + str(max_length) + r"})"
    )

    matches = re.findall(pattern, text)
    batched = ["".join(parts) for parts in matches]
    return batched


def split_text_into_token_budget(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split *text* so each segment encodes to at most *max_tokens* subword tokens.

    Uses a longest-prefix binary search per segment (by character offset), then
    prefers breaking at the last space still within the token budget. Avoids relying
    on a fixed character cap, which can exceed the model's tokenizer limit.

    Args:
        text: Full document string for one logical chunking pass.
        tokenizer: Hugging Face tokenizer (``encode(..., add_special_tokens=False)``).
        max_tokens: Maximum subword count per segment (typically ``MAX_LENGTH``).

    Returns:
        Segments whose concatenation equals *text* exactly (no dropped characters).

    Raises:
        ValueError: If ``max_tokens < 1``, or a minimal slice still exceeds the budget.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not text:
        return []

    def n_tokens(segment: str) -> int:
        return len(tokenizer.encode(segment, add_special_tokens=False))

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        if n_tokens(text[start:n]) <= max_tokens:
            chunks.append(text[start:n])
            break

        lo, hi = start + 1, n + 1
        best = start
        while lo < hi:
            mid = (lo + hi) // 2
            if n_tokens(text[start:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid
        if best <= start:
            raise ValueError(
                "a minimal text slice exceeds max_tokens; increase max_length or inspect the tokenizer"
            )

        end = best
        sp = text.rfind(" ", start + 1, end)
        if sp > start and n_tokens(text[start:sp]) <= max_tokens:
            end = sp

        chunk = text[start:end]
        chunks.append(chunk)
        start = end

    return chunks


@dataclasses.dataclass
class ChunkMapper(BaseDataclass):
    """Maps encoder chunks back to documents and optional pooled span rows.

    When :func:`pelinker.text.embed.texts_to_vrep` runs multiple ``word_modes``, it calls
    :func:`pelinker.text.embed.render_elementary_tensor_table` repeatedly on the **same**
    instance. Fields ``text_word_spans_list``, ``token_word_spans_list``,
    ``tt_expressions``, and ``mapping_table`` therefore reflect **only the last**
    grouping pass; read per-mode results from :class:`ReportBatch` instead.
    """

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
