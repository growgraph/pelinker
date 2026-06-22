# pylint: disable=E1120

import os
import re
from pathlib import Path

from collections.abc import Callable
from string import punctuation, whitespace
from typing import List
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from functools import reduce
import logging

from pelinker.onto import (
    MAX_LENGTH,
    NEGATIVE_LABEL,
    ChunkMapper,
    WordGrouping,
    SimplifiedToken,
    Expression,
    ExpressionHolder,
    ExpressionHolderBatch,
    ReportBatch,
    _wg_for_property,
)
from spacy.language import Language

logger = logging.getLogger(__name__)


def expand_config_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Expand environment variables and ``~`` in config or CLI path strings."""
    if path is None:
        return None
    return Path(os.path.expandvars(os.fspath(path))).expanduser()


def load_models(model_type, sentence=False):
    if model_type == "scibert":
        spec = "allenai/scibert_scivocab_cased"
    elif model_type == "biobert":
        spec = "dmis-lab/biobert-base-cased-v1.2"
    elif model_type == "pubmedbert":
        spec = "neuml/pubmedbert-base-embeddings"
    elif model_type == "biobert-stsb":
        spec = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    elif model_type == "bert":
        spec = "google-bert/bert-base-uncased"
    elif model_type == "bluebert":
        spec = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    else:
        raise ValueError(f"{model_type} unsupported")
    if sentence:
        tokenizer, model = None, SentenceTransformer(spec)
    else:
        tokenizer, model = (
            AutoTokenizer.from_pretrained(spec),
            AutoModel.from_pretrained(spec),
        )
    return tokenizer, model


def layers2str(layers: str | list[int]) -> str:  # used by run/obsolete only
    if isinstance(layers, str):
        layers_str = layers
    else:
        if any(l0 > 0 for l0 in layers):
            raise ValueError(f" there are positive layers: {layers}")
        alayers = sorted([abs(l0) for l0 in layers])
        layers_str = "".join([str(l0) for l0 in alayers])
    return layers_str


def normalize_layers_spec(
    layers_spec: str | list[int],
    *,
    n_hidden_states: int | None = None,
) -> list[int]:
    """Parse and validate indices for the stacked ``hidden_states`` tensor.

    String form follows the same convention as historical :func:`str2layers`: each
    digit is a distinct layer counted from the end, e.g. ``\"1\"`` → ``[-1]``,
    ``\"12\"`` → ``[-2, -1]``. Commas in the string are ignored.

    Args:
        layers_spec: Digit-only string or list of **negative** indices (HF convention).
        n_hidden_states: If set (length of first dim of stacked hidden states), indices
            must satisfy ``layer >= -n_hidden_states``.

    Returns:
        Sorted unique negative layer indices.

    Raises:
        ValueError: Empty spec, positive indices, ``\"sent\"``, or out-of-range indices.
    """
    if isinstance(layers_spec, str):
        spec = layers_spec.strip()
        if not spec:
            raise ValueError("layers_spec string is empty")
        if spec == "sent":
            raise ValueError(
                "layers_spec 'sent' is not valid for transformer hidden-state pooling"
            )
        if "," in spec:
            spec = "".join(spec.split(","))
        if not spec.isdigit():
            raise ValueError(
                f"layers_spec string must be digits only (e.g. '1' or '12'), got {layers_spec!r}"
            )
        layers = sorted({-abs(int(ch)) for ch in spec})
    else:
        if not layers_spec:
            raise ValueError("layers_spec list is empty")
        for i, layer in enumerate(layers_spec):
            if layer >= 0:
                raise ValueError(
                    f"layer index must be negative (HF hidden_states convention), got {layer} at position {i}"
                )
        layers = sorted(set(layers_spec))
    if n_hidden_states is not None:
        for layer in layers:
            if layer < -n_hidden_states:
                raise ValueError(
                    f"layer {layer} is out of range for {n_hidden_states} stacked hidden states"
                )
    return layers


def str2layers(layers_spec: str | list[int]) -> list[int]:
    """Parse layer specification; same rules as :func:`normalize_layers_spec`."""
    return normalize_layers_spec(layers_spec, n_hidden_states=None)


def text_to_tokens_embeddings(
    texts: list[str],
    tokenizer,
    model,
    *,
    keep_hidden_states_on_device: bool = False,
):
    """Run the transformer encoder and return hidden states plus character spans per token.

    Encodes ``texts`` without special tokens, pads to a batch, runs ``model`` with
    ``output_hidden_states=True``, and zeroes padded positions using the attention mask.
    By default hidden states are moved to CPU; set ``keep_hidden_states_on_device=True``
    to keep them on the model device (lower host RAM, higher GPU memory use).

    Args:
        texts: Batch of strings to encode (one chunk per row after batching).
        tokenizer: A Hugging Face ``PreTrainedTokenizer`` compatible with ``model``.
        model: A Hugging Face ``PreTrainedModel`` with ``output_hidden_states`` support.
        keep_hidden_states_on_device: If False (default), stack hidden states on CPU.

    Returns:
        Tuple ``(hidden_states, token_char_spans)``. ``hidden_states`` has shape
        ``(n_layers_including_emb, batch, seq_len, hidden_size)`` (stacked model
        ``hidden_states``). ``token_char_spans`` is one list per batch row; each row
        lists ``(start, end)`` character intervals for each non-empty token (empty
        span pairs from ``offset_mapping`` are dropped).
    """

    encoding = tokenizer.batch_encode_plus(
        texts,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    if model.device.type == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    inputs = {
        k: encoding[k].to(device)
        for k in ["input_ids", "token_type_ids", "attention_mask"]
    }

    with torch.inference_mode():
        outputs = model(output_hidden_states=True, **inputs)
        # n_layers x n_batch x n_len x n_emb
        states = [x.detach() for x in outputs.hidden_states]
        if keep_hidden_states_on_device:
            tt = torch.stack(states)
        else:
            tt = torch.stack([x.to("cpu") for x in states])

    # fill with zeros latent vectors for padded tokens
    mask = encoding["attention_mask"]
    if keep_hidden_states_on_device:
        mask = mask.to(tt.device)
    mask = mask.unsqueeze(-1).unsqueeze(0)
    tt = tt.masked_fill(mask.logical_not(), 0)

    # nb x n_len x 2
    offsets = encoding["offset_mapping"]
    enu_offsets = [
        [(x, y) for x, y in sublist if x != y] for sublist in offsets.tolist()
    ]

    return tt, enu_offsets


def map_spans_to_spans_basic(
    words_boundaries, token_boundaries
) -> dict[tuple[int, int], list[int]]:
    """Map each word character span to tokenizer subword indices.

    Both spans use half-open intervals ``[start, end)`` in character offsets. A
    subword token belongs to a word span iff the two intervals have positive overlap.

    Word spans may overlap (sliding W2/W3 windows). A single forward pointer over
    tokens is incorrect in that case; each word span is matched independently.

    Args:
        words_boundaries: Sequence of ``(wa, wb)`` word/window spans.
        token_boundaries: Sequence of ``(ta, tb)`` subword spans covering the chunk.

    Returns:
        Dict ``(wa, wb) -> [token_index, ...]`` in ascending token order.
    """

    map_ix_jx: dict[tuple[int, int], list[int]] = {}
    n_tok = len(token_boundaries)

    for ix_word in words_boundaries:
        wa, wb = ix_word
        hits: list[int] = []
        for j in range(n_tok):
            ta = int(token_boundaries[j][0])
            tb = int(token_boundaries[j][1])
            if tb <= ta:
                continue
            if ta < wb and tb > wa:
                hits.append(j)
        map_ix_jx[ix_word] = hits

    return map_ix_jx


def text_to_tokens(nlp, text) -> list[SimplifiedToken]:
    stokens = [
        SimplifiedToken(
            **{
                "lemma": token.lemma_,
                "text": token.text,
                "tag": token.tag_,
                "pos": token.pos_,
                "is_stop": bool(token.is_stop),
                "ix": token.idx,
                "ix_end": token.idx + len(token),
            }
        )
        for token in nlp(text)
    ]

    return stokens


def keep_expression_for_prediction(expr: Expression) -> bool:
    """Whether to keep a sliding-window mention for :meth:`~pelinker.model.Linker.predict`.

    Drops any window that contains punctuation (spaCy ``pos_ == "PUNCT"``). Drops
    windows whose tokens are **all** stop words; keeps windows that mix content and
    function words (e.g. ``type of``).
    """
    toks = expr.tokens
    if not toks:
        return False
    if any(t.pos == "PUNCT" for t in toks if t.pos is not None):
        return False
    if all(t.is_stop is True for t in toks):
        return False
    return True


def extract_ordered_mention_tensors(
    report_batch: ReportBatch,
    *,
    keep: Callable[[Expression], bool] | None = None,
) -> list[torch.Tensor]:
    """Pool rows for each expression in W1→W2→W3 order, optionally filtering expressions."""
    word_groupings = [WordGrouping.W1, WordGrouping.W2, WordGrouping.W3]
    tt_list: list[torch.Tensor] = []
    for wg in word_groupings:
        if wg not in report_batch.available_groupings():
            continue
        expression_container = report_batch[wg]
        for expr_holder in expression_container.expression_data:
            for expr, tt in zip(expr_holder.expressions, expr_holder.tt):
                if keep is None or keep(expr):
                    tt_list.append(tt)
    return tt_list


def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text)

    # split on .!? if followed by a capital and not preceded by a capital
    pat = r"(?<=[^A-Z][.!?])\s*(?=[A-Z])"
    phrases_ = re.split(pat, text)
    # trim initial/terminal whitespaces
    trim_whitespace = re.compile(r"^[\s+]+|[\s+]+$")
    phrases_ = [trim_whitespace.sub("", p) for p in phrases_]
    return phrases_


def process_text(
    batched_texts: list[list[str]],
    tokenizer,
    model,
    *,
    keep_hidden_states_on_device: bool = False,
) -> ChunkMapper:
    """Encode all text chunks in one forward pass and build a :class:`~pelinker.onto.ChunkMapper`.

    Flattens ``batched_texts`` (each inner list is one logical document split into
    length-limited chunks), runs :func:`text_to_tokens_embeddings` once, and packages
    tensors with bookkeeping to map chunk indices back to ``(document_index,
    chunk_index_within_document)`` and cumulative character offsets within each document.

    Args:
        batched_texts: ``batched_texts[i]`` is the list of chunk strings for document ``i``.
        tokenizer: Hugging Face tokenizer for ``model``.
        model: Hugging Face model used with hidden states.
        keep_hidden_states_on_device: If True, leave activations on the model device
            (see :func:`text_to_tokens_embeddings`).

    Returns:
        A :class:`~pelinker.onto.ChunkMapper` whose ``tensor`` axis ``1`` runs over all
        chunks in flatten order (same order as ``chunks``).
    """

    flattened_chunks: list[str] = [s for group in batched_texts for s in group]

    it_ic = [
        (i_t, i_c) for i_t, sl in enumerate(batched_texts) for i_c, _ in enumerate(sl)
    ]

    chunk_cumlens = [
        reduce(lambda a, x: a + [a[-1] + x], [len(item) for item in chunks], [0])
        for chunks in batched_texts
    ]

    tt, offsets = text_to_tokens_embeddings(
        flattened_chunks,
        tokenizer,
        model,
        keep_hidden_states_on_device=keep_hidden_states_on_device,
    )

    return ChunkMapper(
        tensor=tt,
        chunks=flattened_chunks,
        token_spans_list=offsets,
        it_ic=it_ic,
        cumulative_lens=chunk_cumlens,
    )


def map_words_to_tokens(
    text_token_spans: list[tuple[int, int]], text_word_spans: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    given text token and word spans,

        words : [...(start_pos_i, end_pos_i)...]
        tokens : [...(start_pos_i, end_pos_i)...]

    we define work -> token spans, i.e. a mapping of which tokens belong to words groups

    if there is a positive overlap between word i and token j spans,
        we consider that token j belongs to work i group

    return a list of work -> token bounds : [...(start_token_i, end_token_i)...]
    and a refreshed


    """

    map_ix_jx = map_spans_to_spans_basic(text_word_spans, text_token_spans)

    # sanitize token offsets
    map_ix_jx = {k: v for k, v in map_ix_jx.items() if v}
    text_word_spans = [x for x in map_ix_jx.keys()]
    token_word_spans = [(y[0], y[-1] + 1) for y in map_ix_jx.values()]
    return text_word_spans, token_word_spans


def tt_aggregate_normalize(tt: torch.Tensor, ls):
    """

    :param tt: incoming dims: n_layers x nb x n_tokens x n_emb
    :param ls:
    :return: nb x n_emb
    """

    # average over layers
    tt_norm = tt[ls].mean(0)

    n = tt_norm.norm(dim=-1)
    n[n == 0] = 1

    # normalize each tokens over embedding dim, then average over tokens
    tt_norm = (tt_norm / n.unsqueeze(-1)).mean(dim=1)

    # normalize each tokens over embedding dim, then average over tokens
    tt_norm = tt_norm / tt_norm.norm(dim=-1).unsqueeze(-1)
    return tt_norm


def tt_normalize(cm: ChunkMapper, layers) -> list[list[torch.Tensor]]:
    """Average selected layers, then average token vectors per word span to form word vectors.

    Indexes ``cm.tensor`` with ``layers`` (layer indices on the first dimension),
    averages across those layers and batch/time to get per-token embeddings, then for
    each chunk takes contiguous token ranges from ``cm.token_word_spans_list`` and
    averages those token vectors (mean pooling) to produce one vector per word span.

    Args:
        cm: Chunk mapper with ``token_word_spans_list`` set by
            :meth:`~pelinker.onto.ChunkMapper.set_token_word_spans` /
            :meth:`~pelinker.onto.ChunkMapper.set_mapping_table`.
        layers: Layer indices (typically negative indices into ``hidden_states``, e.g.
            ``[-1]`` or ``[-2, -1]``).

    Returns:
        Outer list is per chunk; inner list is one ``torch.Tensor`` (embedding) per
        word span in that chunk, in order.
    """
    tt_norm = cm.tensor[layers].mean(0)
    tt_r = []
    for js, (ix_tokens, tt_sent) in enumerate(zip(cm.token_word_spans_list, tt_norm)):
        tt_words_list = []
        for a, b in ix_tokens:
            rr = tt_sent[a:b].mean(0)
            # rr = rr / rr.norm(dim=-1).unsqueeze(-1)
            tt_words_list += [rr]
        tt_r += [tt_words_list]
    return tt_r


def compute_distance_ref(  # used by run/obsolete only
    index,
    tt_descs,
    nb_nn,
    gt_position,
):
    distance_matrix, nearest_neighbors_matrix = index.search(tt_descs, nb_nn)
    dfd = pd.DataFrame(distance_matrix[:, 0], columns=["dist"])
    dfd["position"] = "top"
    dfd2 = pd.DataFrame(distance_matrix[:, 1:].flatten(), columns=["dist"])
    dfd2["position"] = "1-9"
    dfa = pd.concat([dfd, dfd2])
    acc_score = accuracy_score(
        nearest_neighbors_matrix[:, 0],
        gt_position,
    )

    top_mean = distance_matrix[:, 0].mean()
    below_mean = distance_matrix[:, 1:].flatten().mean()

    top_cand_dist = top_mean - below_mean
    m0 = (top_cand_dist, acc_score)
    return m0, dfa


def encode(texts, tokenizer, model, ls):  # used by run/obsolete only
    if ls == "sent":
        tt_labels = model.encode(texts, normalize_embeddings=True)
    else:
        tt_labels_layered, labels_spans = text_to_tokens_embeddings(
            texts, tokenizer, model
        )
        layers = normalize_layers_spec(ls, n_hidden_states=tt_labels_layered.shape[0])
        tt_labels = tt_aggregate_normalize(tt_labels_layered, layers)
    return tt_labels


def fetch_latest_kb(
    path_derived,
) -> tuple[str | None, int]:  # used by run/obsolete only
    file_names = [
        file.name
        for file in path_derived.iterdir()
        if file.is_file() and ".synthesis." in file.name
    ]
    filename_versions = sorted(
        [(f, int(f.split(".")[-2])) for f in file_names], key=lambda x: x[1]
    )
    if filename_versions:
        return filename_versions[-1]
    else:
        return None, -1


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


def get_word_boundaries(text) -> list[tuple[int, int]]:
    """
        render word boundaries in text

    :param text:
    :return:
    """

    ix_whitespaces = [0] + [
        i + 1
        for i, char in enumerate(text)
        if char in whitespace or char in punctuation
    ]
    if text[-1] not in whitespace and text[-1] not in punctuation:
        ix_whitespaces += [len(text) + 1]

    ix_words = [(i, j - 1) for i, j in zip(ix_whitespaces, ix_whitespaces[1:])]
    ix_words = [(i, j) for i, j in ix_words if i != j]
    return ix_words


def render_elementary_tensor_table(
    chunk_mapper: ChunkMapper,
    text_word_spans: list[list[tuple[int, int]]],
    layers: list[int],
) -> None:
    """Map spaCy word spans to tokenizer tokens and fill ``chunk_mapper.tt_expressions``.

    Updates ``chunk_mapper`` in place: converts character-level word spans to token
    index ranges, builds the global mapping table, then computes per-span embedding
    rows via :func:`tt_normalize` and stores them in ``chunk_mapper.tt_expressions``
    (one tensor per chunk, rows aligned with surviving word spans).

    Args:
        chunk_mapper: Mapper with encoder hidden states already in ``tensor``.
        text_word_spans: For each chunk, a list of ``(char_start, char_end)`` spans
            (inclusive/exclusive conventions must match tokenizer offset mapping).
        layers: Layer indices forwarded to :func:`tt_normalize`.

    Returns:
        None; mutates ``chunk_mapper``.
    """

    chunk_mapper.set_token_word_spans(text_word_spans)
    chunk_mapper.set_mapping_table()

    ll_tt = tt_normalize(chunk_mapper, layers)
    chunk_mapper.tt_expressions = [
        torch.stack([t for t in sl]) if sl else torch.tensor([]) for sl in ll_tt
    ]


def build_expression_container(
    cm: ChunkMapper,
    expression_lists_per_chunk: list[list[Expression]],
    word_grouping: WordGrouping,
) -> ExpressionHolderBatch:
    """Merge per-chunk expressions and embedding rows into one holder per document.

    For each document index, concatenates embedding tensors for all its chunks (in
    chunk order) and concatenates expression lists the same way, so downstream code
    can match lemmas and spans at document level.

    Args:
        cm: Chunk mapper with ``tt_expressions`` and ``text_chunk_map`` populated.
        expression_lists_per_chunk: Parallel to ``cm.chunks``: expressions for each
            chunk (already filtered to align with ``tt_expressions`` rows where needed).
        word_grouping: Which :class:`~pelinker.onto.WordGrouping` this batch describes.

    Returns:
        An :class:`~pelinker.onto.ExpressionHolderBatch` with one
        :class:`~pelinker.onto.ExpressionHolder` per input document.
    """
    texts = []
    for itext, ichunks in cm.text_chunk_map.items():
        expressions = reduce(
            lambda a, b: a + b, [expression_lists_per_chunk[i] for i in ichunks]
        )
        tt_merged = torch.cat([cm.tt_expressions[i] for i in ichunks])
        texts.append(ExpressionHolder(tt=tt_merged, expressions=expressions))

    return ExpressionHolderBatch(expression_data=texts, word_grouping=word_grouping)


def texts_to_vrep(
    texts: list[str],
    tokenizer,
    model,
    layers_spec: str | list[int],
    word_modes: list[WordGrouping],
    nlp: Language,
    max_length: int = MAX_LENGTH,
    *,
    chunk_by_token_budget: bool = True,
    keep_hidden_states_on_device: bool = False,
) -> ReportBatch:
    """Turn raw texts into encoder-based vector representations for sliding word windows.

    Pipeline (high level):

    1. Split each document into chunks (token-budget by default, see ``chunk_by_token_budget``).
    2. Encode all chunks in one transformer forward pass (:func:`process_text`).
    3. For each chunk, tokenize with spaCy once (:func:`text_to_tokens`) and build sliding
       windows of ``w`` tokens per :class:`~pelinker.onto.WordGrouping`
       (:func:`token_list_with_window`).
    4. Map word character spans to tokenizer token ranges and pool layer activations
       (:func:`render_elementary_tensor_table` → :func:`tt_normalize`).
    5. Drop expressions whose start character was lost when mapping words to tokens,
       then merge chunks per document (:func:`build_expression_container`).

    The same encoder activations are reused for every ``word_modes`` entry; only the
    spaCy windows and pooling targets differ. Each pass over ``word_modes`` calls
    :func:`render_elementary_tensor_table`, so fields on ``chunk_mapper`` such as
    ``tt_expressions`` and ``text_word_spans_list`` reflect **only the last** grouping;
    use the :class:`~pelinker.onto.ReportBatch` slots for per-mode tensors.

    Args:
        texts: One string per document (logical item in the returned batch).
        tokenizer: Hugging Face tokenizer for ``model``.
        model: Transformer model with hidden states (not sentence-transformers ``encode``).
        layers_spec: Layer selection; string digits (``\"12\"``) or negative indices; see
            :func:`normalize_layers_spec`.
        word_modes: For each mode, build ``W1``/``W2``/… token windows and a separate
            :class:`~pelinker.onto.ExpressionHolderBatch` in the result.
        nlp: Loaded spaCy pipeline (:func:`text_to_tokens`).
        max_length: When ``chunk_by_token_budget`` is True, max **subword tokens** per
            chunk; when False, max **characters** (legacy :func:`split_text_into_batches`).
        chunk_by_token_budget: If True (default), split with :func:`split_text_into_token_budget`.
        keep_hidden_states_on_device: If True, keep stacked hidden states on the model
            device (saves host RAM; requires GPU memory for large batches).

    Returns:
        :class:`~pelinker.onto.ReportBatch` containing the shared
        :class:`~pelinker.onto.ChunkMapper` and, for each ``word_grouping``, document-level
        expression holders with pooled embeddings.
    """
    if chunk_by_token_budget:
        batched_texts = [
            split_text_into_token_budget(s, tokenizer, max_tokens=max_length)
            for s in texts
        ]
    else:
        batched_texts = [
            split_text_into_batches(s, max_length=max_length) for s in texts
        ]

    chunk_mapper: ChunkMapper = process_text(
        batched_texts,
        tokenizer,
        model,
        keep_hidden_states_on_device=keep_hidden_states_on_device,
    )

    layers = normalize_layers_spec(
        layers_spec,
        n_hidden_states=chunk_mapper.tensor.shape[0],
    )

    # spaCy once per encoder chunk (reused for every word_modes pass)
    stoken_per_chunk: list[list[SimplifiedToken]] = [
        text_to_tokens(nlp=nlp, text=chunk) for chunk in chunk_mapper.chunks
    ]

    # ichunk -> itext, ichunk local
    ichunk_to_itext_ichunk_local_list = [
        chunk_mapper.ichunk_to_itext_ichunk_local(k)
        for k, _ in enumerate(stoken_per_chunk)
    ]

    data: list[ExpressionHolderBatch] = []
    for word_grouping in word_modes:
        expression_lists_per_chunk: list[list[Expression]] = [
            token_list_with_window(
                chunk_tokens, word_grouping, ichunk=ichunk_local, itext=itext
            )
            for chunk_tokens, (itext, ichunk_local) in zip(
                stoken_per_chunk, ichunk_to_itext_ichunk_local_list
            )
        ]

        word_spans: list[list[tuple[int, int]]] = [
            [(e.a, e.b) for e in batch] for batch in expression_lists_per_chunk
        ]

        render_elementary_tensor_table(chunk_mapper, word_spans, layers)

        # adjust expressions (drops windows with no tokenizer alignment)
        filtered_expression_lists_per_chunk: list[list[Expression]] = []
        for exprs, word_spans in zip(
            expression_lists_per_chunk, chunk_mapper.text_word_spans_list
        ):
            ix_start = {a for a, _ in word_spans}
            kept = [e for e in exprs if e.a in ix_start]
            n_drop = len(exprs) - len(kept)
            if n_drop:
                logger.debug(
                    "texts_to_vrep: dropped %d expressions without token alignment "
                    "(word_grouping=%s)",
                    n_drop,
                    word_grouping.name,
                )
            filtered_expression_lists_per_chunk.append(kept)

        data += [
            build_expression_container(
                chunk_mapper,
                filtered_expression_lists_per_chunk,
                word_grouping=word_grouping,
            )
        ]
    return ReportBatch(chunk_mapper=chunk_mapper, texts=texts, _data=data)


def token_list_with_window(
    tokens: list[SimplifiedToken], window: WordGrouping, itext=None, ichunk=None
) -> list[Expression]:
    """Build every contiguous ``window``-token slice as an :class:`~pelinker.onto.Expression`.

    Each expression stores the participating :class:`~pelinker.onto.SimplifiedToken`
    objects and, after ``__post_init__``, character bounds ``a``/``b`` for the span.

    Args:
        tokens: spaCy-derived tokens for one chunk.
        window: Window size via :class:`~pelinker.onto.WordGrouping` (``W1`` → 1 token, etc.).
        itext: Document index in the outer batch (optional metadata on expressions).
        ichunk: Chunk index within the document (optional metadata).

    Returns:
        Length ``len(tokens) - w + 1`` list of expressions (empty if ``len(tokens) < w``).
    """
    agg = []
    w = int(window.value)
    for k in range(len(tokens) - w + 1):
        agg.append(Expression(tokens=tokens[k : k + w], itext=itext, ichunk=ichunk))
    return agg


def map_words_to_tokens_list(
    text_token_spans_list: list[list[tuple[int, int]]],
    text_word_spans_list: list[list[tuple[int, int]]],
):
    """
    take a batch of token spans and a batch of word spans

    return a batch of work to token maps
        and also an updated batch of word spans
            (some words can not be mapped to tokens, so they are excluded)


    """

    text_word_spans_list_: list[list[tuple[int, int]]] = []
    token_work_spans_list: list[list[tuple[int, int]]] = []

    for text_token_spans, text_word_spans in zip(
        text_token_spans_list, text_word_spans_list
    ):
        text_word_spans, token_word_spans = map_words_to_tokens(
            text_token_spans, text_word_spans
        )
        text_word_spans_list_ += [text_word_spans]
        token_work_spans_list += [token_word_spans]

    return token_work_spans_list, text_word_spans_list_


def embed_texts(
    phrases: list[str],
    tokenizer,
    model,
    layers: str | list[int],
    nlp: Language,
) -> list[torch.Tensor]:
    """
    Embed a list of text phrases using texts_to_vrep.

    Args:
        phrases: List of text phrases to embed
        tokenizer: Tokenizer for the model
        model: Model for embedding
        layers: Layer specification
        nlp: spaCy ``Language`` pipeline (required for tokenization in ``texts_to_vrep``)

    Returns:
        List of embedding tensors, one per phrase
    """
    from pelinker.onto import WordGrouping

    phrases_list = [str(p).strip() for p in phrases if pd.notna(p) and str(p).strip()]

    if not phrases_list:
        return []

    # Use texts_to_vrep for embedding
    report = texts_to_vrep(
        phrases_list,
        tokenizer,
        model,
        layers,
        [WordGrouping.W1],
        nlp,
    )

    # Extract text-level embeddings using the new method
    text_embeddings = report.get_text_embeddings(layers)

    return text_embeddings


def _available_word_groupings(
    report_batch: ReportBatch,
) -> tuple[frozenset[WordGrouping], tuple[WordGrouping, ...]]:
    """Membership set plus deterministic iteration order for negative-candidate walks."""
    members = frozenset(report_batch.available_groupings())
    ordered = tuple(sorted(members, key=lambda wg: wg.value))
    return members, ordered


def _sample_row_indices(rng: np.random.RandomState, n_pool: int, k: int) -> np.ndarray:
    """Sample ``k`` indices in ``[0, n_pool)``; with replacement only when ``k > n_pool``."""
    if k <= n_pool:
        return rng.choice(n_pool, size=k, replace=False)
    return rng.choice(n_pool, size=k, replace=True)


def _mention_key(
    wg_value: int,
    ichunk: int,
    a: int,
    b: int,
    mention: str,
) -> tuple[int, int, int, int, str]:
    return (wg_value, ichunk, a, b, mention.casefold())


def _mention_row_dict(
    *,
    pmid: str,
    entity: str,
    mention: str,
    a: int,
    b: int,
    a_abs: int,
    b_abs: int,
    itext: int,
    ichunk: int,
    embed_list: list[float],
) -> dict[str, object]:
    return {
        "pmid": pmid,
        "entity": entity,
        "mention": mention,
        "a": a,
        "b": b,
        "a_abs": a_abs,
        "b_abs": b_abs,
        "itext": itext,
        "ichunk": ichunk,
        "embed": embed_list,
    }


def extract_and_embed_mentions(
    entities: list[str],
    data: list[str],
    pmids: list[str],
    tokenizer,
    model,
    nlp,
    layers,
    batch_size,
    word_modes=(WordGrouping.W1, WordGrouping.W2, WordGrouping.W3),
    negatives_per_positive: float = 0.0,
    negative_label: str = NEGATIVE_LABEL,
    random_seed: int | None = None,
    negative_random_state: np.random.RandomState | None = None,
    on_encoder_batch: Callable[[int, int, int], None] | None = None,
) -> List[dict]:
    """
    Modified to return list of dicts instead of DataFrame for better memory management
    and consistent schema handling.

    Negative rows are sampled with :class:`numpy.random.RandomState`. Pass
    ``negative_random_state`` to reuse one RNG across several calls (e.g. successive
    read buffers); otherwise ``random_seed`` builds a fresh ``RandomState`` per call.

    If ``on_encoder_batch`` is set, it is invoked after each encoder mini-batch with
    ``(batch_index_0based, n_batches, n_mention_rows_accumulated)``.
    """
    if negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must be >= 0")
    if not negative_label:
        raise ValueError("negative_label must be a non-empty string")

    negative_sampler: np.random.RandomState | None = None
    if negatives_per_positive > 0:
        negative_sampler = (
            negative_random_state
            if negative_random_state is not None
            else np.random.RandomState(random_seed)
        )

    data_pmids = pmids

    data_batched = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    data_pmids_batched = [
        data_pmids[i : i + batch_size] for i in range(0, len(data), batch_size)
    ]

    # Pre-tokenize entities and resolve each entity's matching grouping once.
    entity_specs: list[tuple[str, list[SimplifiedToken], WordGrouping]] = []
    for entity in entities:
        wg = _wg_for_property(entity)
        if wg is None:
            continue
        entity_specs.append((entity, text_to_tokens(nlp=nlp, text=entity), wg))

    rows = []
    n_batches = len(data_batched)
    for ibatch, text_batch in enumerate(data_batched):
        report_batch = texts_to_vrep(
            text_batch,
            tokenizer,
            model,
            layers,
            list(word_modes),
            nlp,
        )

        batch_pmids = data_pmids_batched[ibatch]
        wg_members, wg_iter_order = _available_word_groupings(report_batch)
        containers_by_wg = {wg: report_batch[wg] for wg in wg_iter_order}

        positive_rows_by_text: list[list[dict]] = [[] for _ in report_batch.texts]
        positive_keys_by_text: list[set[tuple]] = [set() for _ in report_batch.texts]

        # Phase A: discover all positive matches across all entities.
        for entity, entity_tokens, wg in entity_specs:
            if wg not in wg_members:
                continue

            expression_container = containers_by_wg[wg]
            for itext, (text, expr_holder) in enumerate(
                zip(report_batch.texts, expression_container.expression_data)
            ):
                for e, tt in expr_holder.iter_on_lemmas(entity_tokens):
                    offset = report_batch.chunk_mapper.map_chunk_to_text(
                        e.itext, e.ichunk
                    )
                    mention = text[offset + e.a : offset + e.b]
                    mention_key = _mention_key(wg.value, e.ichunk, e.a, e.b, mention)
                    positive_keys_by_text[itext].add(mention_key)
                    positive_rows_by_text[itext].append(
                        _mention_row_dict(
                            pmid=batch_pmids[itext],
                            entity=entity,
                            mention=mention,
                            a=e.a,
                            b=e.b,
                            a_abs=offset + e.a,
                            b_abs=offset + e.b,
                            itext=e.itext,
                            ichunk=e.ichunk,
                            embed_list=tt.numpy().tolist(),
                        )
                    )

        # Phase B: emit positives and optionally add globally-negative samples.
        if negatives_per_positive == 0:
            for text_rows in positive_rows_by_text:
                rows.extend(text_rows)
        else:
            assert negative_sampler is not None
            for itext, text in enumerate(report_batch.texts):
                rows.extend(positive_rows_by_text[itext])
                n_positives = len(positive_rows_by_text[itext])
                n_negatives = int(round(n_positives * negatives_per_positive))
                if n_negatives <= 0:
                    continue

                # Build all candidate mentions in this text across available groupings.
                negative_candidates: dict[tuple, dict] = {}
                pos_keys = positive_keys_by_text[itext]
                for wg in wg_iter_order:
                    expr_holder = containers_by_wg[wg].expression_data[itext]
                    for expression, embedding in zip(
                        expr_holder.expressions, expr_holder.tt
                    ):
                        offset = report_batch.chunk_mapper.map_chunk_to_text(
                            expression.itext, expression.ichunk
                        )
                        mention = text[offset + expression.a : offset + expression.b]
                        mention_key = _mention_key(
                            wg.value,
                            expression.ichunk,
                            expression.a,
                            expression.b,
                            mention,
                        )
                        if mention_key in pos_keys:
                            continue
                        if mention_key not in negative_candidates:
                            negative_candidates[mention_key] = _mention_row_dict(
                                pmid=batch_pmids[itext],
                                entity=negative_label,
                                mention=mention,
                                a=expression.a,
                                b=expression.b,
                                a_abs=offset + expression.a,
                                b_abs=offset + expression.b,
                                itext=expression.itext,
                                ichunk=expression.ichunk,
                                embed_list=embedding.numpy().tolist(),
                            )

                if not negative_candidates:
                    continue

                candidate_values = list(negative_candidates.values())
                n_pool = len(candidate_values)
                idx = _sample_row_indices(negative_sampler, n_pool, n_negatives)
                rows.extend(candidate_values[i] for i in idx)

        if on_encoder_batch is not None:
            on_encoder_batch(ibatch, n_batches, len(rows))

    return rows
