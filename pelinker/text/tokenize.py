import re

from string import punctuation, whitespace

from pelinker.core.onto import (
    Expression,
    SimplifiedToken,
    WordGrouping,
)

"""spaCy tokenization, sentence splitting, and word/subword span mapping."""


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


def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text)

    # split on .!? if followed by a capital and not preceded by a capital
    pat = r"(?<=[^A-Z][.!?])\s*(?=[A-Z])"
    phrases_ = re.split(pat, text)
    # trim initial/terminal whitespaces
    trim_whitespace = re.compile(r"^[\s+]+|[\s+]+$")
    phrases_ = [trim_whitespace.sub("", p) for p in phrases_]
    return phrases_


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


def token_list_with_window(
    tokens: list[SimplifiedToken], window: WordGrouping, itext=None, ichunk=None
) -> list[Expression]:
    """Build every contiguous ``window``-token slice as an :class:`~pelinker.core.onto.Expression`.

    Each expression stores the participating :class:`~pelinker.core.onto.SimplifiedToken`
    objects and, after ``__post_init__``, character bounds ``a``/``b`` for the span.

    Args:
        tokens: spaCy-derived tokens for one chunk.
        window: Window size via :class:`~pelinker.core.onto.WordGrouping` (``W1`` → 1 token, etc.).
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
