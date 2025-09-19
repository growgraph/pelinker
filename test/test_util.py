import torch

from pelinker.onto import Expression
from pelinker.util import (
    get_word_boundaries,
    map_spans_to_spans_basic,
    get_vb_spans,
    text_to_tokens,
    map_spans_to_spans,
    split_text_into_batches,
    token_list_with_window,
    SimplifiedToken,
)


def test_vb_span(nlp, phrase_vb_0):
    spans0 = get_vb_spans(nlp, text=phrase_vb_0, extra_context=True)
    assert len(spans0) == 2


def test_text_info(nlp, phrase_vb_0):
    tokens = text_to_tokens(nlp, text=phrase_vb_0)
    assert isinstance(tokens[0], SimplifiedToken)
    assert tokens[1].lemma == "can"


def test_expression(nlp, phrase_vb_0):
    tokens = text_to_tokens(nlp, text=phrase_vb_0)
    e = Expression(tokens=tokens[:3])
    assert e.b == tokens[2].ix_end


def test_window_expressions(nlp, phrase_vb_0):
    tokens = text_to_tokens(nlp, text=phrase_vb_0)
    window1 = token_list_with_window(tokens, 1)
    window2 = token_list_with_window(tokens, 2)
    window3 = token_list_with_window(tokens, 3)
    assert len(window1) == len(tokens)
    assert len(window2) == len(tokens) - 1
    assert len(window3[0].tokens) == 3


def test_sentence_ix(nlp, sentence, token_bounds):
    word_bnds = get_word_boundaries(sentence)
    char_spans, token_spans = map_spans_to_spans(token_bounds, word_bnds)
    phrase = sentence[char_spans[3][0] : char_spans[3][1]]
    assert phrase == "secrete"


def test_sentence_ix2(nlp, sentence, token_bounds):
    word_bnds = get_word_boundaries(sentence)
    ioi = 3
    char_spans_, token_spans_ = map_spans_to_spans(token_bounds, [word_bnds[ioi]])
    char_spans, token_spans = map_spans_to_spans(token_bounds, word_bnds)
    assert token_spans_[0] == token_spans[ioi]


def test_split_long_text(phrase_split):
    r = split_text_into_batches(phrase_split, max_length=100)
    assert "".join(r) == phrase_split
    assert [len(x) for x in r] == [91, 96, 96, 97, 96, 98, 12]


def test_word_bnds(text_short):
    bnds = get_word_boundaries(text_short)
    assert text_short[bnds[0][0] : bnds[0][1]] == "be"
    assert text_short[bnds[-1][0] : bnds[-1][1]] == "aligned"


def test_map_word_indexes_to_token_indexes():
    # source_text = "The title will not"
    ix_words = [(0, 3), (4, 9), (10, 14), (15, 18)]
    token_offsets = torch.tensor(
        [[0, 3], [4, 9], [10, 14], [15, 18], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    )

    miti = map_spans_to_spans_basic(ix_words, token_offsets)
    assert len(miti) == 4
