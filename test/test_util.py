import pytest
import torch

from pelinker.util import (
    get_word_boundaries,
    map_word_indexes_to_token_indexes,
    get_vb_spans,
    map_char_spans_2_token_spans,
    split_text_into_batches,
)


@pytest.fixture
def sentence():
    return "TAMs can also secrete in the TME a number of immunosuppressive cytokines, such as IL-6, TGF-β, and IL-10 that are able to suppress CD8+ T-cell function."


@pytest.fixture
def token_bounds() -> list[tuple[int, int]]:
    bnds = [
        [0, 4],
        [5, 8],
        [9, 13],
        [14, 21],
        [22, 24],
        [25, 28],
        [29, 32],
        [33, 34],
        [35, 41],
        [42, 44],
        [45, 62],
        [63, 72],
        [72, 73],
        [74, 78],
        [79, 81],
        [82, 84],
        [84, 85],
        [85, 86],
        [86, 87],
        [88, 91],
        [91, 92],
        [92, 93],
        [93, 94],
        [95, 98],
        [99, 101],
        [101, 102],
        [102, 104],
        [105, 109],
        [110, 113],
        [114, 118],
        [119, 121],
        [122, 130],
        [131, 134],
        [134, 135],
        [136, 137],
        [137, 138],
        [138, 142],
        [143, 151],
        [151, 152],
    ]
    return [tuple(item) for item in bnds]


def test_vb_span(nlp, phrase_vb_0, phrase_vb_1):
    spans0 = get_vb_spans(nlp, text=phrase_vb_0, extra_context=True)
    assert len(spans0) == 2


def test_sentence_ix(nlp, sentence, token_bounds):
    word_bnds = get_word_boundaries(sentence)
    char_spans, token_spans = map_char_spans_2_token_spans(token_bounds, word_bnds)
    phrase = sentence[char_spans[3][0] : char_spans[3][1]]
    assert phrase == "secrete"


def test_sentence_ix2(nlp, sentence, token_bounds):
    word_bnds = get_word_boundaries(sentence)
    ioi = 3
    char_spans_, token_spans_ = map_char_spans_2_token_spans(
        token_bounds, [word_bnds[ioi]]
    )
    char_spans, token_spans = map_char_spans_2_token_spans(token_bounds, word_bnds)
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

    miti = map_word_indexes_to_token_indexes(ix_words, token_offsets)
    assert len(miti) == 4
