import torch

from pelinker.onto import (
    ChunkMapper,
    Expression,
    ExpressionHolder,
    ExpressionHolderBatch,
    NEGATIVE_LABEL,
    WordGrouping,
)
from pelinker.util import (
    get_word_boundaries,
    keep_expression_for_prediction,
    map_spans_to_spans_basic,
    text_to_tokens,
    map_words_to_tokens,
    split_text_into_batches,
    token_list_with_window,
    SimplifiedToken,
    extract_and_embed_mentions,
)


def _st(
    ix: int,
    text: str,
    *,
    lemma: str | None = None,
    tag: str = "NN",
    pos: str | None = "NOUN",
    is_stop: bool | None = False,
) -> SimplifiedToken:
    lem = lemma if lemma is not None else text.lower()
    return SimplifiedToken(
        ix=ix,
        ix_end=ix + len(text),
        text=text,
        lemma=lem,
        tag=tag,
        pos=pos,
        is_stop=is_stop,
    )


def test_keep_expression_for_prediction_drops_punctuation_or_all_stop():
    keep = keep_expression_for_prediction

    assert not keep(
        Expression(tokens=[_st(0, "the", tag="DT", pos="DET", is_stop=True)])
    )
    assert not keep(
        Expression(
            tokens=[
                _st(0, "of", tag="IN", pos="ADP", is_stop=True),
                _st(3, "the", tag="DT", pos="DET", is_stop=True),
            ]
        )
    )
    assert keep(
        Expression(
            tokens=[
                _st(0, "type", pos="NOUN", is_stop=False),
                _st(5, "of", tag="IN", pos="ADP", is_stop=True),
            ]
        )
    )
    assert not keep(
        Expression(
            tokens=[
                _st(0, "x", pos="NOUN", is_stop=False),
                _st(2, ",", tag=",", pos="PUNCT", is_stop=False),
            ]
        )
    )


def test_text_info(nlp, phrase_vb_0):
    tokens = text_to_tokens(nlp, text=phrase_vb_0)
    assert isinstance(tokens[0], SimplifiedToken)
    assert tokens[1].lemma == "be"


def test_expression(nlp, phrase_vb_0):
    tokens = text_to_tokens(nlp, text=phrase_vb_0)
    e = Expression(tokens=tokens[:3])
    assert e.b == tokens[2].ix_end


def test_window_expressions(nlp, phrase_vb_0):
    tokens = text_to_tokens(nlp, text=phrase_vb_0)
    window1 = token_list_with_window(tokens, WordGrouping.W1)
    window2 = token_list_with_window(tokens, WordGrouping.W2)
    window3 = token_list_with_window(tokens, WordGrouping.W3)
    assert len(window1) == len(tokens)
    assert len(window2) == len(tokens) - 1
    assert len(window3[0].tokens) == 3


def test_sentence_ix(nlp, sentence, token_bounds):
    word_bnds = get_word_boundaries(sentence)
    char_spans, token_spans = map_words_to_tokens(token_bounds, word_bnds)
    phrase = sentence[char_spans[3][0] : char_spans[3][1]]
    assert phrase == "secrete"


def test_sentence_ix2(nlp, sentence, token_bounds):
    word_bnds = get_word_boundaries(sentence)
    ioi = 3
    char_spans_, token_spans_ = map_words_to_tokens(token_bounds, [word_bnds[ioi]])
    char_spans, token_spans = map_words_to_tokens(token_bounds, word_bnds)
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


class _FakeChunkMapper:
    @staticmethod
    def map_chunk_to_text(_itext, _ichunk):
        return 0


class _FakeReport:
    def __init__(self, texts, by_wg):
        self.texts = texts
        self._by_wg = by_wg
        self.chunk_mapper = _FakeChunkMapper()

    def available_groupings(self):
        return list(self._by_wg.keys())

    def __getitem__(self, wg):
        return self._by_wg[wg]


def _expr(
    text: str, lemma: str, *, ix: int, itext: int = 0, ichunk: int = 0
) -> Expression:
    tok = SimplifiedToken(
        ix=ix,
        ix_end=ix + len(text),
        text=text,
        lemma=lemma,
        tag="NN",
        pos="NOUN",
        is_stop=False,
    )
    return Expression(tokens=[tok], itext=itext, ichunk=ichunk)


def _fake_report_for_negatives():
    e_alpha = _expr("alpha", "alpha", ix=0)
    e_beta = _expr("beta", "beta", ix=6)
    e_gamma = _expr("gamma", "gamma", ix=11)
    holder = ExpressionHolder(
        expressions=[e_alpha, e_beta, e_gamma],
        tt=torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
    )
    batch = ExpressionHolderBatch(
        expression_data=[holder], word_grouping=WordGrouping.W1
    )
    return _FakeReport(texts=["alpha beta gamma"], by_wg={WordGrouping.W1: batch})


def test_extract_mentions_negative_sampling_is_deterministic(monkeypatch):
    monkeypatch.setattr(
        "pelinker.util.texts_to_vrep",
        lambda *args, **kwargs: _fake_report_for_negatives(),
    )
    monkeypatch.setattr(
        "pelinker.util.text_to_tokens",
        lambda nlp, text: [_st(0, text, lemma=text.lower())],
    )
    rows_a = extract_and_embed_mentions(
        entities=["alpha", "beta"],
        data=["alpha beta gamma"],
        pmids=["p1"],
        tokenizer=None,
        model=None,
        nlp=None,
        layers="1",
        batch_size=1,
        negatives_per_positive=1.0,
        random_seed=7,
    )
    rows_b = extract_and_embed_mentions(
        entities=["alpha", "beta"],
        data=["alpha beta gamma"],
        pmids=["p1"],
        tokenizer=None,
        model=None,
        nlp=None,
        layers="1",
        batch_size=1,
        negatives_per_positive=1.0,
        random_seed=7,
    )
    assert rows_a == rows_b
    assert any(r["entity"] == NEGATIVE_LABEL for r in rows_a)


def test_extract_mentions_negatives_are_global_not_per_entity(monkeypatch):
    monkeypatch.setattr(
        "pelinker.util.texts_to_vrep",
        lambda *args, **kwargs: _fake_report_for_negatives(),
    )
    monkeypatch.setattr(
        "pelinker.util.text_to_tokens",
        lambda nlp, text: [_st(0, text, lemma=text.lower())],
    )
    rows = extract_and_embed_mentions(
        entities=["beta", "alpha"],
        data=["alpha beta gamma"],
        pmids=["p1"],
        tokenizer=None,
        model=None,
        nlp=None,
        layers="1",
        batch_size=1,
        negatives_per_positive=1.0,
        random_seed=3,
    )
    positive_mentions = {r["mention"] for r in rows if r["entity"] != NEGATIVE_LABEL}
    negative_mentions = {r["mention"] for r in rows if r["entity"] == NEGATIVE_LABEL}
    assert "alpha" in positive_mentions and "beta" in positive_mentions
    assert not (positive_mentions & negative_mentions)


def test_chunk_mapper_absolute_end_offset_is_exclusive_for_last_word() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda omega"
    chunks = [
        "alpha beta ",
        "gamma delta ",
        "epsilon zeta ",
        "eta theta ",
        "iota kappa ",
        "lambda omega",
    ]
    assert "".join(chunks) == text

    cm = ChunkMapper(
        tensor=torch.zeros((1, len(chunks), 1, 1), dtype=torch.float32),
        chunks=chunks,
        token_spans_list=[[] for _ in chunks],
        it_ic=[(0, i) for i in range(len(chunks))],
        cumulative_lens=[[0, 11, 23, 36, 46, 57, 69]],
    )

    chunk_local_start = chunks[-1].index("omega")
    chunk_local_end_exclusive = chunk_local_start + len("omega")
    abs_start = cm.map_chunk_to_text(0, 5, chunk_local_start)
    abs_end_exclusive = cm.map_chunk_to_text(0, 5, chunk_local_end_exclusive)

    assert text[abs_start:abs_end_exclusive] == "omega"
    assert abs_end_exclusive - 1 == text.rindex("omega") + len("omega") - 1
