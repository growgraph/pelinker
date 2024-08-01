from pelinker.util import get_vb_spans, aggregate_token_groups


def test_vb_span(nlp, phrase_vb_0, phrase_vb_1):
    spans0 = get_vb_spans(nlp, text=phrase_vb_0, extra_context=True)
    assert len(spans0) == 2


def test_aggregate_token_groups(nlp, phrase_vb_0, phrase_vb_1):
    groups = aggregate_token_groups(nlp, text=phrase_vb_1, extra_context=True)
    # in phrase
    # `which are located close to the tumor nests and likely suppress tumor cell growth, and inflammatory CAFs (iCAFs)`
    # suppress is classified as {pos:JJ, dep:conj} in case small spacy model is used

    assert len(groups) == 5
    assert any(any("supp" in t.text for t in group) for group in groups)
