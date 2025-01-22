from pelinker.onto import WordGrouping
from pelinker.util import (
    process_text,
    split_into_sentences,
    texts_to_vrep,
)


def test_splitting(text):
    sents = split_into_sentences(text)
    assert len(sents) == 2


def test_pro_text(batched_texts, tokenizer_model_pubmedbert, nlp):
    t_tokenizer, t_model = tokenizer_model_pubmedbert
    chunk_mapper = process_text(batched_texts, t_tokenizer, t_model)

    assert len(chunk_mapper.flattened_chunks) == 47


# def test_mapping_table(texts, batched_texts, tokenizer_model_pubmedbert, nlp):
#     t_tokenizer, t_model = tokenizer_model_pubmedbert
#     layers = [-1]
#     flattened_chunks: list[str] = [s for group in batched_texts for s in group]
#
#     # word_bnds = [
#     #     get_vb_spans(nlp=nlp, text=s) for batch in batched_texts for s in batch
#     # ]
#     word_bnds = [get_word_boundaries(s) for batch in batched_texts for s in batch]
#
#     ll_tt_stacked, mapping_table = batched_texts_to_vrep(
#         batched_texts, t_tokenizer, t_model, word_spans=word_bnds, layers_spec=layers
#     )
#
#     for itext, ichunk, (_a, _b), (a, b) in mapping_table:
#         print(itext, ichunk, a, b, texts[itext][a:b], flattened_chunks[ichunk][_a:_b])
#         assert texts[itext][a:b] == flattened_chunks[ichunk][_a:_b]


def test_texts_to_vrep(texts, tokenizer_model_pubmedbert, nlp):
    t_tokenizer, t_model = tokenizer_model_pubmedbert
    layers = [-1]
    report = texts_to_vrep(
        texts,
        t_tokenizer,
        t_model,
        layers_spec=layers,
        word_modes=[WordGrouping.VERBAL_STRICT],
        nlp=nlp,
    )
    wg_vs = report["word_groupings"][WordGrouping.VERBAL_STRICT]
    assert len(wg_vs) == len(texts)
    assert sum([len(x) for x in wg_vs]) == 18


def test_texts_to_vrep_sentence(texts, tokenizer_model_pubmedbert, nlp):
    t_tokenizer, t_model = tokenizer_model_pubmedbert
    layers = [-1]
    report = texts_to_vrep(
        texts,
        t_tokenizer,
        t_model,
        layers_spec=layers,
        word_modes=[WordGrouping.SENTENCE],
        nlp=nlp,
    )
    wg_vs = report["word_groupings"][WordGrouping.SENTENCE]
    assert len(wg_vs) == len(texts)
    assert sum([len(x) for x in wg_vs]) == 10
