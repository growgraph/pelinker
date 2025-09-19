from pelinker.util import process_text, split_into_sentences


def test_splitting(text):
    sents = split_into_sentences(text)
    assert len(sents) == 2


def test_pro_text(batched_texts, tokenizer_model_pubmedbert, nlp):
    t_tokenizer, t_model = tokenizer_model_pubmedbert
    chunk_mapper = process_text(batched_texts, t_tokenizer, t_model)

    assert len(chunk_mapper.flattened_chunks) == 47
