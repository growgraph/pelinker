import torch

from pelinker.util import texts_to_vrep, batched_texts_to_vrep
from pelinker.matching import match_pattern
from pelinker.onto import WordGrouping, MAX_LENGTH


def test_vrep_pipeline(tokenizer_model_scibert, nlp):
    tokenizer, model = tokenizer_model_scibert
    s = "we used a modified functional balance (fb) model to predict growth response of helianthus annuus l. to elevated co2."
    texts = [s]
    i_model = 7
    indexes_of_interest = [match_pattern("model", x) for x in texts]
    layers_spec = [-2, -1]
    word_mode = WordGrouping.W1

    ll_tt_stacked, mapping_table = batched_texts_to_vrep(
        [texts], tokenizer, model, indexes_of_interest, layers_spec
    )

    embs = texts_to_vrep(
        texts, tokenizer, model, layers_spec, word_mode, max_length=MAX_LENGTH, nlp=nlp
    )

    assert torch.max(ll_tt_stacked[0] - embs["tensor"][i_model]) < 1e-10
