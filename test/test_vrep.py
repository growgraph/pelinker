from pelinker.util import texts_to_vrep
from pelinker.onto import WordGrouping, MAX_LENGTH


def test_vrep_word_groupings(tokenizer_model_scibert, nlp):
    tokenizer, model = tokenizer_model_scibert
    s = "we used a modified functional balance (fb) model to predict growth response of helianthus annuus l. to elevated co2."
    texts = [s]
    layers_spec = [-2, -1]

    report = texts_to_vrep(
        texts,
        tokenizer,
        model,
        layers_spec,
        [WordGrouping.W1, WordGrouping.W2],
        max_length=MAX_LENGTH,
        nlp=nlp,
    )

    wg_w1 = report[WordGrouping.W1].expression_data[0].expressions
    wg_w2 = report[WordGrouping.W2].expression_data[0].expressions

    assert len(wg_w2) + 1 == len(wg_w1)
