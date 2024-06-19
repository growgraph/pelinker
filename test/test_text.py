import pytest

from pelinker.util import process_text, split_into_sentences, MAX_LENGTH


@pytest.fixture()
def text():
    return """The title will not be center aligned with the subplot titles. 
        To set the position of the title you can use plt.suptitle("Title", x=center)."""


def test_splitting(text):
    sents = split_into_sentences(text)
    assert len(sents) == 2


def test_pro_text(text, t_model, t_tokenizer, nlp):
    sents, spans, _ = process_text(text, t_tokenizer, t_model, nlp)

    a, b = spans[0][0][0]
    assert sents[0][a:b] == "be"
    a, b = spans[1][1][0]
    assert sents[1][a:b] == "use"

    sents, spans, _ = process_text(text, t_tokenizer, t_model, nlp, MAX_LENGTH)

    assert len(sents) == 1
    a, b = spans[0][0][0]
    assert sents[0][a:b] == "be"
    a, b = spans[0][3][0]
    assert sents[0][a:b] == "use"
