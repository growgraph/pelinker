import pytest
import spacy
from transformers import AutoModel, AutoTokenizer


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def pubmedbert():
    return "neuml/pubmedbert-base-embeddings"


@pytest.fixture(scope="module")
def t_tokenizer(pubmedbert):
    return AutoTokenizer.from_pretrained(pubmedbert)


@pytest.fixture(scope="module")
def t_model(pubmedbert):
    return AutoModel.from_pretrained(pubmedbert)
