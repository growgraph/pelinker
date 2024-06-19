import pytest
import spacy
from transformers import AutoModel, AutoTokenizer


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def t_tokenizer():
    return AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")


@pytest.fixture(scope="module")
def t_model():
    return AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
