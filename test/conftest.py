import pytest
import spacy
import pandas as pd
from importlib.resources import files
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="module")
def df_properties():
    file_path = files("data.derived").joinpath("properties.synthesis.csv")
    return pd.read_csv(file_path)


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


@pytest.fixture(scope="module")
def s_model(pubmedbert):
    return SentenceTransformer.from_pretrained(pubmedbert)
