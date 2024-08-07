import pytest
import spacy
import pandas as pd
from importlib.resources import files
from pelinker.util import load_models


@pytest.fixture(scope="module")
def df_properties():
    file_path = files("data.derived").joinpath("properties.synthesis.0.csv")
    return pd.read_csv(file_path)


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_trf")


@pytest.fixture(scope="module")
def tokenizer_model_biobert_stsb():
    tokenizer, model = load_models("biobert-stsb", True)
    return tokenizer, model


@pytest.fixture(scope="module")
def tokenizer_model_pubmedbert():
    tokenizer, model = load_models("pubmedbert", False)
    return tokenizer, model


@pytest.fixture(scope="module")
def phrase_vb_0():
    return (
        "TAMs can also secrete in the TME a number of immunosuppressive cytokines, such as IL-6, TGF-β, "
        "and IL-10 that are able to suppress CD8+ T-cell function (76)."
    )


@pytest.fixture(scope="module")
def phrase_vb_1():
    return (
        "Two different subsets of CAFs with diverse functions have been identified in pancreatic cancer: "
        "myofibroblastic (myCAFs), which are located close to the tumor nests "
        "and likely suppress tumor cell growth, and inflammatory CAFs (iCAFs) "
        "which are located more distantly from the tumor nests "
        "and secrete inflammatory factors with pro-tumorigenic functions."
    )
