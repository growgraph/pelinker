import pytest

from pelinker.util import MAX_LENGTH
from pelinker.model import LinkerModel
from importlib.resources import files
import joblib


@pytest.fixture()
def text():
    return """TAMs can also secrete in the TME a number of immunosuppressive cytokines, such as IL-6, TGF-β, and IL-10 
    that are able to suppress CD8+ T-cell function (76). 
    Specifically, IL-6 is expressed at high levels in PDAC, and its increasing circulating level is associated with 
    advanced disease and poor prognosis (77). 
    The inhibition of IL-6 signaling along with CD40 blockade is able to revert the TME to support an antitumor immune 
    response, by reducing TGF-β activation and fibrosis 
    deposition due to a decreased collagen type I production (78). """


def test_load(text, t_model, t_tokenizer, nlp):
    layers = [-6, -5, -4, -3, -2, -1]
    layers_str = LinkerModel.encode_layers(layers)

    file_path = files("pelinker.store").joinpath(
        f"pelinker.model.pubmedbert.{layers_str}.gz"
    )
    model = joblib.load(file_path)
    r = model.link(text, t_tokenizer, t_model, nlp, MAX_LENGTH, False)
    assert r["entities"][-1]["mention"] == "decreased"
