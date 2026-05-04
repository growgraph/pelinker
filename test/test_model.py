import pytest

from pelinker.onto import MAX_LENGTH
from pelinker.model import Linker
from importlib.resources import files


@pytest.fixture()
def text():
    return """TAMs can also secrete in the TME a number of immunosuppressive cytokines, such as IL-6, TGF-β, and IL-10 
    that are able to suppress CD8+ T-cell function (76). 
    Specifically, IL-6 is expressed at high levels in PDAC, and its increasing circulating level is associated with 
    advanced disease and poor prognosis (77). 
    The inhibition of IL-6 signaling along with CD40 blockade is able to revert the TME to support an antitumor immune 
    response, by reducing TGF-β activation and fibrosis 
    deposition due to a decreased collagen type I production (78). """


@pytest.mark.skip("fix later")
def test_load(text):
    layer_spec = "sent"
    layers = Linker.str2layers(layer_spec)

    layers_str = Linker.layers2str(layers)

    file_path = files("pelinker.store").joinpath(
        f"pelinker.model.biobert-stsb.{layers_str}"
    )
    model = Linker.load(file_path)

    r = model.predict([text], max_length=MAX_LENGTH, threshold=1.0)
    assert r.entities[-1]["mention"] == "decreased"
