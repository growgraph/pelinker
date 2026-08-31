import pytest
import spacy
import pandas as pd
from importlib.resources import files
from pelinker.text.chunking import split_text_into_batches
from pelinker.text.models import load_models

SPACY_MODEL = "en_core_web_lg"


def _load_spacy_or_skip(name: str):
    """Skip rather than error when the pipeline is not installed locally / in CI.

    ``en_core_web_lg`` is a large out-of-band download (``spacy download``), so a fresh
    checkout would otherwise report errors that say nothing about the code under test.
    Install it to run these for real; see the ``heavy`` marker in ``pyproject.toml``.
    """
    try:
        return spacy.load(name)
    except OSError:
        pytest.skip(
            f"spaCy model {name!r} not installed (run: uv run spacy download {name})"
        )


def _load_hf_or_skip(model_type: str, flag: bool):
    """Skip rather than error when HF weights cannot be fetched (offline CI, no cache)."""
    try:
        return load_models(model_type, flag)
    except Exception as exc:  # network, auth, cache miss — all mean "cannot run here"
        pytest.skip(f"transformer weights for {model_type!r} unavailable: {exc}")


@pytest.fixture(scope="module")
def df_properties():
    file_path = files("data.derived").joinpath("properties.synthesis.0.csv")
    return pd.read_csv(file_path)


@pytest.fixture(scope="module")
def nlp():
    return _load_spacy_or_skip(SPACY_MODEL)


@pytest.fixture(scope="module")
def tokenizer_model_biobert_stsb():
    return _load_hf_or_skip("biobert-stsb", True)


@pytest.fixture(scope="module")
def tokenizer_model_pubmedbert():
    return _load_hf_or_skip("pubmedbert", False)


@pytest.fixture(scope="module")
def tokenizer_model_scibert():
    return _load_hf_or_skip("scibert", False)


@pytest.fixture(scope="module")
def phrase_vb_0():
    return (
        "TAMs was also secreting in the TME a number of immunosuppressive cytokines, such as IL-6, TGF-β, "
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


@pytest.fixture(scope="module")
def text_short():
    return "be center aligned"


@pytest.fixture(scope="module")
def phrase_split():
    return (
        "Biofilm-related infections are associated with high mortality and morbidity, combined with increased "
        "hospital stays and overall treatment costs. Traditional antibiotics play an essential role "
        "in controlling biofilms; however, they are becoming less effective due to the "
        "emergence of drug-resistant bacterial strains. The need to treat "
        "biofilms on medical implants is particularly acute, and one persistent challenge is "
        "selectively directing nanoparticles to the biofilm site. "
        "Here, we present a protein-based functionalization strategy "
        "that targets the extracellular matrix of biofilms."
    )


@pytest.fixture()
def text():
    return """The title will not be center aligned with the subplot titles. 
        To set the position of the title you can use plt.suptitle("Title", x=center)."""


@pytest.fixture()
def text2():
    return """Biofilm-related infections are associated with high mortality and morbidity, 
    combined with increased hospital stays and overall treatment costs. 
    Traditional antibiotics play an essential role in controlling biofilms; 
    however, they are becoming less effective due to the emergence of drug-resistant bacterial strains. 
    The need to treat biofilms on medical implants is particularly acute, and one persistent challenge is 
    selectively directing nanoparticles to the biofilm site. 
    Here, we present a protein-based functionalization strategy 
    that targets the extracellular matrix of biofilms."""


@pytest.fixture()
def texts(text, text2):
    return [text, text2]


@pytest.fixture()
def batched_texts(texts):
    return [split_text_into_batches(s, max_length=20) for s in texts]


@pytest.fixture
def sentence():
    return "TAMs can also secrete in the TME a number of immunosuppressive cytokines, such as IL-6, TGF-β, and IL-10 that are able to suppress CD8+ T-cell function."


@pytest.fixture
def token_bounds() -> list[tuple[int, int]]:
    bnds = [
        [0, 4],
        [5, 8],
        [9, 13],
        [14, 21],
        [22, 24],
        [25, 28],
        [29, 32],
        [33, 34],
        [35, 41],
        [42, 44],
        [45, 62],
        [63, 72],
        [72, 73],
        [74, 78],
        [79, 81],
        [82, 84],
        [84, 85],
        [85, 86],
        [86, 87],
        [88, 91],
        [91, 92],
        [92, 93],
        [93, 94],
        [95, 98],
        [99, 101],
        [101, 102],
        [102, 104],
        [105, 109],
        [110, 113],
        [114, 118],
        [119, 121],
        [122, 130],
        [131, 134],
        [134, 135],
        [136, 137],
        [137, 138],
        [138, 142],
        [143, 151],
        [151, 152],
    ]
    return [tuple(item) for item in bnds]
