"""Identifier extraction and provenance for the gold sample."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "run" / "eval" / "sample_gold.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sample_gold", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sg = _load_module()

FULL = (
    "Row(doi='https://doi.org/10.1016/x', mag=2017017066, "
    "openalex='https://openalex.org/W2017017066', pmcid=None, "
    "pmid='https://pubmed.ncbi.nlm.nih.gov/12714192')"
)
NO_PMID = (
    "Row(doi=None, mag=3586560, openalex='https://openalex.org/W3586560', "
    "pmcid=None, pmid=None)"
)


def test_every_identifier_is_extracted() -> None:
    ids = sg.extract_identifiers(FULL)

    assert ids["doi"] == "https://doi.org/10.1016/x"
    assert ids["mag"] == "2017017066"
    assert ids["openalex"] == "https://openalex.org/W2017017066"
    assert ids["pmcid"] is None
    # A pmid URL is reduced to the numeric id used everywhere else.
    assert ids["pmid"] == "12714192"


def test_none_fields_become_none_not_the_string() -> None:
    ids = sg.extract_identifiers(NO_PMID)

    assert ids["doi"] is None and ids["pmid"] is None and ids["pmcid"] is None
    assert ids["mag"] == "3586560"


def test_doc_uid_prefers_pmid() -> None:
    """The gold set is pmid-only, and a pmid is what a biomedical reader resolves."""
    assert sg.document_uid(sg.extract_identifiers(FULL)) == "12714192"

    # Without a pmid the remaining identifiers still order deterministically.
    no_pmid = sg.extract_identifiers(NO_PMID)
    assert sg.document_uid(no_pmid).endswith("W3586560")


def test_doc_uid_is_none_when_nothing_identifies_the_row() -> None:
    ids = sg.extract_identifiers(
        "Row(doi=None, mag=None, openalex=None, pmcid=None, pmid=None)"
    )

    assert sg.document_uid(ids) is None


def test_unparsable_ids_field_yields_no_identifier() -> None:
    assert sg.document_uid(sg.extract_identifiers("")) is None
    assert sg.document_uid(sg.extract_identifiers(float("nan"))) is None


def test_id_field_precedence_is_explicit() -> None:
    assert sg.ID_FIELDS == ("pmid", "doi", "openalex", "mag", "pmcid")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Row(pmid='12714192')", "12714192"),
        ("Row(pmid='https://pubmed.ncbi.nlm.nih.gov/999')", "999"),
    ],
)
def test_pmid_accepts_bare_or_url_form(raw: str, expected: str) -> None:
    assert sg.extract_identifiers(raw)["pmid"] == expected


def test_the_drawn_sample_is_pmid_only_and_fully_identified() -> None:
    """Guards the real artifact: every gold document must be citable by pmid."""
    import pandas as pd

    manifest = Path(
        "~/data/pelinker/evaluation-materials/gold/sample_manifest.csv"
    ).expanduser()
    if not manifest.exists():
        pytest.skip("gold sample not drawn in this environment")
    frame = pd.read_csv(manifest)

    assert frame["pmid"].notna().all()
    assert frame["doc_uid"].notna().all()
    assert (frame["doc_uid"].astype(str) == frame["pmid"].astype(str)).all()
    assert not frame["pmid"].duplicated().any()
    # mag is retained: it is the key the fit corpus uses for overlap checks.
    assert frame["mag"].notna().all()


# --------------------------------------------------------------- strata and draws


def _pool() -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for i in range(60):
        rows.append(
            {
                "doc_id": i,
                "doc_uid": str(1000 + i),
                "publication_year": 1995 + i % 30,
                "n_verb_mentions": 0 if i < 12 else 1 + i % 5,
                "labels_detected": ""
                if i < 12
                else ("rare" if i % 3 == 0 else "common"),
            }
        )
    return pd.DataFrame(rows)


def test_tail_labels_are_those_outside_the_top_k() -> None:
    import pandas as pd

    labels = pd.Series(["a|b", "a", "a|c", "b"])

    assert sg.tail_labels(labels, top_k=2) == frozenset({"c"})


def test_strata_split_density_and_tail() -> None:
    pool = _pool()
    strata = sg.assign_strata(pool, frozenset({"rare"}))

    assert strata[pool["n_verb_mentions"] == 0].str.endswith("-none").all()
    positive = strata[pool["n_verb_mentions"] > 0]
    assert positive.str.endswith(("-tail", "-common")).all()
    assert (strata[pool["labels_detected"] == "rare"].str.endswith("-tail")).all()


def test_a_draw_records_inclusion_probabilities_and_the_negative_quota() -> None:
    pool = _pool()
    pool["stratum"] = sg.assign_strata(pool, frozenset({"rare"}))

    drawn = sg.draw_batch(pool, n_total=20, none_fraction=0.1, tail_weight=2.0, seed=1)

    assert len(drawn) == 20
    assert (drawn["n_verb_mentions"] == 0).sum() == 2
    sizes = pool.groupby("stratum").size()
    for name, group in drawn.groupby("stratum"):
        assert group["p_incl"].iloc[0] == len(group) / sizes[name]


def test_tail_weight_over_allocates_tail_strata() -> None:
    import pandas as pd

    sizes = pd.Series({"a-tail": 10, "b-common": 10})
    weights = pd.Series({"a-tail": 3.0, "b-common": 1.0})

    alloc = sg.allocate(sizes, 8, weights)

    assert alloc.sum() == 8
    assert alloc["a-tail"] == 6 and alloc["b-common"] == 2


def test_allocation_never_exceeds_a_stratum() -> None:
    import pandas as pd

    sizes = pd.Series({"a": 2, "b": 50})
    alloc = sg.allocate(sizes, 10, pd.Series({"a": 10.0, "b": 1.0}))

    assert alloc["a"] == 2 and alloc.sum() == 10


def test_roles_follow_the_requested_proportions() -> None:
    roles = sg.assign_roles(12, 4, 1, 1)

    assert roles.count("primary") == 8
    assert roles.count("double") == 2
    assert roles.count("reserve") == 2
