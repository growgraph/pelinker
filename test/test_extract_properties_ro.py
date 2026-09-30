"""RO extraction: which properties count as symmetric."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

rdflib = pytest.importorskip("rdflib")

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "run"
    / "preprocessing"
    / "extract_properties_ro.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("extract_properties_ro", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


epr = _load_module()

_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix obo: <http://purl.obolibrary.org/obo/> .

obo:RO_1 a owl:ObjectProperty, owl:SymmetricProperty .
obo:RO_2 a owl:ObjectProperty ;
    owl:propertyChainAxiom ( [ owl:inverseOf obo:RO_3 ] obo:RO_3 ) .
obo:RO_3 a owl:ObjectProperty .
obo:RO_4 a owl:ObjectProperty ;
    owl:propertyChainAxiom ( obo:RO_3 obo:RO_5 ) .
obo:RO_5 a owl:ObjectProperty .
"""


def _graph() -> "rdflib.Graph":
    g = rdflib.Graph()
    g.parse(data=_TTL, format="turtle")
    return g


def _ids(iris) -> set[str]:
    return {epr.iri_to_entity_id(i) for i in iris}


def test_declared_symmetric_properties_are_found() -> None:
    assert "RO.1" in _ids(epr.symmetric_property_iris(_graph()))


def test_an_inverse_then_forward_chain_is_symmetric_by_construction() -> None:
    """ "a connected to b iff some c connects a and c connects b" reads the same both ways."""
    assert "RO.2" in _ids(epr.symmetric_property_iris(_graph()))


def test_an_ordinary_chain_is_not_symmetric() -> None:
    found = _ids(epr.symmetric_property_iris(_graph()))

    assert "RO.4" not in found
    assert "RO.3" not in found
