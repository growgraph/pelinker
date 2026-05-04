"""Tests for fit CLI embedding metadata (filename inference and per-path sources)."""

from pathlib import Path

import pytest

from pelinker.cli.fit import (
    _embedding_metadata,
    _parse_embedding_parquet_path,
    _parse_embedding_parquet_stem,
)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("res_pubmedbert_1", ("pubmedbert", "1")),
        ("corpus_pubmedbert_12", ("pubmedbert", "12")),
        ("out_res_biobert_1_2_3", ("biobert", "1,2,3")),
        ("pubmedbert_1", ("pubmedbert", "1")),
        ("prefix_bluebert_9", ("bluebert", "9")),
        ("x_scibert_2", ("scibert", "2")),
        ("run_bert_1", ("bert", "1")),
        ("res_biobert-stsb_1", ("biobert-stsb", "1")),
    ],
)
def test_parse_embedding_parquet_stem_ok(stem: str, expected: tuple[str, str]) -> None:
    assert _parse_embedding_parquet_stem(stem) == expected


@pytest.mark.parametrize(
    "stem",
    [
        "mentions",
        "corpus_smoke_trunc",
        "unknownmodel_1",
        "res_pubmedbert_",
        "res_pubmedbert_xy",
    ],
)
def test_parse_embedding_parquet_stem_none(stem: str) -> None:
    assert _parse_embedding_parquet_stem(stem) is None


def test_parse_embedding_parquet_path_parquet_gz(tmp_path: Path) -> None:
    p = tmp_path / "res_pubmedbert_1.parquet.gz"
    p.touch()
    assert _parse_embedding_parquet_path(p) == ("pubmedbert", "1")


def test_embedding_metadata_infers_per_path_when_lists_omitted(tmp_path: Path) -> None:
    a = tmp_path / "res_pubmedbert_1.parquet"
    b = tmp_path / "res_biobert_2.parquet"
    a.touch()
    b.touch()
    meta = _embedding_metadata(
        [a, b],
        model_type="pubmedbert",
        layers_spec="1",
        model_types=None,
        layers_specs=None,
    )
    assert [s.model_type for s in meta.sources] == ["pubmedbert", "biobert"]
    assert [s.layers_spec for s in meta.sources] == ["1", "2"]


def test_embedding_metadata_explicit_lists_override_filename(tmp_path: Path) -> None:
    a = tmp_path / "res_pubmedbert_1.parquet"
    a.touch()
    meta = _embedding_metadata(
        [a],
        model_type="bert",
        layers_spec="9",
        model_types=["scibert"],
        layers_specs=["3"],
    )
    assert meta.sources[0].model_type == "scibert"
    assert meta.sources[0].layers_spec == "3"


def test_embedding_metadata_mixed_infer_layers_only(tmp_path: Path) -> None:
    a = tmp_path / "res_pubmedbert_1.parquet"
    b = tmp_path / "res_pubmedbert_2.parquet"
    a.touch()
    b.touch()
    meta = _embedding_metadata(
        [a, b],
        model_type="pubmedbert",
        layers_spec="1",
        model_types=["pubmedbert", "pubmedbert"],
        layers_specs=None,
    )
    assert [s.layers_spec for s in meta.sources] == ["1", "2"]


def test_embedding_metadata_mixed_infer_model_only(tmp_path: Path) -> None:
    a = tmp_path / "res_pubmedbert_1.parquet"
    b = tmp_path / "res_biobert_1.parquet"
    a.touch()
    b.touch()
    meta = _embedding_metadata(
        [a, b],
        model_type="bert",
        layers_spec="1",
        model_types=None,
        layers_specs=["1", "1"],
    )
    assert [s.model_type for s in meta.sources] == ["pubmedbert", "biobert"]
    assert [s.layers_spec for s in meta.sources] == ["1", "1"]
