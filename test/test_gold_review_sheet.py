"""Agreement over partial second-annotator coverage, and the review round trip."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "run" / "eval" / "gold_review_sheet.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("gold_review_sheet", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


grs = _load_module()

TEXT_A = "TAMs secrete IL-10 and regulate IL-6 levels in the tumor."
TEXT_B = "The kinase activates the receptor in cortical neurons."


def _hit(a: int, b: int, entity_id: str, direction: str = "forward") -> dict:
    return {"a": a, "b": b, "entity_id": entity_id, "direction": direction}


def _doc(doc_id: int, text: str, hits: list[dict]) -> dict:
    return {"doc_id": doc_id, "text": text, "ground_truth": hits}


def _write(path: Path, docs: list[dict]) -> str:
    path.write_text(json.dumps(docs), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------- agreement


def test_kappa_ignores_documents_the_second_annotator_never_saw() -> None:
    """The double slice is small; scoring A's whole file against it would sink κ."""
    shared_a = _doc(1, TEXT_A, [_hit(5, 12, "PEL.1")])
    shared_b = _doc(1, TEXT_A, [_hit(5, 12, "PEL.1")])
    unseen = _doc(2, TEXT_B, [_hit(11, 20, "PEL.2"), _hit(25, 33, "PEL.3")])

    report = grs.kappa_report([shared_a, unseen], [shared_b])

    assert report["n_docs_shared"] == 1
    # Only the shared document contributes; the unannotated one is absent, not disagreed.
    assert report["n_pairs"] == 1
    assert report["n_only_a"] == 0


def test_spans_only_the_second_annotator_found_count_against_agreement() -> None:
    docs_a = [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1")])]
    docs_b = [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1"), _hit(22, 30, "PEL.2")])]

    report = grs.kappa_report(docs_a, docs_b)

    assert report["n_only_b"] == 1
    assert report["n_pairs"] == 2
    # Perfect agreement on the shared span alone would be 1.0; the extra span is a
    # disagreement, so it must not be.
    assert report["kappa_entity_id"] < 1.0


def test_a_missed_span_and_an_extra_span_are_penalized_alike() -> None:
    both = _hit(5, 12, "PEL.1")
    extra = _hit(22, 30, "PEL.2")
    a_misses = grs.kappa_report(
        [_doc(1, TEXT_A, [both])], [_doc(1, TEXT_A, [both, extra])]
    )
    b_misses = grs.kappa_report(
        [_doc(1, TEXT_A, [both, extra])], [_doc(1, TEXT_A, [both])]
    )

    assert a_misses["kappa_entity_id"] == b_misses["kappa_entity_id"]
    assert (a_misses["n_only_b"], b_misses["n_only_a"]) == (1, 1)


def test_identical_files_agree_completely() -> None:
    docs = [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1"), _hit(22, 30, "PEL.2")])]

    report = grs.kappa_report(docs, [dict(d) for d in docs])

    assert report["n_only_a"] == report["n_only_b"] == 0
    assert report["kappa_entity_id"] == pytest.approx(1.0)


# ------------------------------------------------------------------ round trip


@pytest.fixture
def kb_csv(tmp_path: Path) -> str:
    path = tmp_path / "kb.csv"
    pd.DataFrame(
        {
            "entity_id": ["PEL.1", "PEL.2"],
            "label": ["regulates", "secretes"],
            "is_canonical": [True, True],
            "canonical_entity_id": ["PEL.1", "PEL.2"],
        }
    ).to_csv(path, index=False)
    return str(path)


def _export(tmp_path: Path, kb_csv: str, gold: str, other: str | None) -> pd.DataFrame:
    out = tmp_path / "review.tsv"
    args = ["export", "--gold", gold, "--kb-csv-path", kb_csv, "--output", str(out)]
    if other is not None:
        args += ["--other", other]
    result = CliRunner().invoke(grs.main, args)
    assert result.exit_code == 0, result.output
    return pd.read_csv(out, sep="\t").fillna("")


def test_export_marks_uncovered_documents_rather_than_showing_disagreement(
    tmp_path: Path, kb_csv: str
) -> None:
    gold = _write(
        tmp_path / "a.json",
        [
            _doc(1, TEXT_A, [_hit(5, 12, "PEL.1")]),
            _doc(2, TEXT_B, [_hit(11, 20, "PEL.2")]),
        ],
    )
    other = _write(tmp_path / "b.json", [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1")])])

    sheet = _export(tmp_path, kb_csv, gold, other)

    uncovered = sheet.loc[sheet["doc_id"] == 2].iloc[0]
    assert uncovered["agrees"] == "not-covered"
    covered = sheet.loc[sheet["doc_id"] == 1].iloc[0]
    assert covered["agrees"] == "True"


def test_export_surfaces_spans_only_the_second_annotator_proposed(
    tmp_path: Path, kb_csv: str
) -> None:
    gold = _write(tmp_path / "a.json", [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1")])])
    other = _write(
        tmp_path / "b.json",
        [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1"), _hit(22, 30, "PEL.2")])],
    )

    sheet = _export(tmp_path, kb_csv, gold, other)

    proposed = sheet.loc[sheet["origin"] == "other"]
    assert len(proposed) == 1
    assert proposed.iloc[0]["entity_id"] == "PEL.2"
    assert "【" in proposed.iloc[0]["context"]  # the adjudicator can see the mention


def test_accepting_a_proposed_span_adds_it_to_verified_gold(
    tmp_path: Path, kb_csv: str
) -> None:
    gold = _write(tmp_path / "a.json", [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1")])])
    other = _write(
        tmp_path / "b.json",
        [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1"), _hit(22, 30, "PEL.2")])],
    )
    sheet_df = _export(tmp_path, kb_csv, gold, other)
    sheet_df["verdict"] = "accept"
    sheet_path = tmp_path / "filled.tsv"
    sheet_df.to_csv(sheet_path, sep="\t", index=False)
    out = tmp_path / "verified.json"

    result = CliRunner().invoke(
        grs.main,
        [
            "import",
            "--gold",
            gold,
            "--sheet",
            str(sheet_path),
            "--other",
            other,
            "--verified-by",
            "curator",
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    hits = json.loads(out.read_text(encoding="utf-8"))[0]["ground_truth"]
    assert [h["entity_id"] for h in hits] == ["PEL.1", "PEL.2"]
    assert {h["source"] for h in hits} == {"human"}
    assert {h["annotator"] for h in hits} == {"curator"}


def test_importing_a_proposed_span_without_its_source_file_is_refused(
    tmp_path: Path, kb_csv: str
) -> None:
    """The hit lives in the other annotator's file; silently dropping it would bias recall."""
    gold = _write(tmp_path / "a.json", [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1")])])
    other = _write(
        tmp_path / "b.json",
        [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1"), _hit(22, 30, "PEL.2")])],
    )
    sheet_df = _export(tmp_path, kb_csv, gold, other)
    sheet_df["verdict"] = "accept"
    sheet_path = tmp_path / "filled.tsv"
    sheet_df.to_csv(sheet_path, sep="\t", index=False)

    result = CliRunner().invoke(
        grs.main,
        [
            "import",
            "--gold",
            gold,
            "--sheet",
            str(sheet_path),
            "--verified-by",
            "curator",
            "--output",
            str(tmp_path / "verified.json"),
        ],
    )

    assert result.exit_code != 0
    assert "--other" in result.output


def test_rejecting_a_proposed_span_leaves_gold_untouched(
    tmp_path: Path, kb_csv: str
) -> None:
    gold = _write(tmp_path / "a.json", [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1")])])
    other = _write(
        tmp_path / "b.json",
        [_doc(1, TEXT_A, [_hit(5, 12, "PEL.1"), _hit(22, 30, "PEL.2")])],
    )
    sheet_df = _export(tmp_path, kb_csv, gold, other)
    sheet_df["verdict"] = sheet_df["origin"].map({"gold": "accept", "other": "reject"})
    sheet_path = tmp_path / "filled.tsv"
    sheet_df.to_csv(sheet_path, sep="\t", index=False)
    out = tmp_path / "verified.json"

    result = CliRunner().invoke(
        grs.main,
        [
            "import",
            "--gold",
            gold,
            "--sheet",
            str(sheet_path),
            "--verified-by",
            "curator",
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    hits = json.loads(out.read_text(encoding="utf-8"))[0]["ground_truth"]
    assert [h["entity_id"] for h in hits] == ["PEL.1"]


def test_import_writes_a_correction_summary_split_by_surface_type(
    tmp_path: Path, kb_csv: str
) -> None:
    """The paper reports how much a curator changed, per anchored/paraphrase slice."""
    anchored = {**_hit(5, 12, "PEL.1"), "surface_anchored": True}
    paraphrase = {**_hit(22, 30, "PEL.2"), "surface_anchored": False}
    gold = _write(tmp_path / "a.json", [_doc(1, TEXT_A, [anchored, paraphrase])])
    sheet_df = _export(tmp_path, kb_csv, gold, None)
    assert set(sheet_df["surface_anchored"].astype(str)) == {"True", "False"}
    sheet_df["verdict"] = ["accept", "reject"]
    sheet_path = tmp_path / "filled.tsv"
    sheet_df.to_csv(sheet_path, sep="\t", index=False)
    out = tmp_path / "verified.json"

    result = CliRunner().invoke(
        grs.main,
        [
            "import",
            "--gold",
            gold,
            "--sheet",
            str(sheet_path),
            "--verified-by",
            "curator",
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    hits = json.loads(out.read_text(encoding="utf-8"))[0]["ground_truth"]
    assert [h["verdict"] for h in hits] == ["accept"]
    summary = json.loads(out.with_suffix(".verification.json").read_text())
    assert summary["correction_rate"] == 0.5
    assert summary["by_slice"] == {
        "anchored": {"accept": 1, "fix": 0, "reject": 0},
        "paraphrase": {"accept": 0, "fix": 0, "reject": 1},
    }
