"""Sample audit: eligibility re-check, empty documents, concentration, coverage curve."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_MODULE_PATH = Path(__file__).resolve().parents[1] / "run" / "eval" / "audit_sample.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_sample", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


au = _load_module()

KB = pd.DataFrame(
    {
        "entity_id": ["PEL.1", "RO.2", "PEL.3"],
        "label": ["regulates", "regulated by", "secretes"],
        "canonical_entity_id": ["PEL.1", "PEL.1", "PEL.3"],
    }
)
MANIFEST = pd.DataFrame(
    {
        "doc_id": [1, 2, 3],
        "publication_year": [2015, 2015, 1980],
        "stratum": ["y0-high-tail", "y0-none", "y0-low-common"],
        "batch": [0, 0, 1],
        "n_verb_mentions": [3, 0, 1],
    }
)
GOLD = {
    1: [{"entity_id": "PEL.1"}, {"entity_id": "PEL.1"}],
    2: [{"entity_id": "PEL.3"}],
    3: [],
}


def test_converse_ids_count_under_the_canonical_label() -> None:
    by_id, by_label = au.canonical_label_map(KB)

    assert by_id["RO.2"] == "regulates"
    assert by_label["regulated by"] == "regulates"


def test_gold_section_flags_empty_documents_and_hit_negatives() -> None:
    by_id, _ = au.canonical_label_map(KB)

    section, counts = au.gold_section(MANIFEST, GOLD, by_id, min_mentions=2)

    assert section["documents_without_hits"] == 1
    assert section["negative_controls_with_hits"] == 1
    assert section["labels_at_min_mentions"] == 1
    assert counts.to_dict() == {"regulates": 2, "secretes": 1}


def test_coverage_curve_is_cumulative_over_batches() -> None:
    by_id, by_label = au.canonical_label_map(KB)
    _, counts = au.gold_section(MANIFEST, GOLD, by_id, min_mentions=2)
    pool = pd.Series(["regulated by|secretes", "regulates", "secretes", ""])

    cov = au.coverage_section(MANIFEST, GOLD, counts, pool, by_id, by_label, 2)

    assert cov["reachable_labels"] == 2
    assert cov["covered_reachable"] == 1
    assert [c["distinct_labels"] for c in cov["curve"]] == [2, 2]


def test_quality_section_reapplies_the_gate() -> None:
    texts = {1: "short", 2: "short", 3: "short"}

    section = au.quality_section(MANIFEST, texts)

    assert section["eligibility"] == {"too_short": 2, "year_before_min": 1}


def test_documents_absent_from_the_gold_file_are_not_scored(tmp_path: Path) -> None:
    """Reserve documents are never annotated; they must not count as 'without hits'."""
    import json

    from click.testing import CliRunner

    manifest = MANIFEST.assign(role=["primary", "primary", "reserve"])
    manifest.to_csv(tmp_path / "sample_manifest.csv", index=False)
    (tmp_path / "sample_texts.jsonl").write_text(
        "".join(json.dumps({"doc_id": d, "text": "x"}) + "\n" for d in (1, 2, 3)),
        encoding="utf-8",
    )
    gold = [{"doc_id": d, "ground_truth": GOLD[d]} for d in (1, 2)]
    (tmp_path / "gold.llm-a.json").write_text(json.dumps(gold), encoding="utf-8")
    kb_path = tmp_path / "kb.csv"
    KB.to_csv(kb_path, index=False)

    result = CliRunner().invoke(
        au.main, ["--gold-dir", str(tmp_path), "--kb-csv-path", str(kb_path)]
    )

    assert result.exit_code == 0, result.output
    audit = json.loads((tmp_path / "sample_audit.json").read_text())
    assert audit["documents_not_in_gold_file"] == 1
    assert audit["gold"]["documents"] == 2
    assert audit["gold"]["documents_without_hits"] == 0


def test_paraphrase_share_counts_unanchored_hits() -> None:
    by_id, _ = au.canonical_label_map(KB)
    gold = {
        1: [
            {"entity_id": "PEL.1", "surface_anchored": True},
            {"entity_id": "PEL.1", "surface_anchored": False},
        ],
        2: [{"entity_id": "PEL.3", "surface_anchored": True}],
        3: [],
    }

    section, _ = au.gold_section(MANIFEST, gold, by_id, min_mentions=2)

    assert section["paraphrase_hits"] == 1
    assert section["paraphrase_share"] == round(1 / 3, 3)
