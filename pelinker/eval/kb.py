"""The canonical view of the property KB, shared by annotation and scoring.

The KB carries both members of many converse pairs ("regulates" / "regulated by") as
separate entries. ``run/preprocessing/derive_inverse_pairs.py`` links them and elects one
member of each pair as canonical, adding ``is_canonical`` and ``canonical_entity_id``.

Two consumers depend on that view, and they must agree or the measurement is silently
wrong:

- **Annotation** offers canonical labels only, carrying orientation in ``direction``, so a
  passive mention has exactly one valid encoding.
- **Scoring** therefore compares ids in canonical space. A system that answers "regulated
  by" where gold says ("regulates", inverse) picked the right relation; scoring it wrong
  would measure the vocabulary's redundancy rather than the system.

A KB without those columns predates the derivation. Both consumers refuse it rather than
falling back, because the fallback changes what a number means without changing its name.

The id-level helpers live in :mod:`pelinker.kb.classes`, because training uses the same
view; they are re-exported here for the evaluation code.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from pelinker.kb.classes import (
    CANONICAL_COLUMNS,
    canonical_id_map,
    require_canonical_kb,
    symmetric_ids,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "canonical_id_map",
    "canonical_kb",
    "pending_review_pairs",
    "require_canonical_kb",
    "symmetric_ids",
]


def canonical_kb(
    kb: pd.DataFrame, *, kb_csv_path: str | pathlib.Path | None = None
) -> pd.DataFrame:
    """Only the canonical member of each converse pair.

    Offering both members would give a passive mention two equally valid encodings — the
    ambiguity the canonical vocabulary exists to remove.
    """
    require_canonical_kb(kb, kb_csv_path=kb_csv_path)
    return kb.loc[kb["is_canonical"].astype(bool)]


def pending_review_pairs(kb: pd.DataFrame) -> pd.DataFrame:
    """Rows whose converse pair came from a heuristic and is not curator-reviewed yet.

    A wrong heuristic pair merges two distinct relations into one canonical id, which is
    invisible downstream: gold, prompts and scores all agree with each other and are all
    wrong together. Callers warn on a non-empty result.
    """
    if "needs_review" not in kb.columns:
        return kb.iloc[0:0]
    return kb.loc[kb["needs_review"].astype(bool)]
