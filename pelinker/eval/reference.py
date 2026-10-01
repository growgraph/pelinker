"""The reference inventory T̂: a scoring-only fold of the property KB.

The KB is taken as given, so when two entries mean the same thing in text the KB keeps
both. Scoring needs to know about such identities; training must not. T̂ carries them, in
tiers:

- **tier 0**, the KB's own declarations: each converse-pair member folds onto the pair's
  canonical member (the same map the class views use, :func:`canonical_id_map`);
- **tier 1** (``kb_implied``): identities the KB implies but does not declare, listed in an
  equivalences file. For example, a converse minted for a verb that was later marked
  symmetric is the same relation as that verb;
- **tier 2**: synonymy and polysemy verdicts from blind adjudication, which join this
  module with the revision sheet.

T̂ is used for scoring only. Training labels come from :mod:`pelinker.kb.classes`, which
accepts no judgement about the KB, so nothing here can reach the fit it is used to score.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from pelinker.kb.classes import canonical_id_map, symmetric_ids

EQUIVALENCE_COLUMNS = ("entity_id", "equivalent_to", "tier", "note")
TIER_KB_IMPLIED = "kb_implied"
EQUIVALENCE_TIERS = (TIER_KB_IMPLIED,)


def load_equivalences(
    path: str | pathlib.Path, *, kb: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Read an equivalences file (``entity_id,equivalent_to,tier,note``).

    Each row says ``entity_id`` is the same relation as ``equivalent_to``. Rows are
    refused, not skipped, when the tier is unknown, a row maps an id onto itself, an id
    has two targets, or (given ``kb``) either id is not in the KB: a silently dropped
    identity changes a score without changing its name.
    """
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in EQUIVALENCE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {', '.join(missing)}")
    frame = frame.assign(
        entity_id=frame["entity_id"].str.strip(),
        equivalent_to=frame["equivalent_to"].str.strip(),
        tier=frame["tier"].str.strip(),
    )
    unknown = sorted(set(frame["tier"]) - set(EQUIVALENCE_TIERS))
    if unknown:
        raise ValueError(
            f"{path}: unknown tier {unknown}; expected one of {EQUIVALENCE_TIERS}"
        )
    self_maps = frame.loc[frame["entity_id"] == frame["equivalent_to"], "entity_id"]
    if not self_maps.empty:
        raise ValueError(f"{path}: ids mapped onto themselves: {sorted(self_maps)}")
    repeated = frame.loc[frame["entity_id"].duplicated(), "entity_id"]
    if not repeated.empty:
        raise ValueError(f"{path}: ids with more than one target: {sorted(repeated)}")
    if kb is not None:
        known = set(kb["entity_id"].astype(str))
        unknown_ids = sorted(
            (set(frame["entity_id"]) | set(frame["equivalent_to"])) - known
        )
        if unknown_ids:
            raise ValueError(f"{path}: ids not in the KB: {unknown_ids}")
    return frame


def fold_map(
    kb: pd.DataFrame, equivalences: pd.DataFrame | None = None
) -> dict[str, str]:
    """Every KB id → its T̂ class representative (tiers 0 and 1).

    Tier 0 folds a converse member onto its canonical member; tier 1 then folds along the
    equivalences, following chains to their end. A cycle is refused. Ids absent from the
    map are left alone by callers, as with :func:`canonical_id_map`.
    """
    tier0 = canonical_id_map(kb)
    tier1: dict[str, str] = {}
    if equivalences is not None:
        for source, target in zip(
            equivalences["entity_id"], equivalences["equivalent_to"]
        ):
            tier1[tier0.get(source, source)] = tier0.get(target, target)

    def _resolve(entity_id: str) -> str:
        seen = {entity_id}
        current = tier0.get(entity_id, entity_id)
        while current in tier1:
            current = tier1[current]
            if current in seen:
                raise ValueError(f"equivalence cycle through {entity_id!r}")
            seen.add(current)
        return current

    return {entity_id: _resolve(entity_id) for entity_id in tier0}


def folded_symmetric_ids(kb: pd.DataFrame, fold: dict[str, str]) -> frozenset[str]:
    """Representatives of symmetric relations, where direction carries no information."""
    return frozenset(fold.get(entity_id, entity_id) for entity_id in symmetric_ids(kb))
