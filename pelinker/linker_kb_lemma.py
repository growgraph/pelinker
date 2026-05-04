"""Lemma-based KB training-entity index (same resolution as embedding-time matching)."""

from __future__ import annotations

from pelinker.onto import WordGrouping, _wg_for_property
from pelinker.util import text_to_tokens


def build_kb_lemma_index(
    labels_map: dict[str, str], nlp: object
) -> dict[WordGrouping, dict[str, str]]:
    """Build ``{word_grouping: {lemma_string: kb_training_entity_label}}`` from ``labels_map`` values.

    Mirrors the per-property lemma matching used at training time in
    :func:`pelinker.util.extract_and_embed_mentions` (``_wg_for_property`` for the bucket,
    lemma strings for comparison), inverted to an O(1) lookup keyed by mention lemma.
    """
    index: dict[WordGrouping, dict[str, str]] = {}
    for prop in set(labels_map.values()):
        wg = _wg_for_property(prop)
        if wg is None:
            continue
        tokens = text_to_tokens(nlp, prop)
        lemma = " ".join(t.lemma for t in tokens)
        index.setdefault(wg, {})[lemma] = prop
    return index


def lookup_kb_training_entity_label(
    word_grouping: WordGrouping | None,
    lemma: str,
    kb_lemma_by_wg: dict[WordGrouping, dict[str, str]],
) -> str | None:
    """Resolve training ``entity`` label string from mention ``word_grouping`` + space-joined lemmas."""
    if word_grouping is None or not lemma:
        return None
    if not isinstance(word_grouping, WordGrouping):
        return None
    return kb_lemma_by_wg.get(word_grouping, {}).get(str(lemma))


def enrich_entity_predictions_kb_validation(
    rows: list[dict[str, object]],
    kb_lemma_by_wg: dict[WordGrouping, dict[str, str]],
    labels_map: dict[str, str],
) -> None:
    """Add validation-only fields to each prediction row (mutates ``rows`` in place)."""
    for row in rows:
        wg_obj = row.get("word_grouping")
        wg: WordGrouping | None = wg_obj if isinstance(wg_obj, WordGrouping) else None
        lemma = str(row.get("lemma", "") or "")
        from_lemma = lookup_kb_training_entity_label(wg, lemma, kb_lemma_by_wg)
        eid = row.get("entity_id_predicted")
        for_prediction = labels_map.get(str(eid)) if eid is not None else None
        row["kb_training_entity_from_lemma"] = from_lemma
        row["kb_training_entity_for_prediction"] = for_prediction
        row["lemma_kb_matches_predicted_entity"] = (
            from_lemma is not None
            and for_prediction is not None
            and from_lemma == for_prediction
        )
