"""Lemma-based KB training-entity index (same resolution as embedding-time matching)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from pelinker.core.onto import WordGrouping, _wg_for_property
from pelinker.text.tokenize import text_to_tokens


def build_kb_lemma_index(
    labels_map: dict[str, str], nlp: object
) -> dict[WordGrouping, dict[str, str]]:
    """Build ``{word_grouping: {lemma_string: kb_training_entity_label}}`` from ``labels_map`` values.

    Mirrors the per-property lemma matching used at training time in
    :func:`pelinker.text.embed.extract_and_embed_mentions` (``_wg_for_property`` for the bucket,
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


@dataclass(frozen=True)
class KbLemmaValidationMetrics:
    """Aggregate of the per-row ``lemma_kb_matches_predicted_entity`` flag.

    This is the closest thing the pipeline has to a **task-level** accuracy: it asks
    whether the entity a mention was linked to is the same one its own lemma resolves to
    in the KB. The flag was already computed per row and attached to debug output, but
    nothing ever aggregated it, so no run produced a single number for linking quality —
    only intrinsic clustering scores (DBCV/ARI).

    It is a weak, distant-supervision signal, not ground truth: it can only score rows
    whose lemma resolves to a KB entity at all (:attr:`n_resolvable`), and it rewards
    agreement with the same lemma matching that produced the training mentions. Read
    :attr:`match_rate` as "does linking stay consistent with the KB dictionary", not as
    end-task accuracy.
    """

    n_rows: int
    n_resolvable: int
    """Rows whose lemma resolved to some KB entity — the denominator of :attr:`match_rate`."""
    n_matches: int
    n_predicted: int
    """Rows that received an entity prediction at all."""

    @property
    def match_rate(self) -> float | None:
        """Matches over resolvable rows; ``None`` when nothing was resolvable."""
        if self.n_resolvable == 0:
            return None
        return self.n_matches / self.n_resolvable

    @property
    def resolvable_rate(self) -> float | None:
        """Share of rows the metric can say anything about at all."""
        if self.n_rows == 0:
            return None
        return self.n_resolvable / self.n_rows

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_rows": int(self.n_rows),
            "n_resolvable": int(self.n_resolvable),
            "n_matches": int(self.n_matches),
            "n_predicted": int(self.n_predicted),
            "match_rate": self.match_rate,
            "resolvable_rate": self.resolvable_rate,
        }


def aggregate_kb_lemma_validation(
    rows: Sequence[dict[str, object]],
) -> KbLemmaValidationMetrics:
    """Summarize rows already enriched by :func:`enrich_entity_predictions_kb_validation`.

    Rows lacking the enrichment fields count toward ``n_rows`` but not ``n_resolvable``,
    so mixing enriched and unenriched rows lowers the resolvable rate rather than
    silently inflating the match rate.
    """
    n_resolvable = 0
    n_matches = 0
    n_predicted = 0
    for row in rows:
        if row.get("entity_id_predicted") is not None:
            n_predicted += 1
        if row.get("kb_training_entity_from_lemma") is None:
            continue
        n_resolvable += 1
        if bool(row.get("lemma_kb_matches_predicted_entity")):
            n_matches += 1
    return KbLemmaValidationMetrics(
        n_rows=len(rows),
        n_resolvable=n_resolvable,
        n_matches=n_matches,
        n_predicted=n_predicted,
    )
