"""One scoring path for the linker and every baseline, over gold-annotated documents.

Two regimes, deliberately separated:

- **End-to-end** — the system proposes spans *and* ids on raw text; scored with
  :func:`pelinker.kb.ground_truth.score_predictions_against_ground_truth` (span
  detection P/R/F1 + entity accuracy over matched spans).
- **Linking-only** — gold spans are given and the system only assigns an id per span.
  This isolates the discriminative comparison from span proposal, which for the linker
  (and the reference baselines) is lemma matching either way.

Every evaluation records wall-clock time and document count, so runtime tables come from
the same run as the quality numbers.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from pelinker.kb.ground_truth import (
    GroundTruthScore,
    GtSpan,
    load_ground_truth_spans,
    score_predictions_against_ground_truth,
)


@dataclass(frozen=True)
class GoldDoc:
    """One gold-annotated document: text plus its spans (``itext`` already set)."""

    doc_id: int
    text: str
    spans: tuple[GtSpan, ...]


def load_gold_docs(path: pathlib.Path | str) -> list[GoldDoc]:
    """Read the document-list gold JSON, keeping text and spans grouped per document.

    ``itext`` on the returned spans is the document's position in the file — the same
    convention :func:`load_ground_truth_spans` applies — so predictions indexed by
    position in the returned list line up.
    """
    p = pathlib.Path(path).expanduser()
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    spans_by_doc: dict[int, list[GtSpan]] = {}
    for span in load_ground_truth_spans(p):
        spans_by_doc.setdefault(span.itext, []).append(span)
    docs: list[GoldDoc] = []
    for itext, item in enumerate(items):
        docs.append(
            GoldDoc(
                doc_id=int(item.get("doc_id", itext)),
                text=str(item["text"]),
                spans=tuple(spans_by_doc.get(itext, [])),
            )
        )
    return docs


PredictFn = Callable[[list[str]], Sequence[Mapping[str, Any]]]
"""Texts → prediction rows (``itext``/``a``/``b``/``entity_id_predicted``)."""

LinkFn = Callable[[str, int, int], str | None]
"""(text, a, b) → predicted entity id, or ``None`` to abstain."""


@dataclass(frozen=True)
class EvalRun:
    """A scored regime plus its runtime, ready for a results table."""

    system: str
    regime: str
    n_docs: int
    wall_seconds: float
    score: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "regime": self.regime,
            "n_docs": self.n_docs,
            "wall_seconds": round(self.wall_seconds, 3),
            **self.score,
        }


def evaluate_end_to_end(
    predict_fn: PredictFn,
    docs: Sequence[GoldDoc],
    *,
    system: str,
    match_mode: str = "overlap",
    predicted_id_to_kb_in: Mapping[str, str] | None = None,
) -> EvalRun:
    """Span+link scoring of a full predictor over the gold documents."""
    gold = [span for doc in docs for span in doc.spans]
    t0 = time.perf_counter()
    predictions = predict_fn([doc.text for doc in docs])
    wall = time.perf_counter() - t0
    score: GroundTruthScore = score_predictions_against_ground_truth(
        list(predictions),
        gold,
        match_mode=match_mode,
        predicted_id_to_kb_in=predicted_id_to_kb_in,
    )
    return EvalRun(
        system=system,
        regime="end_to_end",
        n_docs=len(docs),
        wall_seconds=wall,
        score=score.to_jsonable(),
    )


@dataclass(frozen=True)
class LinkingOnlyScore:
    """Id assignment quality when spans are given."""

    n_spans: int
    n_predicted: int
    n_correct: int

    @property
    def coverage(self) -> float | None:
        """Fraction of gold spans the system committed an id to (1 − abstention)."""
        if self.n_spans == 0:
            return None
        return self.n_predicted / self.n_spans

    @property
    def accuracy_on_predicted(self) -> float | None:
        if self.n_predicted == 0:
            return None
        return self.n_correct / self.n_predicted

    @property
    def accuracy_overall(self) -> float | None:
        """Abstentions count as wrong — the honest headline number."""
        if self.n_spans == 0:
            return None
        return self.n_correct / self.n_spans

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_spans": self.n_spans,
            "n_predicted": self.n_predicted,
            "n_correct": self.n_correct,
            "coverage": self.coverage,
            "accuracy_on_predicted": self.accuracy_on_predicted,
            "accuracy_overall": self.accuracy_overall,
        }


def evaluate_linking_only(
    link_fn: LinkFn,
    docs: Sequence[GoldDoc],
    *,
    system: str,
) -> EvalRun:
    """Score id assignment over the gold spans that carry an entity id."""
    n_spans = n_predicted = n_correct = 0
    t0 = time.perf_counter()
    for doc in docs:
        for span in doc.spans:
            if span.entity_id is None:
                continue
            n_spans += 1
            predicted = link_fn(doc.text, span.a, span.b)
            if predicted is None:
                continue
            n_predicted += 1
            if str(predicted) == span.entity_id:
                n_correct += 1
    wall = time.perf_counter() - t0
    score = LinkingOnlyScore(
        n_spans=n_spans, n_predicted=n_predicted, n_correct=n_correct
    )
    return EvalRun(
        system=system,
        regime="linking_only",
        n_docs=len(docs),
        wall_seconds=wall,
        score=score.to_jsonable(),
    )
