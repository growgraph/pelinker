"""Span-level scoring against the char-offset ground truth.

``data/ground_truth/*.gt.json`` carries ``{"text": ..., "ground_truth": [{"itext", "a",
"b", "entity_id"}, ...]}``. ``pelinker-link-files`` has always parsed it and echoed it
into the output, but nothing ever scored against it — the README pointed at
``run/testing/run_pel_test.py`` "to obtain the accuracy of the model", and that script
does not exist. So the repo had no task-level number at all, only intrinsic clustering
metrics (DBCV/ARI).

Two levels are reported, deliberately separated
-----------------------------------------------
**Detection** — did the linker find the span? Precision/recall/F1 over character spans,
matched within a document. This is unambiguous and comparable across model versions.

**Entity agreement** — of the spans it found, did it assign the right id? This one needs
care. Since the KB-out work, ``entity_id_predicted`` is a *minted cluster id*
(``kb::C0007``), while the ground truth carries *input KB* ids (``PEL.000032``,
``RO.0002206``). Comparing them directly scores zero for reasons that have nothing to do
with quality. Pass ``predicted_id_to_kb_in`` to translate, and read
:attr:`GroundTruthScore.n_id_comparable` before trusting
:attr:`GroundTruthScore.entity_accuracy` — when it is 0, the ids were never comparable
and the metric is undefined rather than bad.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

MatchMode = "overlap", "exact"

GT_DIRECTIONS = ("forward", "inverse", "symmetric", "na")
"""Predicate orientation relative to the KB entry's canonical argument order.

``forward`` — surface arguments follow the KB entry (``X activates Y``);
``inverse`` — reversed, typically passive voice or a converse phrasing
(``Y is activated by X``); ``symmetric`` — the predicate has no orientation
(``X interacts with Y``); ``na`` — orientation not annotated / not applicable.
"""


@dataclass(frozen=True)
class GtSpan:
    """One ground-truth annotation: a character range in a document, plus its KB id.

    The required core (``itext``/``a``/``b``/``entity_id``) is what the scorer consumes.
    The optional fields carry annotation provenance and the directionality contract;
    older ``*.gt.json`` files without them load unchanged.
    """

    itext: int
    a: int
    b: int
    entity_id: str | None = None

    direction: str | None = None
    """One of :data:`GT_DIRECTIONS`, or ``None`` for legacy annotations."""
    subject_span: tuple[int, int] | None = None
    object_span: tuple[int, int] | None = None
    surface: str | None = None
    annotator: str | None = None
    source: str | None = None
    """Annotation channel, e.g. ``"llm"`` or ``"human"``."""
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.b <= self.a:
            raise ValueError(f"span end must exceed start, got a={self.a} b={self.b}")
        if self.direction is not None and self.direction not in GT_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {GT_DIRECTIONS}, got {self.direction!r}"
            )


def _optional_span(value: object, *, field: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be a [start, end] pair, got {value!r}")
    return int(value[0]), int(value[1])


def load_ground_truth_spans(path: pathlib.Path | str) -> list[GtSpan]:
    """Read ``*.gt.json`` (single object or list of objects) into spans."""
    p = pathlib.Path(path).expanduser()
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    spans: list[GtSpan] = []
    for doc_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{p}: document {doc_index} must be an object")
        for hit in item.get("ground_truth") or []:
            spans.append(
                GtSpan(
                    # Fall back to position in the file when itext is absent.
                    itext=int(hit.get("itext", doc_index)),
                    a=int(hit["a"]),
                    b=int(hit["b"]),
                    entity_id=(
                        None if hit.get("entity_id") is None else str(hit["entity_id"])
                    ),
                    direction=(
                        None if hit.get("direction") is None else str(hit["direction"])
                    ),
                    subject_span=_optional_span(
                        hit.get("subject_span"), field="subject_span"
                    ),
                    object_span=_optional_span(
                        hit.get("object_span"), field="object_span"
                    ),
                    surface=(
                        None if hit.get("surface") is None else str(hit["surface"])
                    ),
                    annotator=(
                        None if hit.get("annotator") is None else str(hit["annotator"])
                    ),
                    source=(None if hit.get("source") is None else str(hit["source"])),
                    confidence=(
                        None
                        if hit.get("confidence") is None
                        else float(hit["confidence"])
                    ),
                )
            )
    return spans


def _overlaps(a1: int, b1: int, a2: int, b2: int) -> bool:
    return a1 < b2 and a2 < b1


@dataclass(frozen=True)
class GroundTruthScore:
    """Detection quality, and entity accuracy over the spans that were detected."""

    n_gold: int
    n_predicted: int
    n_matched: int
    match_mode: str

    n_id_comparable: int
    """Matched pairs where both sides carried an id that could be compared.

    Zero means entity accuracy is **undefined**, not zero — most often because the model
    emits minted KB-out ids and no ``predicted_id_to_kb_in`` map was supplied."""
    n_id_correct: int

    @property
    def precision(self) -> float | None:
        if self.n_predicted == 0:
            return None
        return self.n_matched / self.n_predicted

    @property
    def recall(self) -> float | None:
        if self.n_gold == 0:
            return None
        return self.n_matched / self.n_gold

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0.0:
            return None
        return 2 * p * r / (p + r)

    @property
    def entity_accuracy(self) -> float | None:
        """Correct ids over comparable matched pairs; ``None`` when nothing is comparable."""
        if self.n_id_comparable == 0:
            return None
        return self.n_id_correct / self.n_id_comparable

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_gold": int(self.n_gold),
            "n_predicted": int(self.n_predicted),
            "n_matched": int(self.n_matched),
            "match_mode": self.match_mode,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_id_comparable": int(self.n_id_comparable),
            "n_id_correct": int(self.n_id_correct),
            "entity_accuracy": self.entity_accuracy,
        }


def score_predictions_against_ground_truth(
    predictions: Sequence[Mapping[str, Any]],
    gold: Iterable[GtSpan],
    *,
    match_mode: str = "overlap",
    predicted_id_to_kb_in: Mapping[str, str] | None = None,
) -> GroundTruthScore:
    """Greedy one-to-one span matching within each document.

    Args:
        predictions: Rows with ``itext``, ``a``, ``b`` and ``entity_id_predicted``
            (the shape ``Linker.predict`` emits).
        match_mode: ``"overlap"`` counts any character overlap; ``"exact"`` requires
            identical boundaries. Overlap is the fairer default — the linker's spans come
            from lemma matching and legitimately differ from annotation boundaries by a
            token — but exact is available when boundary fidelity is the question.
        predicted_id_to_kb_in: Maps ``entity_id_predicted`` onto input-KB ids so the
            comparison is meaningful. Without it, minted KB-out ids never match and
            ``n_id_comparable`` stays 0.

    Matching is greedy in document order, and each gold span is consumed at most once, so
    duplicate predictions over one annotation count as false positives rather than
    inflating recall.
    """
    if match_mode not in MatchMode:
        raise ValueError(f"match_mode must be one of {MatchMode}, got {match_mode!r}")

    gold_by_doc: dict[int, list[GtSpan]] = {}
    n_gold = 0
    for span in gold:
        gold_by_doc.setdefault(span.itext, []).append(span)
        n_gold += 1

    consumed: set[tuple[int, int]] = set()
    n_matched = 0
    n_id_comparable = 0
    n_id_correct = 0

    for row in predictions:
        try:
            itext = int(row.get("itext", 0) or 0)
            a = int(row["a"])
            b = int(row["b"])
        except (KeyError, TypeError, ValueError):
            continue
        hit: GtSpan | None = None
        for span in gold_by_doc.get(itext, ()):
            key = (span.a, span.b)
            if key in consumed:
                continue
            if match_mode == "exact":
                ok = span.a == a and span.b == b
            else:
                ok = _overlaps(a, b, span.a, span.b)
            if ok:
                hit = span
                consumed.add(key)
                break
        if hit is None:
            continue
        n_matched += 1

        pred_id_raw = row.get("entity_id_predicted")
        if pred_id_raw is None or hit.entity_id is None:
            continue
        pred_id = str(pred_id_raw)
        if predicted_id_to_kb_in is not None:
            mapped = predicted_id_to_kb_in.get(pred_id)
            if mapped is None:
                # Unmappable prediction: not comparable, so it neither helps nor hurts.
                continue
            pred_id = str(mapped)
        n_id_comparable += 1
        if pred_id == hit.entity_id:
            n_id_correct += 1

    return GroundTruthScore(
        n_gold=n_gold,
        n_predicted=len(predictions),
        n_matched=n_matched,
        match_mode=match_mode,
        n_id_comparable=n_id_comparable,
        n_id_correct=n_id_correct,
    )
