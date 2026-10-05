"""What counts as one class of the property KB, for training and for scoring.

The KB is taken as given: its entries and labels are never edited here. What varies is
the *view*, a deterministic projection computed from the KB's own declarations, namely
the converse pairs and ``is_symmetric`` carried by the pairs KB that
``run/preprocessing/derive_inverse_pairs.py`` writes:

- ``raw``: the matched label itself. The two members of a converse pair are two classes.
- ``rel``: the canonical relation. Both members of a pair fold onto the canonical member,
  and orientation is ignored.
- ``reldir``: the canonical relation and the mention's direction relative to it. A
  passive mention of ``R`` is ``(R, inverse)`` whether or not the KB holds a converse
  entry for ``R``.

**Why the selection objective uses ``reldir``.** The weak-label matcher prefers the label
whose own reading has the mention's voice. A passive mention of a relation with a
converse entry is therefore stored under that entry, which is its own ``raw`` class. A
passive mention of a relation without one is stored under the active label with
``direction=inverse``, sharing the active class. Scored against ``raw`` labels, an
agreement objective rewards separating the voices of some relations and penalizes it for
others, depending only on which converse entries the KB happens to hold. ``reldir``
treats every relation alike. ``rel`` folds the voices together, so it would reward
clusters that merge them and remove the only direction signal an embedding-only linker
has.

**What views never use.** Views are built from the KB's declarations only. Judgements
*about* the KB — that two entries mean the same thing in text, or that one entry covers
two senses — form a scoring-only reference inventory. They are deliberately not accepted
here, so they cannot leak into the training labels of the fit they are used to score.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

import pandas as pd

ClassView = Literal["raw", "rel", "reldir"]
CLASS_VIEWS: tuple[ClassView, ...] = get_args(ClassView)

CANONICAL_COLUMNS = ("is_canonical", "canonical_entity_id")

DIRECTION_FORWARD = "forward"
DIRECTION_INVERSE = "inverse"
DIRECTION_SYMMETRIC = "symmetric"

RELATION_COLUMN = "relation"
"""Canonical relation label of a mention row (added by :func:`add_view_columns`)."""
RELATION_DIRECTION_COLUMN = "relation_direction"
"""Direction of a mention row relative to its canonical relation."""
ENTITY_CLASS_COLUMN = "entity_class"
"""The row's class key in the chosen view; agreement metrics score against it."""
VIEW_COLUMNS = (RELATION_COLUMN, RELATION_DIRECTION_COLUMN, ENTITY_CLASS_COLUMN)

CLASS_SEPARATOR = "|"
"""Joins relation and direction in a ``reldir`` class key (``"regulates|inverse"``)."""

_DERIVATION_SCRIPT = "run/preprocessing/derive_inverse_pairs.py"
_OPPOSITE = {DIRECTION_FORWARD: DIRECTION_INVERSE, DIRECTION_INVERSE: DIRECTION_FORWARD}


def _suggest_pairs_path(kb_csv_path: str | pathlib.Path | None) -> str:
    """The sibling ``*.pairs.csv`` to point at, if one is on disk next to the input."""
    if kb_csv_path is None:
        return ""
    path = pathlib.Path(kb_csv_path)
    stem = path.name.split(".csv")[0]
    # `properties.synthesis.2.inverse` -> `properties.synthesis.2`: the derivation reads
    # the inverse file and writes the pairs file beside it.
    base = stem[: -len(".inverse")] if stem.endswith(".inverse") else stem
    candidate = path.with_name(f"{base}.pairs.csv")
    return str(candidate) if candidate.exists() else ""


def require_canonical_kb(
    kb: pd.DataFrame, *, kb_csv_path: str | pathlib.Path | None = None
) -> None:
    """Raise unless ``kb`` carries the converse-pair derivation's columns."""
    missing = [c for c in CANONICAL_COLUMNS if c not in kb.columns]
    if not missing:
        return
    where = f"{kb_csv_path}: " if kb_csv_path else ""
    suggestion = _suggest_pairs_path(kb_csv_path)
    hint = (
        f"Pass {suggestion} instead."
        if suggestion
        else f"Build one with {_DERIVATION_SCRIPT} and pass its *.pairs.csv output."
    )
    raise ValueError(
        f"{where}KB is missing {', '.join(missing)}, so it predates the converse-pair "
        f"derivation. A *.inverse.csv is the *input* to {_DERIVATION_SCRIPT}, not its "
        f"output. {hint}"
    )


def canonical_id_map(
    kb: pd.DataFrame, *, kb_csv_path: str | pathlib.Path | None = None
) -> dict[str, str]:
    """Every KB id → the canonical id of its converse pair.

    Built from the **whole** KB, not the canonical half, since its purpose is to accept a
    converse-member id and fold it onto the pair's canonical member. Ids absent from the
    map are left alone by callers: an id the KB does not know is not this function's to
    rewrite.
    """
    require_canonical_kb(kb, kb_csv_path=kb_csv_path)
    mapping: dict[str, str] = {}
    for _, row in kb.iterrows():
        entity_id = str(row["entity_id"])
        canonical = row.get("canonical_entity_id")
        mapping[entity_id] = (
            str(canonical) if isinstance(canonical, str) and canonical else entity_id
        )
    return mapping


def symmetric_ids(kb: pd.DataFrame) -> frozenset[str]:
    """Ids of relations with no orientation ("interacts with", "overlaps").

    Their only valid ``direction`` is ``symmetric``: a reversed mention is the same
    relation read the same way. A KB without ``is_symmetric`` marks nothing.
    """
    if "is_symmetric" not in kb.columns:
        return frozenset()
    flagged = kb["is_symmetric"].fillna(False).astype(bool)
    return frozenset(kb.loc[flagged, "entity_id"].astype(str))


def file_sha256(path: str | pathlib.Path) -> str:
    """Content hash of a KB file, recorded wherever a view's meaning depends on it."""
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class KbClasses:
    """The label-level facts a view needs, read from a pairs KB.

    Attributes:
        canonical_label: Every KB label → the label of its pair's canonical member (a
            label that is its own canonical member maps to itself).
        converse_labels: Labels whose own reading is the converse of their canonical
            member ("regulated by" for ``regulates``). A direction measured against one
            of these is flipped to be measured against the canonical member.
        symmetric_labels: Labels of relations with no orientation, including every label
            whose canonical member is symmetric.
    """

    canonical_label: Mapping[str, str]
    converse_labels: frozenset[str]
    symmetric_labels: frozenset[str]

    @classmethod
    def from_kb(
        cls, kb: pd.DataFrame, *, kb_csv_path: str | pathlib.Path | None = None
    ) -> KbClasses:
        """Read the view facts from a pairs KB; any other KB is refused."""
        require_canonical_kb(kb, kb_csv_path=kb_csv_path)
        labels = kb["label"].astype(str).str.strip()
        duplicated = sorted(set(labels[labels.duplicated()]))
        if duplicated:
            raise ValueError(
                f"KB labels must be unique to define classes; duplicated: {duplicated}"
            )
        label_of = dict(zip(kb["entity_id"].astype(str), labels))
        canon_ids = canonical_id_map(kb, kb_csv_path=kb_csv_path)
        symmetric = symmetric_ids(kb)
        canonical_label: dict[str, str] = {}
        converse: set[str] = set()
        symmetric_labels: set[str] = set()
        for entity_id, label in label_of.items():
            canonical_id = canon_ids.get(entity_id, entity_id)
            canonical_label[label] = label_of.get(canonical_id, label)
            if canonical_id != entity_id:
                converse.add(label)
            if entity_id in symmetric or canonical_id in symmetric:
                symmetric_labels.add(label)
        return cls(
            canonical_label=canonical_label,
            converse_labels=frozenset(converse - symmetric_labels),
            symmetric_labels=frozenset(symmetric_labels),
        )

    @classmethod
    def from_csv(cls, path: str | pathlib.Path) -> KbClasses:
        """:meth:`from_kb` on a KB CSV file."""
        return cls.from_kb(pd.read_csv(path), kb_csv_path=path)


def relation_direction(label: str, direction: object, classes: KbClasses) -> str:
    """A mention's direction relative to the canonical member of its label's pair.

    ``direction`` is relative to ``label`` itself, as the weak-label matcher records it.
    A missing value (a lexical, non-verb match) reads the label's own way, so it counts
    as ``forward``. A symmetric relation is always ``symmetric``.
    """
    if label in classes.symmetric_labels or direction == DIRECTION_SYMMETRIC:
        return DIRECTION_SYMMETRIC
    own = direction if direction in _OPPOSITE else DIRECTION_FORWARD
    assert isinstance(own, str)
    return _OPPOSITE[own] if label in classes.converse_labels else own


def class_key(relation: str, direction: str, view: ClassView, *, label: str) -> str:
    """The class a mention belongs to in ``view``."""
    if view == "raw":
        return label
    if view == "rel" or direction == DIRECTION_SYMMETRIC:
        return relation
    return f"{relation}{CLASS_SEPARATOR}{direction}"


def add_view_columns(
    frame: pd.DataFrame,
    classes: KbClasses,
    view: ClassView,
    *,
    passthrough_labels: frozenset[str] = frozenset(),
    entity_column: str = "entity",
    direction_column: str = "direction",
) -> pd.DataFrame:
    """A copy of ``frame`` with :data:`VIEW_COLUMNS` added; ``entity`` is left as is.

    Args:
        frame: Mention rows with a label column and, optionally, the matcher's
            direction column.
        classes: View facts from :meth:`KbClasses.from_kb`.
        view: Which class key :data:`ENTITY_CLASS_COLUMN` carries.
        passthrough_labels: Labels outside the KB that keep themselves as relation and
            class (the synthetic negative label).

    Raises:
        ValueError: A label is neither in the KB nor passed through. A view over a KB
            the mentions were not labelled with would silently mean something else.
    """
    if view not in CLASS_VIEWS:
        raise ValueError(f"class view must be one of {CLASS_VIEWS}, got {view!r}")
    labels = frame[entity_column].astype(str)
    unknown = sorted(
        set(labels) - set(classes.canonical_label) - set(passthrough_labels)
    )
    if unknown:
        shown = ", ".join(repr(u) for u in unknown[:10])
        more = f" (+{len(unknown) - 10} more)" if len(unknown) > 10 else ""
        raise ValueError(
            f"{len(unknown)} mention labels are not in the class KB: {shown}{more}. "
            "Pass the KB the mentions were embedded with."
        )
    directions = (
        frame[direction_column]
        if direction_column in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    relations: list[str] = []
    rel_dirs: list[str] = []
    keys: list[str] = []
    for label, direction in zip(labels, directions):
        if label in passthrough_labels and label not in classes.canonical_label:
            relations.append(label)
            rel_dirs.append(DIRECTION_FORWARD)
            keys.append(label)
            continue
        relation = classes.canonical_label[label]
        rel_dir = relation_direction(
            label, None if pd.isna(direction) else direction, classes
        )
        relations.append(relation)
        rel_dirs.append(rel_dir)
        keys.append(class_key(relation, rel_dir, view, label=label))
    out = frame.copy()
    out[RELATION_COLUMN] = relations
    out[RELATION_DIRECTION_COLUMN] = rel_dirs
    out[ENTITY_CLASS_COLUMN] = keys
    return out


def check_class_view(view: str, class_kb_path: str | pathlib.Path | None) -> ClassView:
    """Validate a CLI's ``--class-view`` / ``--class-kb-path`` pair; return the view.

    Raises:
        ValueError: An unknown view, or a view other than ``raw`` without a KB path.
    """
    for known in CLASS_VIEWS:
        if view == known:
            if known != "raw" and class_kb_path is None:
                raise ValueError(
                    f"--class-view {known} needs --class-kb-path (the pairs KB the "
                    "mentions were embedded with); pass --class-view raw to score "
                    "against the matched labels."
                )
            return known
    raise ValueError(f"class view must be one of {CLASS_VIEWS}, got {view!r}")


def reference_label_column(frame: pd.DataFrame) -> str:
    """The column agreement metrics compare clusters against.

    :data:`ENTITY_CLASS_COLUMN` when a view was applied, else the raw ``entity`` label.
    """
    return ENTITY_CLASS_COLUMN if ENTITY_CLASS_COLUMN in frame.columns else "entity"
