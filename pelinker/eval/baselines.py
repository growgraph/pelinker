"""Reference baselines for the gold evaluation.

All three baselines share one span proposer (:class:`LexicalSpanProposer`, lemma-window
matching against KB labels — the same idea the linker's stage (A) uses), and differ only
in how a span gets its id:

- :class:`LexicalLemmaBaseline` — the trivial baseline: the id of the KB label whose
  lemma sequence the span matches.
- :class:`EncoderKnnBaseline` — embed the span's context window and the KB's
  ``label: description`` strings with a sentence encoder; cosine top-1.
- :class:`LlmLinkerBaseline` — ask an LLM to pick an id from the full KB for the marked
  mention; disk-cached, so re-scoring is free. Configure a different model (ideally a
  different family) from the one that pre-annotated the gold, or the comparison is
  circular. Provider seam and credentials: :mod:`pelinker.eval.llm`.

Heavy imports (spaCy models, sentence-transformers, the LLM SDK) happen at construction
or call time, never at module import.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import numpy as np
import pandas as pd

from pelinker.eval.kb_prompt import render_kb_catalog
from pelinker.eval.llm import DEFAULT_PROVIDER, complete

_MAX_WINDOW = 4
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _label_records(kb: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(entity_id, label, description) rows with a usable label."""
    out: list[tuple[str, str, str]] = []
    for _, r in kb.iterrows():
        label = r.get("label")
        if pd.isna(label) or not str(label).strip():
            continue
        desc = "" if pd.isna(r.get("description")) else str(r["description"])
        out.append((str(r["entity_id"]), str(label), desc))
    return out


class LexicalSpanProposer:
    """Lemma-window span proposal over KB labels — shared by every baseline's e2e run."""

    def __init__(self, kb: pd.DataFrame, nlp: Any) -> None:
        self._nlp = nlp
        self._index: dict[tuple[str, ...], str] = {}
        # Sorted so a duplicated label resolves to the smallest id, deterministically.
        for entity_id, label, _ in sorted(_label_records(kb)):
            doc = nlp(label)
            lemmas = tuple(
                t.lemma_.lower() for t in doc if not (t.is_punct or t.is_space)
            )
            if 0 < len(lemmas) <= _MAX_WINDOW:
                self._index.setdefault(lemmas, entity_id)

    def lookup_lemmas(self, lemmas: tuple[str, ...]) -> str | None:
        return self._index.get(lemmas)

    def propose(self, text: str) -> list[tuple[int, int, str]]:
        """Non-overlapping ``(a, b, entity_id)`` spans, longest match first."""
        doc = self._nlp(text)
        tokens = [t for t in doc if not (t.is_punct or t.is_space)]
        lemmas = [t.lemma_.lower() for t in tokens]
        taken: set[int] = set()
        spans: list[tuple[int, int, str]] = []
        for width in range(_MAX_WINDOW, 0, -1):
            for i in range(len(tokens) - width + 1):
                if any(j in taken for j in range(i, i + width)):
                    continue
                entity_id = self._index.get(tuple(lemmas[i : i + width]))
                if entity_id is None:
                    continue
                a = tokens[i].idx
                b = tokens[i + width - 1].idx + len(tokens[i + width - 1])
                spans.append((a, b, entity_id))
                taken.update(range(i, i + width))
        return sorted(spans)

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """End-to-end prediction rows in the scorer's shape."""
        rows: list[dict[str, Any]] = []
        for itext, text in enumerate(texts):
            for a, b, entity_id in self.propose(text):
                rows.append(
                    {"itext": itext, "a": a, "b": b, "entity_id_predicted": entity_id}
                )
        return rows


class LexicalLemmaBaseline:
    """Trivial baseline: the span's id is the id of the lemma-matched KB label."""

    def __init__(self, kb: pd.DataFrame, nlp: Any) -> None:
        self._proposer = LexicalSpanProposer(kb, nlp)
        self._nlp = nlp

    @property
    def proposer(self) -> LexicalSpanProposer:
        return self._proposer

    def link(self, text: str, a: int, b: int) -> str | None:
        """Id for a given span: exact lemma match, else the longest matching sub-window.

        The back-off matters for fairness. Gold spans are annotated as whole predicate
        phrases ("is activated by") while KB labels are often the bare form
        ("activated"), so an exact-only baseline would abstain on precisely the mentions
        the task is about — making the comparison a strawman rather than a floor.
        """
        doc = self._nlp(text[a:b])
        lemmas = [t.lemma_.lower() for t in doc if not (t.is_punct or t.is_space)]
        exact = self._proposer.lookup_lemmas(tuple(lemmas))
        if exact is not None:
            return exact
        for width in range(min(len(lemmas), _MAX_WINDOW), 0, -1):
            for i in range(len(lemmas) - width + 1):
                hit = self._proposer.lookup_lemmas(tuple(lemmas[i : i + width]))
                if hit is not None:
                    return hit
        return None

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        return self._proposer.predict(texts)


class EncoderKnnBaseline:
    """Sentence-encoder cosine top-1 of the mention against the KB label strings.

    Two defaults are deliberate, and both were chosen by sweeping them on gold rather
    than assumed:

    - ``context_chars=0`` — the mention is embedded alone. Padding it with surrounding
      sentence text makes the vector a function of the *sentence* rather than the
      predicate: adjacent mentions then collapse onto the same prediction, and accuracy
      falls off a cliff as the window grows.
    - ``include_description=False`` — labels are matched bare. A KB description is prose
      about the relation, and concatenating it pulls the label vector away from the
      surface form the mention actually resembles.

    Getting these wrong turns a competitive baseline into a strawman, which is precisely
    the failure the baseline exists to rule out.
    """

    def __init__(
        self,
        kb: pd.DataFrame,
        *,
        model_name: str = "neuml/pubmedbert-base-embeddings",
        context_chars: int = 0,
        include_description: bool = False,
        min_similarity: float | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._context_chars = context_chars
        self._min_similarity = min_similarity
        records = _label_records(kb)
        self._entity_ids = [eid for eid, _, _ in records]
        self._model = SentenceTransformer(model_name)
        label_texts = [
            f"{label}: {desc}" if (include_description and desc) else label
            for _, label, desc in records
        ]
        kb_matrix = np.asarray(
            self._model.encode(label_texts, batch_size=32, show_progress_bar=False)
        )
        self._kb_matrix = kb_matrix / np.linalg.norm(kb_matrix, axis=1, keepdims=True)

    def _context(self, text: str, a: int, b: int) -> str:
        lo = max(0, a - self._context_chars)
        hi = min(len(text), b + self._context_chars)
        return text[lo:hi]

    def link(self, text: str, a: int, b: int) -> str | None:
        vec = np.asarray(
            self._model.encode([self._context(text, a, b)], show_progress_bar=False)
        )[0]
        vec = vec / np.linalg.norm(vec)
        sims = self._kb_matrix @ vec
        best = int(np.argmax(sims))
        if self._min_similarity is not None and sims[best] < self._min_similarity:
            return None
        return self._entity_ids[best]


class LlmLinkerBaseline:
    """LLM constrained choice over the KB for a marked mention; disk-cached."""

    _SYSTEM_TEMPLATE = (
        "You link predicate mentions in biomedical text to a fixed list of relations.\n\n"
        "Relations (numbered; label in quotes):\n\n"
        "{kb_table}\n\n"
        "The user message contains a passage with ONE mention marked between 【 and 】. "
        "Reply with the single best-matching relation label, copied exactly and without "
        "quotes, or NONE if no relation fits. Reply with only that label."
    )

    def __init__(
        self,
        kb: pd.DataFrame,
        *,
        model: str,
        cache_dir: pathlib.Path | str,
        provider: str = DEFAULT_PROVIDER,
        context_chars: int = 240,
    ) -> None:
        records = _label_records(kb)
        # Labels, not ids, and rendered through the same cleaner the annotator uses: a
        # baseline handicapped by a messier prompt than the system it is compared with
        # is not a baseline, it is a strawman.
        self._label_index = {label.strip().casefold(): eid for eid, label, _ in records}
        self._system = self._SYSTEM_TEMPLATE.replace(
            "{kb_table}", render_kb_catalog(kb)
        )
        self._model_name = model
        self._provider = provider
        self._cache_dir = pathlib.Path(cache_dir)
        self._context_chars = context_chars

    def link(self, text: str, a: int, b: int) -> str | None:
        lo = max(0, a - self._context_chars)
        hi = min(len(text), b + self._context_chars)
        marked = text[lo:a] + "【" + text[a:b] + "】" + text[b:hi]
        raw = complete(
            system=self._system,
            user=marked,
            cache_dir=self._cache_dir,
            provider=self._provider,
            model=self._model_name,
            max_output_tokens=64,
        )
        answer = _FENCE_RE.sub("", raw.strip()).strip().strip('"')
        if not answer or answer.upper() == "NONE":
            return None
        return self._label_index.get(answer.casefold())
