"""Reference baselines for the gold evaluation.

All three baselines share one span proposer (:class:`LexicalSpanProposer`, lemma-window
matching against KB labels — the same idea the linker's stage (A) uses), and differ only
in how a span gets its id:

- :class:`LexicalLemmaBaseline` — the trivial baseline: the id of the KB label whose
  lemma sequence the span matches.
- :class:`EncoderKnnBaseline` — embed the span's context window and the KB's
  ``label: description`` strings with a sentence encoder; cosine top-1.
- :class:`LlmLinkerBaseline` — ask an LLM to pick an id from the full KB for the marked
  mention; disk-cached, so re-scoring is free. Use a different model family than the one
  that pre-annotated the gold.

Heavy imports (spaCy models, sentence-transformers, anthropic) happen at construction or
call time, never at module import.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Any

import numpy as np
import pandas as pd

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
    """Sentence-encoder cosine top-1 against KB ``label: description`` strings."""

    def __init__(
        self,
        kb: pd.DataFrame,
        *,
        model_name: str = "neuml/pubmedbert-base-embeddings",
        context_chars: int = 120,
        min_similarity: float | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._context_chars = context_chars
        self._min_similarity = min_similarity
        records = _label_records(kb)
        self._entity_ids = [eid for eid, _, _ in records]
        self._model = SentenceTransformer(model_name)
        label_texts = [
            f"{label}: {desc}" if desc else label for _, label, desc in records
        ]
        kb_matrix = np.asarray(self._model.encode(label_texts, batch_size=32))
        self._kb_matrix = kb_matrix / np.linalg.norm(kb_matrix, axis=1, keepdims=True)

    def _context(self, text: str, a: int, b: int) -> str:
        lo = max(0, a - self._context_chars)
        hi = min(len(text), b + self._context_chars)
        return text[lo:hi]

    def link(self, text: str, a: int, b: int) -> str | None:
        vec = np.asarray(self._model.encode([self._context(text, a, b)]))[0]
        vec = vec / np.linalg.norm(vec)
        sims = self._kb_matrix @ vec
        best = int(np.argmax(sims))
        if self._min_similarity is not None and sims[best] < self._min_similarity:
            return None
        return self._entity_ids[best]


class LlmLinkerBaseline:
    """LLM constrained choice over the KB for a marked mention; disk-cached."""

    _SYSTEM_TEMPLATE = (
        "You link predicate mentions in biomedical text to a fixed knowledge base of "
        "relation properties.\n\nKnowledge base (entity_id, label, description):\n\n"
        "{kb_table}\n\n"
        "The user message contains a passage with ONE mention marked between 【 and 】. "
        "Reply with the single best-matching entity_id, exactly as written in the "
        "knowledge base, or NONE if no property fits. Reply with only that token."
    )

    def __init__(
        self,
        kb: pd.DataFrame,
        *,
        model: str,
        cache_dir: pathlib.Path | str,
        context_chars: int = 240,
    ) -> None:
        records = _label_records(kb)
        self._kb_ids = {eid for eid, _, _ in records}
        kb_table = "\n".join(f"{eid}\t{label}\t{desc}" for eid, label, desc in records)
        self._system = self._SYSTEM_TEMPLATE.replace("{kb_table}", kb_table)
        self._model_name = model
        self._cache_dir = pathlib.Path(cache_dir)
        self._context_chars = context_chars

    def _cached_completion(self, user: str) -> str:
        key = hashlib.sha256(
            json.dumps([self._model_name, self._system, user]).encode("utf-8")
        ).hexdigest()[:32]
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self._model_name,
            max_tokens=64,
            system=[
                {
                    "type": "text",
                    "text": self._system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"model": self._model_name, "text": text}), encoding="utf-8"
        )
        return text

    def link(self, text: str, a: int, b: int) -> str | None:
        lo = max(0, a - self._context_chars)
        hi = min(len(text), b + self._context_chars)
        marked = text[lo:a] + "【" + text[a:b] + "】" + text[b:hi]
        raw = _FENCE_RE.sub("", self._cached_completion(marked).strip()).strip()
        token = raw.split()[0] if raw.split() else ""
        if token == "NONE" or token not in self._kb_ids:
            return None
        return token
