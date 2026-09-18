"""LLM pre-annotation of predicate mentions, for human verification.

Reads the sampler's ``sample_texts.jsonl`` (+ ``sample_manifest.csv`` for roles) and the
property KB, prompts an LLM per abstract, and writes gold-candidate annotations in the
``*.gt.json`` document-list shape that ``pelinker.kb.ground_truth.load_ground_truth_spans``
reads.

Two deliberate design points:

- **The model answers with labels, never ids.** Opaque ids carry no meaning for a
  language model; resolving them here makes an unmatched label a logged rejection
  instead of a plausible-looking wrong id.
- **Only canonical labels are offered**, with orientation carried in ``direction``, so
  a passive mention has exactly one valid encoding.
- **The model never emits character offsets.** LLMs are unreliable at counting
  characters, so the contract is verbatim ``surface`` strings plus a 1-based
  ``occurrence`` index; this driver resolves offsets deterministically and rejects
  anything that does not resolve (rejections land in ``annotate_report.json``, they are
  never silently dropped).
- **Responses are disk-cached** keyed on provider, model and prompts, so re-runs and
  post-hoc parsing fixes are free — parsing happens after the cache, not before.

Requires the ``eval`` extra (``uv sync --extra dev --extra eval``) and a credential for
the chosen provider: Gemini by default (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``). For the
agreement slice, run twice with different ``--provider`` / ``--model`` / ``--annotator``
values and compare with ``gold_review_sheet.py``; a second *model family* makes the κ
more informative than a second checkpoint of the same one.

Usage:

    uv run python run/eval/annotate_llm.py \
        --sample-dir <workdir>/gold \
        --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
        --output-path <workdir>/gold/gold.llm-a.json \
        --roles primary,double
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import click
import pandas as pd

from pelinker.eval.kb_prompt import render_kb_catalog
from pelinker.eval.llm import DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDERS, complete
from pelinker.kb.ground_truth import GT_DIRECTIONS
from pelinker.core.paths import ExpandedPath

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "annotate_predicates.md"
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass(frozen=True)
class PromptParts:
    system: str
    user_template: str


def load_prompt_template(path: Path) -> PromptParts:
    """Split the template file on its ``## System`` / ``## User`` headings."""
    text = path.read_text(encoding="utf-8")
    try:
        _, after_system = text.split("## System", 1)
        system, user = after_system.split("## User", 1)
    except ValueError as err:
        raise click.ClickException(
            f"{path}: expected '## System' and '## User' sections"
        ) from err
    return PromptParts(system=system.strip(), user_template=user.strip())


def canonical_kb(kb: pd.DataFrame) -> pd.DataFrame:
    """Only the canonical member of each converse pair is offered to the annotator.

    A KB without ``is_canonical`` predates the pair derivation, so both members of every
    converse pair would be offered and a passive mention would have two equally valid
    encodings — the ambiguity the canonical vocabulary exists to remove. That is a silent
    change to what the gold means, so it is refused rather than tolerated.
    """
    if "is_canonical" not in kb.columns:
        raise click.ClickException(
            "KB has no 'is_canonical' column: it predates the converse-pair derivation. "
            "Build it with run/preprocessing/derive_inverse_pairs.py and pass "
            "data/derived/properties.synthesis.2.pairs.csv."
        )
    return kb.loc[kb["is_canonical"].astype(bool)]


def kb_table_text(kb: pd.DataFrame) -> str:
    """Numbered relation catalog for the prompt (see :mod:`pelinker.eval.kb_prompt`).

    No ids: they mean nothing to the model, and transcribing them is a copying task that
    fails silently. The converse form is shown so the model can recognise a passive
    mention as this relation running the other way rather than reaching for a different
    label.
    """
    return render_kb_catalog(kb)


def build_label_index(kb: pd.DataFrame) -> dict[str, tuple[str, bool]]:
    """Normalized label -> (canonical entity id, needs_direction_flip).

    Built from the **whole** KB, not just the canonical half that the prompt offers. A
    model that answers with a converse label ("expressed in" for "expresses") has made a
    recoverable presentation error, not a semantic one: the relation is right and the
    orientation is the other way round. Dropping those answers would discard converse-form
    mentions specifically — the exact population the directionality analysis needs — so
    they are mapped onto the canonical id with the direction flipped, and flagged.
    """
    index: dict[str, tuple[str, bool]] = {}
    for _, r in kb.iterrows():
        label = str(r["label"]).strip().casefold()
        canonical = r.get("canonical_entity_id")
        entity_id = str(r["entity_id"])
        if isinstance(canonical, str) and canonical:
            flip = canonical != entity_id
            index[label] = (canonical, flip)
        else:
            index[label] = (entity_id, False)
    return index


_OPPOSITE_DIRECTION = {"forward": "inverse", "inverse": "forward"}


def flip_direction(direction: str | None) -> str | None:
    """Swap forward/inverse; ``symmetric`` and ``na`` have no opposite."""
    if direction is None:
        return None
    return _OPPOSITE_DIRECTION.get(direction, direction)


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def iter_occurrences(text: str, surface: str) -> list[tuple[int, int]]:
    """All word-boundary-respecting occurrences of ``surface`` in ``text``.

    Plain substring search silently mislocates a short answer inside a longer word —
    "reduce" inside "reduces", "related to" inside "correlated to" — producing a span that
    looks valid and points at the wrong token. Requiring word boundaries turns that into a
    visible miss instead.
    """
    if not surface:
        return []
    spans: list[tuple[int, int]] = []
    start = text.find(surface)
    while start >= 0:
        end = start + len(surface)
        left_ok = start == 0 or not (
            _is_word_char(text[start - 1]) and _is_word_char(surface[0])
        )
        right_ok = end == len(text) or not (
            _is_word_char(text[end]) and _is_word_char(surface[-1])
        )
        if left_ok and right_ok:
            spans.append((start, end))
        start = text.find(surface, start + 1)
    return spans


def find_occurrence(
    text: str,
    surface: str,
    occurrence: int,
    *,
    claimed: set[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int] | None, bool]:
    """Resolve the ``occurrence``-th (1-based) ``surface``; returns (span, used_fallback).

    Models count occurrences unreliably — far more unreliably than they quote text — so a
    requested index that does not exist falls back to the first unclaimed occurrence of
    the same surface. The surface itself is still required verbatim: this recovers the
    *position*, never the content, and the fallback is recorded on the hit.
    """
    if not surface or occurrence < 1:
        return None, False
    spans = iter_occurrences(text, surface)
    if not spans:
        return None, False
    taken = claimed or set()
    if occurrence <= len(spans):
        candidate = spans[occurrence - 1]
        if candidate not in taken:
            return candidate, False
    # Either the index does not exist, or it points at a span an earlier record already
    # took — models re-emit the same mention with the same index. Fall back to the first
    # unclaimed occurrence; when every occurrence is spoken for, the record is a
    # duplicate and is rejected rather than annotated twice.
    for span in spans:
        if span not in taken:
            return span, True
    return None, False


def nearest_occurrence(text: str, needle: str, anchor: int) -> tuple[int, int] | None:
    """Span of the occurrence of ``needle`` closest to char position ``anchor``."""
    if not needle:
        return None
    spans: list[tuple[int, int]] = []
    start = text.find(needle)
    while start >= 0:
        spans.append((start, start + len(needle)))
        start = text.find(needle, start + 1)
    if not spans:
        return None
    return min(spans, key=lambda s: abs(s[0] - anchor))


def parse_llm_response(raw: str) -> list[dict]:
    """Parse the model's JSON array, tolerating stray code fences."""
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array")
    return [d for d in data if isinstance(d, dict)]


def resolve_annotations(
    text: str,
    items: list[dict],
    *,
    label_index: dict[str, tuple[str, bool]],
    annotator: str,
) -> tuple[list[dict], list[dict]]:
    """LLM records → offset-resolved gt hits; unresolvable records are reported.

    Two recoveries are applied and flagged rather than dropped, because dropping either
    would bias the gold set rather than merely shrink it: a mis-counted ``occurrence``
    (which loses repeated mentions) and a converse label used in place of its canonical
    partner (which loses inverse-direction mentions specifically).
    """
    hits: list[dict] = []
    rejected: list[dict] = []
    claimed: set[tuple[int, int]] = set()
    for item in items:
        surface = str(item.get("surface") or "")
        occurrence = int(item.get("occurrence") or 1)
        label = str(item.get("label") or "")
        resolved = label_index.get(label.strip().casefold())
        direction = item.get("direction")
        reasons = []
        span, occurrence_fallback = find_occurrence(
            text, surface, occurrence, claimed=claimed
        )
        if span is None:
            reasons.append("surface_not_found")
        if resolved is None:
            reasons.append("unknown_label")
        if direction is not None and direction not in GT_DIRECTIONS:
            reasons.append("bad_direction")
        if reasons:
            rejected.append({**item, "reasons": reasons})
            continue
        assert span is not None and resolved is not None
        entity_id, needs_flip = resolved
        if needs_flip:
            direction = flip_direction(direction)
        a, b = span
        claimed.add(span)
        hit: dict = {
            "a": a,
            "b": b,
            "entity_id": entity_id,
            "label": label,
            "surface": surface,
            "direction": direction,
            "annotator": annotator,
            "source": "llm",
        }
        if occurrence_fallback:
            hit["occurrence_fallback"] = True
        if needs_flip:
            hit["label_canonicalized"] = True
        if item.get("confidence") is not None:
            hit["confidence"] = float(item["confidence"])
        for key, field in (
            ("subject_text", "subject_span"),
            ("object_text", "object_span"),
        ):
            arg = item.get(key)
            if arg:
                arg_span = nearest_occurrence(text, str(arg), a)
                if arg_span is not None:
                    hit[field] = list(arg_span)
        hits.append(hit)

    hits, nested = drop_nested_spans(hits)
    rejected.extend(nested)
    return hits, rejected


def drop_nested_spans(hits: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep one annotation per mention site: the longest span wins.

    A model routinely emits both the bare verb and the verb-plus-preposition for one site
    ("expressed" and "expressed in"). Keeping both would put two labels on one mention —
    the very defect this gold set exists to measure in the pipeline's own supervision.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    ordered = sorted(hits, key=lambda h: h["b"] - h["a"], reverse=True)
    for hit in ordered:
        if any(k["a"] <= hit["a"] and hit["b"] <= k["b"] for k in kept):
            dropped.append({**hit, "reasons": ["nested_in_longer_span"]})
        else:
            kept.append(hit)
    kept.sort(key=lambda h: (h["a"], h["b"]))
    return kept, dropped


@click.command()
@click.option("--sample-dir", required=True, type=ExpandedPath(exists=True))
@click.option("--kb-csv-path", required=True, type=ExpandedPath(exists=True))
@click.option("--output-path", required=True, type=ExpandedPath())
@click.option(
    "--provider",
    type=click.Choice(PROVIDERS),
    default=DEFAULT_PROVIDER,
    show_default=True,
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option(
    "--annotator",
    default=None,
    help="Annotator tag recorded on every hit (default: the model id).",
)
@click.option(
    "--roles",
    default="primary,double",
    show_default=True,
    help="Comma-separated manifest roles to annotate.",
)
@click.option(
    "--cache-dir",
    default=None,
    type=ExpandedPath(file_okay=False),
    help="LLM response cache (default: <sample-dir>/.llm_cache).",
)
@click.option("--prompt-path", default=str(_PROMPT_PATH), show_default=True)
@click.option("--max-tokens", default=8000, show_default=True)
@click.option("--limit", default=None, type=int, help="Annotate only the first N docs.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print request sizes and exit without calling the LLM.",
)
def main(
    sample_dir: str,
    kb_csv_path: str,
    output_path: str,
    provider: str,
    model: str,
    annotator: str | None,
    roles: str,
    cache_dir: str | None,
    prompt_path: str,
    max_tokens: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    annotator = annotator or model
    sample = Path(sample_dir)
    cache = Path(cache_dir) if cache_dir else sample / ".llm_cache"

    manifest = pd.read_csv(sample / "sample_manifest.csv")
    wanted_roles = {r.strip() for r in roles.split(",") if r.strip()}
    wanted_ids = set(
        manifest.loc[manifest["role"].isin(wanted_roles), "doc_id"].astype(int)
    )

    docs: list[dict] = []
    with (sample / "sample_texts.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if int(rec["doc_id"]) in wanted_ids:
                docs.append(rec)
    if limit is not None:
        docs = docs[:limit]
    logger.info("Annotating %d docs (roles: %s)", len(docs), sorted(wanted_roles))

    kb_all = pd.read_csv(kb_csv_path)
    kb = canonical_kb(kb_all)
    # Prompt: canonical only. Decoder: the whole KB, so a converse answer is recovered
    # (with its direction flipped) instead of discarded.
    label_index = build_label_index(kb_all)
    logger.info(
        "Offering %d  canonical relation labels (decoder accepts %d incl. converse forms)",
        len(kb),
        len(label_index),
    )
    prompt = load_prompt_template(Path(prompt_path))
    system = prompt.system.replace("{kb_table}", kb_table_text(kb))

    if dry_run:
        user_chars = [len(prompt.user_template) + len(d["text"]) for d in docs]
        logger.info(
            "Dry run: system=%d chars (cached prefix), %d requests, "
            "user chars total=%d mean=%.0f",
            len(system),
            len(docs),
            sum(user_chars),
            (sum(user_chars) / len(user_chars)) if user_chars else 0.0,
        )
        return

    out_docs: list[dict] = []
    all_rejected: list[dict] = []
    n_hits = 0
    for itext, doc in enumerate(docs):
        text = doc["text"]
        user = prompt.user_template.replace("{abstract}", text)
        raw = complete(
            system=system,
            user=user,
            cache_dir=cache,
            provider=provider,
            model=model,
            max_output_tokens=max_tokens,
            # The prompt asks for a JSON array; have the provider guarantee that shape
            # rather than parsing it back out of prose.
            json_mode=True,
        )
        try:
            items = parse_llm_response(raw)
        except (ValueError, json.JSONDecodeError):
            logger.warning("doc_id=%s: unparsable response", doc["doc_id"])
            all_rejected.append({"doc_id": doc["doc_id"], "reasons": ["unparsable"]})
            items = []
        hits, rejected = resolve_annotations(
            text, items, label_index=label_index, annotator=annotator
        )
        for r in rejected:
            all_rejected.append({"doc_id": doc["doc_id"], **r})
        for h in hits:
            h["itext"] = itext
        n_hits += len(hits)
        out_docs.append(
            {
                "doc_id": doc["doc_id"],
                # Provenance travels with the annotation: a gold span whose document
                # cannot be cited is not usable evidence.
                "doc_uid": doc.get("doc_uid"),
                "ids": doc.get("ids", {}),
                "text": text,
                "ground_truth": hits,
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_docs, ensure_ascii=False, indent=1), encoding="utf-8")
    report = {
        "provider": provider,
        "model": model,
        "annotator": annotator,
        "n_docs": len(out_docs),
        "n_hits": n_hits,
        "n_rejected": len(all_rejected),
        "rejected": all_rejected,
    }
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    logger.info(
        "Wrote %d docs / %d hits to %s (%d rejected records → %s)",
        len(out_docs),
        n_hits,
        out,
        len(all_rejected),
        report_path,
    )


if __name__ == "__main__":
    main()
