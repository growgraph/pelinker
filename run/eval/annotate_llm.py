"""LLM pre-annotation of predicate mentions, for human verification.

Reads the sampler's ``sample_texts.jsonl`` (+ ``sample_manifest.csv`` for roles) and the
property KB, prompts an LLM per abstract, and writes gold-candidate annotations in the
``*.gt.json`` document-list shape that ``pelinker.kb.ground_truth.load_ground_truth_spans``
reads.

Two deliberate design points:

- **The model never emits character offsets.** LLMs are unreliable at counting
  characters, so the contract is verbatim ``surface`` strings plus a 1-based
  ``occurrence`` index; this driver resolves offsets deterministically and rejects
  anything that does not resolve (rejections land in ``annotate_report.json``, they are
  never silently dropped).
- **Responses are disk-cached** keyed on (model, system prompt, user prompt), so re-runs
  and post-hoc parsing fixes are free — parsing happens after the cache, not before.

Requires the ``eval`` extra (``uv sync --extra dev --extra eval``) and an Anthropic
credential in the environment. For the agreement slice, run twice with different
``--model`` / ``--annotator`` values and compare with ``gold_review_sheet.py``.

Usage:

    uv run python run/eval/annotate_llm.py \
        --sample-dir <workdir>/gold \
        --kb-csv-path data/derived/properties.synthesis.2.csv \
        --output-path <workdir>/gold/gold.llm-a.json \
        --roles primary,double
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import click
import pandas as pd

from pelinker.kb.ground_truth import GT_DIRECTIONS

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


def kb_table_tsv(kb: pd.DataFrame) -> str:
    rows = []
    for _, r in kb.iterrows():
        desc = "" if pd.isna(r.get("description")) else str(r["description"])
        rows.append(f"{r['entity_id']}\t{r['label']}\t{desc}")
    return "\n".join(rows)


def find_occurrence(text: str, surface: str, occurrence: int) -> tuple[int, int] | None:
    """Char span of the ``occurrence``-th (1-based) verbatim ``surface`` in ``text``."""
    if not surface or occurrence < 1:
        return None
    start = -1
    for _ in range(occurrence):
        start = text.find(surface, start + 1)
        if start < 0:
            return None
    return start, start + len(surface)


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
    kb_ids: set[str],
    annotator: str,
) -> tuple[list[dict], list[dict]]:
    """LLM records → offset-resolved gt hits; unresolvable records are reported."""
    hits: list[dict] = []
    rejected: list[dict] = []
    for item in items:
        surface = str(item.get("surface") or "")
        occurrence = int(item.get("occurrence") or 1)
        entity_id = str(item.get("entity_id") or "")
        direction = item.get("direction")
        reasons = []
        span = find_occurrence(text, surface, occurrence)
        if span is None:
            reasons.append("surface_not_found")
        if entity_id not in kb_ids:
            reasons.append("unknown_entity_id")
        if direction is not None and direction not in GT_DIRECTIONS:
            reasons.append("bad_direction")
        if reasons:
            rejected.append({**item, "reasons": reasons})
            continue
        assert span is not None
        a, b = span
        hit: dict = {
            "a": a,
            "b": b,
            "entity_id": entity_id,
            "surface": surface,
            "direction": direction,
            "annotator": annotator,
            "source": "llm",
        }
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
    return hits, rejected


def cached_completion(
    *,
    cache_dir: Path,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> str:
    """Call the LLM once per unique (model, system, user); replay from disk after."""
    key = hashlib.sha256(
        json.dumps([model, system, user], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:32]
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system,
                # The KB table is a large stable prefix shared by every abstract.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "model": model,
                "text": text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return text


@click.command()
@click.option("--sample-dir", required=True, type=click.Path(exists=True))
@click.option("--kb-csv-path", required=True, type=click.Path(exists=True))
@click.option("--output-path", required=True, type=click.Path())
@click.option("--model", default="claude-opus-5", show_default=True)
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

    kb = pd.read_csv(kb_csv_path)
    kb_ids = set(kb["entity_id"].astype(str))
    prompt = load_prompt_template(Path(prompt_path))
    system = prompt.system.replace("{kb_table}", kb_table_tsv(kb))

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
        raw = cached_completion(
            cache_dir=cache,
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
        )
        try:
            items = parse_llm_response(raw)
        except (ValueError, json.JSONDecodeError):
            logger.warning("doc_id=%s: unparsable response", doc["doc_id"])
            all_rejected.append({"doc_id": doc["doc_id"], "reasons": ["unparsable"]})
            items = []
        hits, rejected = resolve_annotations(
            text, items, kb_ids=kb_ids, annotator=annotator
        )
        for r in rejected:
            all_rejected.append({"doc_id": doc["doc_id"], **r})
        for h in hits:
            h["itext"] = itext
        n_hits += len(hits)
        out_docs.append(
            {
                "doc_id": doc["doc_id"],
                "pmid": doc.get("pmid"),
                "text": text,
                "ground_truth": hits,
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_docs, ensure_ascii=False, indent=1), encoding="utf-8")
    report = {
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
