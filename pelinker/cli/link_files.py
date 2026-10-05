"""Run ``Linker.predict`` on UTF-8 text files (same logic as ``pelinker.cli.server`` / ``/link``)."""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

import click
import pandas as pd

from pelinker.kb.ground_truth import (
    GroundTruthScore,
    GtSpan,
    score_predictions_against_ground_truth,
)
from pelinker.kb.kb_out import kb_out_to_kb_in_map
from pelinker.linker.kb_lemma import (
    KbLemmaValidationMetrics,
    aggregate_kb_lemma_validation,
)
from pelinker.model import DEFAULT_CLUSTER_MEMBERSHIP_THRESHOLD, Linker
from pelinker.core.onto import MAX_LENGTH
from pelinker.core.paths import ExpandedPath

logger = logging.getLogger(__name__)


_MENTION_ANOMALY_FORMATS = {".parquet", ".csv", ".jsonl"}
_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
_DOCUMENT_CONTAINER_KEYS = ("documents", "docs", "items", "records", "data")
_TEXT_FIELD_KEYS = ("text", "content", "body", "document")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace Enum keys (e.g. WordGrouping) and Enum values for ``json.dumps``."""
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            key = k.name if isinstance(k, Enum) else k
            out[key] = _sanitize_for_json(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, Enum):
        return obj.name
    return obj


def _write_mention_anomaly(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-mention anomaly rows. Format inferred from ``path`` extension."""
    suffix = path.suffix.lower()
    if suffix not in _MENTION_ANOMALY_FORMATS:
        raise ValueError(
            f"Unsupported --dump-mention-anomaly extension {suffix!r}; "
            f"expected one of {sorted(_MENTION_ANOMALY_FORMATS)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str))
                fh.write("\n")
        return
    df = pd.DataFrame(rows)
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _fmt_rate(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _score_against_ground_truth(
    out: dict[str, Any],
    ground_truth_by_doc: list[list[dict[str, Any]] | None],
    predicted_id_to_kb_in: dict[str, str] | None = None,
) -> "GroundTruthScore | None":
    """Score emitted entities against the char-offset gold spans, if any are present."""
    gold: list[GtSpan] = []
    for doc_index, hits in enumerate(ground_truth_by_doc):
        for hit in hits or []:
            try:
                gold.append(
                    GtSpan(
                        itext=int(hit.get("itext", doc_index)),
                        a=int(hit["a"]),
                        b=int(hit["b"]),
                        entity_id=(
                            None
                            if hit.get("entity_id") is None
                            else str(hit["entity_id"])
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                # A malformed hit should not sink the whole run; it is simply unscorable.
                logger.warning("Skipping unparsable ground-truth hit: %r", hit)
    if not gold:
        return None
    entities = out.get("entities") or []
    return score_predictions_against_ground_truth(
        entities, gold, predicted_id_to_kb_in=predicted_id_to_kb_in
    )


def _aggregate_lemma_validation(pres: object) -> "KbLemmaValidationMetrics | None":
    """Roll the per-row KB-lemma flag into a rate, when the fields were computed."""
    entities = getattr(pres, "entities", None)
    if not entities:
        return None
    rows = [dict(r) for r in entities]
    if not any("kb_training_entity_from_lemma" in r for r in rows):
        return None  # --kb-validation was not requested
    return aggregate_kb_lemma_validation(rows)


def _normalize_ground_truth_hits(
    hits: list[dict[str, Any]], doc_index: int
) -> list[dict[str, Any]]:
    """Ensure each hit has ``itext`` set to the global document index."""
    out: list[dict[str, Any]] = []
    for hit in hits:
        row = dict(hit)
        row["itext"] = doc_index
        out.append(row)
    return out


def _parse_ground_truth(
    raw_gt: object, source: Path, context: str
) -> list[dict[str, Any]] | None:
    if raw_gt is None:
        return None
    if not isinstance(raw_gt, list):
        raise ValueError(f"{source}: {context} 'ground_truth' must be a list or null")
    gt: list[dict[str, Any]] = []
    for j, hit in enumerate(raw_gt):
        if not isinstance(hit, dict):
            raise ValueError(f"{source}: {context} ground_truth[{j}] must be an object")
        gt.append(dict(hit))
    return gt


def _parse_document_item(
    item: object, source: Path, context: str
) -> tuple[str, list[dict[str, Any]] | None]:
    if isinstance(item, str):
        return item, None
    if not isinstance(item, dict):
        raise ValueError(
            f"{source}: {context} must be an object with a text field or a string document"
        )

    text_field_keys = [k for k in _TEXT_FIELD_KEYS if k in item]
    if not text_field_keys:
        raise ValueError(
            f"{source}: {context} is missing a text field "
            f"(supported: {', '.join(_TEXT_FIELD_KEYS)})"
        )
    if len(text_field_keys) > 1:
        raise ValueError(
            f"{source}: {context} has multiple text fields {text_field_keys}; use exactly one"
        )

    text = item[text_field_keys[0]]
    if not isinstance(text, str):
        raise ValueError(
            f"{source}: {context} '{text_field_keys[0]}' field must be a string"
        )
    gt = _parse_ground_truth(item.get("ground_truth"), source, context)
    return text, gt


def _parse_json_documents(
    data: object, source: Path
) -> list[tuple[str, list[dict[str, Any]] | None]]:
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{source}: JSON array must contain at least one document")
        docs: list[tuple[str, list[dict[str, Any]] | None]] = []
        for i, item in enumerate(data):
            docs.append(_parse_document_item(item, source, f"item {i}"))
        return docs

    if isinstance(data, dict):
        # Single-document objects can use one of the accepted text fields.
        if any(k in data for k in _TEXT_FIELD_KEYS):
            return [_parse_document_item(data, source, "JSON object")]

        for container_key in _DOCUMENT_CONTAINER_KEYS:
            if container_key in data:
                return _parse_json_documents(data[container_key], source)

        if "texts" in data:
            return _parse_json_documents(data["texts"], source)

    raise ValueError(
        f"{source}: JSON must be a document object, a list of documents/strings, "
        f"or a wrapper containing one of {', '.join(_DOCUMENT_CONTAINER_KEYS)} / texts"
    )


def _parse_jsonl_documents(
    raw: str, source: Path
) -> list[tuple[str, list[dict[str, Any]] | None]]:
    items: list[object] = []
    for idx, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid JSONL at line {idx}: {exc}") from exc
        items.append(item)
    return _parse_json_documents(items, source)


def _load_documents_from_file(
    path: Path,
) -> list[tuple[str, list[dict[str, Any]] | None]]:
    """
    Load one or more documents from a path.

    If the file parses as JSON and matches the supported shape, returns structured
    documents. Otherwise treats the entire file as a single UTF-8 text document.
    """
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        return [(raw, None)]

    try:
        if path.suffix.lower() in _JSONL_SUFFIXES:
            return _parse_jsonl_documents(raw, path)
        data = json.loads(raw)
        return _parse_json_documents(data, path)
    except (json.JSONDecodeError, ValueError):
        return [(raw, None)]


def _flatten_inputs(
    files: tuple[Path, ...],
) -> tuple[list[str], list[list[dict[str, Any]] | None]]:
    texts: list[str] = []
    ground_truth_by_doc: list[list[dict[str, Any]] | None] = []
    doc_index = 0
    for fp in files:
        try:
            chunks = _load_documents_from_file(fp)
        except OSError as exc:
            logger.error("Cannot read %s: %s", fp, exc)
            raise SystemExit(1) from exc
        except ValueError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc

        for text, gt in chunks:
            texts.append(text)
            if gt is not None:
                ground_truth_by_doc.append(_normalize_ground_truth_hits(gt, doc_index))
            else:
                ground_truth_by_doc.append(None)
            doc_index += 1

    return texts, ground_truth_by_doc


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-m",
    "--model",
    "model_path",
    type=ExpandedPath(path_type=Path),
    required=True,
    help="Linker artifact path (same as Linker.dump / Linker.load, with or without .gz).",
)
@click.option(
    "--thr-score",
    type=float,
    default=DEFAULT_CLUSTER_MEMBERSHIP_THRESHOLD,
    show_default=True,
    help=(
        "Minimum cluster membership score for emitted entities "
        "(passed to Linker.predict threshold; same role as server thr_score)."
    ),
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Move transformer heads to CUDA when available.",
)
@click.option(
    "--include-anomaly-metrics",
    is_flag=True,
    help="Include PCA residual / Mahalanobis anomaly metrics in entity outputs.",
)
@click.option(
    "--kb-validation",
    is_flag=True,
    help="Include kb matching",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=ExpandedPath(path_type=Path),
    default=None,
    help="Write the entity report JSON (UTF-8) to this path.",
)
@click.option(
    "--dump-mention-anomaly",
    "dump_mention_anomaly",
    type=ExpandedPath(path_type=Path),
    default=None,
    help=(
        "If set, write one row per extracted mention with is_kb_match and PCA anomaly "
        "metrics (residual / Mahalanobis / max-z). Format inferred from extension: "
        ".parquet, .csv, .jsonl."
    ),
)
@click.option(
    "--max-length",
    type=int,
    default=MAX_LENGTH,
    show_default=True,
    help="Tokenizer chunk length.",
)
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=ExpandedPath(exists=True, readable=True, path_type=Path),
)
def main(
    model_path: Path,
    files: tuple[Path, ...],
    thr_score: float,
    use_gpu: bool,
    include_anomaly_metrics: bool,
    kb_validation: bool,
    output_path: Path | None,
    dump_mention_anomaly: Path | None,
    max_length: int,
) -> None:
    """Load a dumped Linker and predict entities for each input.

    Inputs are UTF-8 files. If a file parses as JSON, supported shapes are:

    \\b
      - A single object: {"text": "...", "ground_truth": [ ... optional hits ... ]}
      - A list of objects: [{"text": "...", "ground_truth": [...]}, ...]

    Each optional ``ground_truth`` hit is typically an object with character offsets
    ``a``, ``b`` and a class / entity id (e.g. ``entity_id``). ``itext`` in the file
    is ignored and rewritten to match the global document index in the output.

    Any file that is not valid structured JSON (or does not start with ``{`` / ``[``)
    is read as plain text (one document per file).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    texts, ground_truth_by_doc = _flatten_inputs(files)
    if not texts:
        logger.error("No input documents after parsing files")
        raise SystemExit(1)

    try:
        linker = Linker.load(model_path)
    except FileNotFoundError:
        logger.exception("Model not found (expected .gz next to the given path)")
        raise SystemExit(1)

    want_mention_dump = dump_mention_anomaly is not None
    try:
        pres = linker.predict(
            texts,
            max_length=max_length,
            threshold=thr_score,
            use_gpu=use_gpu,
            include_mention_anomaly=want_mention_dump,
            include_prediction_kb_validation=kb_validation,
        )
        filtered = pres.filter_by_score(thr_score)
        public_entity_fields = not include_anomaly_metrics and not kb_validation
        out = filtered.to_dict(
            include_entity_anomaly_metrics=include_anomaly_metrics,
            public_entity_fields=public_entity_fields,
        )
    except Exception:
        logger.exception("predict failed")
        raise SystemExit(1)

    if want_mention_dump:
        try:
            rows = list(pres.debug_mentions) if pres.debug_mentions is not None else []
            _write_mention_anomaly(dump_mention_anomaly, rows)
        except Exception:
            logger.exception("mention anomaly dump failed")
            raise SystemExit(1)

    if any(g is not None for g in ground_truth_by_doc):
        out["ground_truth"] = ground_truth_by_doc
        # Previously the ground truth was parsed and echoed but never scored; the README
        # pointed at a scoring script that does not exist. Score it here instead.
        # The KB-out catalog translates minted cluster ids back onto input-KB ids so
        # entity accuracy is computable, not permanently undefined.
        id_bridge = (
            kb_out_to_kb_in_map(linker.kb_out_catalog)
            if linker.kb_out_catalog is not None
            else None
        )
        score = _score_against_ground_truth(out, ground_truth_by_doc, id_bridge)
        if score is not None:
            out["ground_truth_score"] = score.to_jsonable()
            logger.info(
                "Ground truth: matched %d/%d gold spans (P=%s R=%s F1=%s); "
                "entity accuracy %s over %d comparable pairs",
                score.n_matched,
                score.n_gold,
                _fmt_rate(score.precision),
                _fmt_rate(score.recall),
                _fmt_rate(score.f1),
                _fmt_rate(score.entity_accuracy),
                score.n_id_comparable,
            )
            if score.n_id_comparable == 0 and score.n_matched > 0:
                logger.info(
                    "Entity accuracy is undefined: no predicted id could be mapped "
                    "onto an input-KB id (old artifact without a KB-out catalog, or "
                    "every matched cluster is negative-dominated). Detection "
                    "precision/recall above are still meaningful."
                )

    lemma_metrics = _aggregate_lemma_validation(pres)
    if lemma_metrics is not None:
        out["kb_lemma_validation"] = lemma_metrics.to_jsonable()
        logger.info(
            "KB-lemma consistency: %s over %d resolvable of %d rows",
            _fmt_rate(lemma_metrics.match_rate),
            lemma_metrics.n_resolvable,
            lemma_metrics.n_rows,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_sanitize_for_json(out), ensure_ascii=False)
        output_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
