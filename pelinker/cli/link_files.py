"""Run ``Linker.predict`` on UTF-8 text files (same logic as ``pelinker.cli.server`` / ``/link``)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click

from pelinker.model import Linker
from pelinker.onto import MAX_LENGTH

logger = logging.getLogger(__name__)


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


def _parse_json_documents(
    data: object, source: Path
) -> list[tuple[str, list[dict[str, Any]] | None]]:
    if isinstance(data, dict) and "text" in data:
        text = data["text"]
        if not isinstance(text, str):
            raise ValueError(f"{source}: JSON object 'text' must be a string")
        raw_gt = data.get("ground_truth")
        gt: list[dict[str, Any]] | None
        if raw_gt is None:
            gt = None
        elif isinstance(raw_gt, list):
            gt = []
            for i, item in enumerate(raw_gt):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"{source}: ground_truth[{i}] must be an object with "
                        "char spans (e.g. a, b) and class/entity fields"
                    )
                gt.append(dict(item))
        else:
            raise ValueError(f"{source}: 'ground_truth' must be a list or null")
        return [(text, gt)]

    if isinstance(data, list):
        if not data:
            raise ValueError(f"{source}: JSON array must contain at least one document")
        docs: list[tuple[str, list[dict[str, Any]] | None]] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{source}: item {i} must be an object with a 'text' field"
                )
            if "text" not in item:
                raise ValueError(f"{source}: item {i} is missing required field 'text'")
            text = item["text"]
            if not isinstance(text, str):
                raise ValueError(f"{source}: item {i} 'text' must be a string")
            raw_gt = item.get("ground_truth")
            gt = None
            if raw_gt is not None:
                if not isinstance(raw_gt, list):
                    raise ValueError(
                        f"{source}: item {i} 'ground_truth' must be a list or null"
                    )
                gt = []
                for j, hit in enumerate(raw_gt):
                    if not isinstance(hit, dict):
                        raise ValueError(
                            f"{source}: item {i} ground_truth[{j}] must be an object"
                        )
                    gt.append(dict(hit))
            docs.append((text, gt))
        return docs

    raise ValueError(
        f"{source}: JSON must be an object with 'text' or a list of such objects"
    )


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
    if not stripped or stripped[0] not in "[{":
        return [(raw, None)]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [(raw, None)]
    return _parse_json_documents(data, path)


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
    type=click.Path(path_type=Path),
    required=True,
    help="Linker artifact path (same as Linker.dump / Linker.load, with or without .gz).",
)
@click.option(
    "--thr-score",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum cluster membership score (same role as server thr_score).",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    help="Move transformer heads to CUDA when available.",
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
    type=click.Path(exists=True, readable=True, path_type=Path),
)
def main(
    model_path: Path,
    files: tuple[Path, ...],
    thr_score: float,
    use_gpu: bool,
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

    try:
        raw = linker.predict(
            texts,
            max_length=max_length,
            threshold=0.0,
            use_gpu=use_gpu,
        )
        out = Linker.filter_report(raw, thr_score=thr_score)
    except Exception:
        logger.exception("predict failed")
        raise SystemExit(1)

    if any(g is not None for g in ground_truth_by_doc):
        out["ground_truth"] = ground_truth_by_doc

    click.echo(json.dumps(out, default=str, indent=2))


if __name__ == "__main__":
    main()
