"""Run ``Linker.predict`` on UTF-8 text files (same logic as ``pelinker.cli.server`` / ``/link``)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from pelinker.model import Linker
from pelinker.onto import MAX_LENGTH

logger = logging.getLogger(__name__)


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
    """Load a dumped Linker and predict entities for each file (one text per file)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    texts: list[str] = []
    for fp in files:
        try:
            texts.append(fp.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.error("Cannot read %s: %s", fp, exc)
            raise SystemExit(1) from exc

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

    click.echo(json.dumps(out, default=str, indent=2))


if __name__ == "__main__":
    main()
