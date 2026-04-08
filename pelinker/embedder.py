import os
import pathlib
from collections.abc import Sequence

import logging

import pandas as pd
import spacy
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from pelinker.config import (
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    EmbeddingTrainingConfig,
)
from pelinker.ops import _detect_file_format, _detect_headers_and_columns
from pelinker.util import expand_config_path, extract_and_embed_mentions, load_models
from pelinker.onto import WordGrouping
from pelinker.util import str2layers
from pelinker.io.parquet import ParquetWriter

logger = logging.getLogger(__name__)


def _progress_disabled() -> bool:
    """Set ``PELINKER_NO_PROGRESS=1`` to turn off the embed progress bar (e.g. CI logs)."""
    return os.environ.get("PELINKER_NO_PROGRESS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _embed_corpus_single_source(
    *,
    source: EmbeddingSourceSpec,
    training: EmbeddingTrainingConfig,
    output_parquet_path: pathlib.Path,
    use_gpu: bool,
) -> None:
    """Run one encoder over the corpus and append rows to a single parquet file."""
    logger.info(
        "Embedding source model_type=%r layers_spec=%r",
        source.model_type,
        source.layers_spec,
    )

    tokenizer, model = load_models(source.model_type, sentence=False)
    if use_gpu:
        model.to("cuda")
        spacy.require_gpu()

    nlp = spacy.load(training.nlp_model)
    layers = str2layers(source.layers_spec)

    df_props = pd.read_csv(training.kb_csv_path)
    entities = df_props["label"].tolist()

    logger.info("Loaded %s properties", df_props.shape[0])
    logger.info("Layers are set to %s", layers)

    file_format = _detect_file_format(training.input_text_table_path)
    has_header, pmid_col, text_col = _detect_headers_and_columns(
        training.input_text_table_path, file_format
    )

    logger.info("Detected file format: %s", file_format.upper())
    logger.info("Headers detected: %s", has_header)
    logger.info("Using columns: %r, %r", pmid_col, text_col)

    if training.max_input_buffers is not None:
        logger.info(
            "Processing only first %s text-table read passes (max_input_buffers=%s)",
            training.max_input_buffers,
            training.max_input_buffers,
        )
    else:
        logger.info("Processing full text table (all read passes)")

    compression = (
        "gzip" if training.input_text_table_path.suffix.endswith(".gz") else None
    )
    sep = "\t" if file_format == "tsv" else ","

    reader = pd.read_csv(
        training.input_text_table_path,
        sep=sep,
        header=0 if has_header else None,
        compression=compression,
        chunksize=training.input_buffer_rows,
    )

    expanded_out = expand_config_path(output_parquet_path)
    if expanded_out is None:
        raise ValueError("output_parquet_path resolved to None")
    output_parquet_path = pathlib.Path(expanded_out)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_writer = ParquetWriter(output_parquet_path)

    # Rich defaults to "no animations" when isatty() is false (Hydra, IDE runners).
    # force_terminal=True keeps the bar working there; use PELINKER_NO_PROGRESS=1 to
    # disable (clean logs when stderr is a file).
    show_progress = not _progress_disabled()
    console = Console(
        stderr=True,
        width=180,
        legacy_windows=False,
        force_terminal=show_progress,
    )

    try:
        total_processed = 0
        cumulative_mentions = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=8,
            transient=False,
            disable=not show_progress,
        ) as progress:
            max_buf = training.max_input_buffers
            if max_buf is not None:
                task_id = progress.add_task(
                    "[cyan]Embed[/cyan] · starting…",
                    total=float(max_buf),
                )
            else:
                task_id = progress.add_task(
                    "[cyan]Embed[/cyan] · starting…",
                    total=None,
                )

            bs = training.encoder_batch_size

            for i, chunk in enumerate(reader):
                if max_buf is not None and i >= max_buf:
                    logger.info(
                        "Reached max_input_buffers=%s table read passes, stopping",
                        max_buf,
                    )
                    break

                chunk_texts = list(chunk[text_col])
                pmids = [str(pmid) for pmid in chunk[pmid_col]]
                buf_label = f"{i + 1}/{max_buf}" if max_buf is not None else str(i + 1)
                n_enc = (len(chunk_texts) + bs - 1) // bs if chunk_texts else 0

                def on_encoder_batch(
                    ibatch: int,
                    n_enc_batches: int,
                    n_rows_in_buffer: int,
                ) -> None:
                    texts_in_buf = min((ibatch + 1) * bs, len(chunk_texts))
                    texts_cum = total_processed + texts_in_buf
                    mentions_cum = cumulative_mentions + n_rows_in_buffer
                    desc = (
                        f"[cyan]Embed[/cyan] · buf {buf_label} · "
                        f"encoder batch {ibatch + 1}/{n_enc_batches} · "
                        f"texts {texts_cum} · mentions {mentions_cum}"
                    )
                    if max_buf is not None and n_enc_batches:
                        progress.update(
                            task_id,
                            completed=float(i) + (ibatch + 1) / n_enc_batches,
                            description=desc,
                        )
                    else:
                        progress.update(task_id, description=desc)

                if n_enc == 0:
                    rows_data = []
                    if max_buf is not None:
                        progress.update(
                            task_id,
                            completed=float(i + 1),
                            description=(
                                f"[cyan]Embed[/cyan] · buf {buf_label} · "
                                f"encoder batch 0/0 · texts {total_processed} · "
                                f"mentions {cumulative_mentions}"
                            ),
                        )
                    else:
                        progress.update(
                            task_id,
                            description=(
                                f"[cyan]Embed[/cyan] · buf {buf_label} · "
                                f"encoder batch 0/0 · texts {total_processed} · "
                                f"mentions {cumulative_mentions}"
                            ),
                        )
                else:
                    rows_data = extract_and_embed_mentions(
                        entities=entities,
                        data=chunk_texts,
                        batch_size=bs,
                        pmids=pmids,
                        tokenizer=tokenizer,
                        model=model,
                        nlp=nlp,
                        layers=layers,
                        word_modes=(
                            WordGrouping.W1,
                            WordGrouping.W2,
                            WordGrouping.W3,
                        ),
                        on_encoder_batch=on_encoder_batch,
                    )

                if rows_data:
                    parquet_writer.write_batch(rows_data)

                total_processed += len(chunk_texts)
                cumulative_mentions += len(rows_data)
                logger.debug(
                    "Buffer %s: processed %s texts (%s mentions this buffer); running total texts: %s",
                    i,
                    len(chunk_texts),
                    len(rows_data),
                    total_processed,
                )

                del rows_data

            if max_buf is not None:
                task_state = progress.tasks[task_id]
                if 0.0 < task_state.completed < (task_state.total or 0.0):
                    progress.update(task_id, total=task_state.completed)

    finally:
        parquet_writer.close()
        logger.info("Output saved to %s", output_parquet_path)


def embed_kb_corpus(
    *,
    metadata: EmbeddingModelMetadata,
    training: EmbeddingTrainingConfig,
    output_parquet_path: pathlib.Path | None = None,
    output_parquet_paths: Sequence[pathlib.Path] | None = None,
) -> None:
    """
    Embed a knowledge base corpus by processing text data and extracting mentions.

    For ``len(metadata.sources) > 1``, pass ``output_parquet_paths`` with one path per
    source (same order as ``metadata.sources``). For a single source, pass
    ``output_parquet_path``.
    """
    k = len(metadata.sources)
    if k == 1:
        if output_parquet_path is None:
            raise ValueError(
                "output_parquet_path is required for a single embedding source"
            )
        ope = expand_config_path(output_parquet_path)
        if ope is None:
            raise ValueError("output_parquet_path resolved to None")
        paths: list[pathlib.Path] = [pathlib.Path(ope)]
    else:
        if output_parquet_paths is None:
            raise ValueError(
                "output_parquet_paths is required when metadata has multiple sources "
                f"({k}); provide one output path per source in metadata order."
            )
        paths = []
        for p in output_parquet_paths:
            ep = expand_config_path(p)
            if ep is None:
                raise ValueError("output_parquet_paths must not contain None")
            paths.append(pathlib.Path(ep))
        if len(paths) != k:
            raise ValueError(
                f"output_parquet_paths must have length {k} (metadata.sources), got {len(paths)}"
            )

    use_gpu = training.use_gpu
    if use_gpu:
        if torch.cuda.is_available():
            logger.info("Using GPU in upcoming processes")
        else:
            logger.warning("CUDA is not available. Running on CPU instead")
            use_gpu = False

    for spec, out_path in zip(metadata.sources, paths):
        _embed_corpus_single_source(
            source=spec,
            training=training,
            output_parquet_path=out_path,
            use_gpu=use_gpu,
        )

    logger.info("All %s embedding source(s) written.", k)
