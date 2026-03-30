import pathlib
from collections.abc import Sequence

import spacy
import torch
import pandas as pd
import tqdm
import logging

from pelinker.config import (
    EmbeddingModelMetadata,
    EmbeddingSourceSpec,
    EmbeddingTrainingConfig,
)
from pelinker.ops import _detect_file_format, _detect_headers_and_columns
from pelinker.util import load_models, extract_and_embed_mentions
from pelinker.onto import WordGrouping
from pelinker.util import str2layers
from pelinker.io.parquet import ParquetWriter

logger = logging.getLogger(__name__)


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

    if training.head is not None:
        logger.info(
            "Processing only first %s chunks (head=%s)", training.head, training.head
        )
    else:
        logger.info("Processing all chunks")

    compression = (
        "gzip" if training.input_text_table_path.suffix.endswith(".gz") else None
    )
    sep = "\t" if file_format == "tsv" else ","

    reader = pd.read_csv(
        training.input_text_table_path,
        sep=sep,
        header=0 if has_header else None,
        compression=compression,
        chunksize=training.chunk_size,
    )

    output_parquet_path = pathlib.Path(output_parquet_path).expanduser()
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_writer = ParquetWriter(output_parquet_path)

    try:
        total_processed = 0

        for i, chunk in tqdm.tqdm(
            enumerate(reader), desc="Processing chunks", leave=True, position=0
        ):
            if training.head is not None and i >= training.head:
                logger.info("Reached head limit of %s chunks, stopping", training.head)
                break
            chunk_texts = list(chunk[text_col])
            pmids = [str(pmid) for pmid in chunk[pmid_col]]

            rows_data = extract_and_embed_mentions(
                entities=entities,
                data=chunk_texts,
                batch_size=training.batch_size,
                pmids=pmids,
                tokenizer=tokenizer,
                model=model,
                nlp=nlp,
                layers=layers,
                word_modes=(WordGrouping.W1, WordGrouping.W2, WordGrouping.W3),
            )

            if rows_data:
                parquet_writer.write_batch(rows_data)

            total_processed += len(chunk_texts)
            logger.info(
                "Chunk %s completed. Processed %s texts, extracted %s mentions. Total texts: %s",
                i,
                len(chunk_texts),
                len(rows_data),
                total_processed,
            )

            del rows_data

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
        paths: list[pathlib.Path] = [pathlib.Path(output_parquet_path)]
    else:
        if output_parquet_paths is None:
            raise ValueError(
                "output_parquet_paths is required when metadata has multiple sources "
                f"({k}); provide one output path per source in metadata order."
            )
        paths = [pathlib.Path(p) for p in output_parquet_paths]
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
