import pathlib
import spacy
import torch
import pandas as pd
import tqdm
import logging

from pelinker.config import EmbeddingModelMetadata, EmbeddingTrainingConfig
from pelinker.ops import _detect_file_format, _detect_headers_and_columns
from pelinker.util import load_models, extract_and_embed_mentions
from pelinker.onto import WordGrouping
from pelinker.util import str2layers
from pelinker.io.parquet import ParquetWriter

logger = logging.getLogger(__name__)


def embed_kb_corpus(
    *,
    metadata: EmbeddingModelMetadata,
    training: EmbeddingTrainingConfig,
    output_parquet_path: pathlib.Path,
) -> None:
    """
    Embed a knowledge base corpus by processing text data and extracting mentions.

    Args:
        metadata: Which model(s) and layer specs define the embedding (first source is used until
            multi-source fusion is implemented).
        training: Corpus paths and embedding runtime (chunk/batch size, GPU, spaCy model, etc.).
        output_parquet_path: Output Parquet file to append to.
    """
    if len(metadata.sources) > 1:
        logger.warning(
            "Multiple embedding sources in metadata; only the first source is used "
            "(concatenation / fusion not implemented yet)."
        )
    primary = metadata.sources[0]

    use_gpu = training.use_gpu
    output_parquet_path = pathlib.Path(output_parquet_path)

    if use_gpu:
        if torch.cuda.is_available():
            logger.info("Using GPU in upcoming processes")
        else:
            logger.warning("CUDA is not available. Running on CPU instead")
            use_gpu = False

    # Load models
    logger.info(f"Loading models: {primary.model_type} and {training.nlp_model}")
    tokenizer, model = load_models(primary.model_type, sentence=False)
    if use_gpu:
        model.to("cuda")
        spacy.require_gpu()

    nlp = spacy.load(training.nlp_model)
    layers = str2layers(primary.layers_spec)

    df_props = pd.read_csv(training.kb_csv_path)
    entities = df_props["label"].tolist()

    logger.info(f"Loaded {df_props.shape[0]} properties")
    logger.info(f"Layers are set to {layers}")

    # Detect file format and headers
    file_format = _detect_file_format(training.input_text_table_path)
    has_header, pmid_col, text_col = _detect_headers_and_columns(
        training.input_text_table_path, file_format
    )

    logger.info(f"Detected file format: {file_format.upper()}")
    logger.info(f"Headers detected: {has_header}")
    logger.info(f"Using columns: '{pmid_col}', {text_col}'")

    # Log head parameter usage
    if training.head is not None:
        logger.info(
            f"Processing only first {training.head} chunks (head={training.head})"
        )
    else:
        logger.info("Processing all chunks")

    # Set up input reader with detected format and headers
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

    # Create output directory
    output_parquet_path = output_parquet_path.expanduser()
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Parquet writer
    parquet_writer = ParquetWriter(output_parquet_path)

    try:
        total_processed = 0

        for i, chunk in tqdm.tqdm(
            enumerate(reader), desc="Processing chunks", leave=True, position=0
        ):
            # Apply head limit if specified
            if training.head is not None and i >= training.head:
                logger.info(f"Reached head limit of {training.head} chunks, stopping")
                break
            try:
                # Extract texts and pmids using detected column names
                chunk_texts = list(chunk[text_col])
                pmids = [
                    str(pmid) for pmid in chunk[pmid_col]
                ]  # Convert PMIDs to strings

                # Extract mentions and embeddings
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

                # Write to Parquet
                if rows_data:
                    parquet_writer.write_batch(rows_data)

                total_processed += len(chunk_texts)
                logger.info(
                    f"Chunk {i} completed. Processed {len(chunk_texts)} texts, "
                    f"extracted {len(rows_data)} mentions. "
                    f"Total texts processed: {total_processed}"
                )

                del rows_data

            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                raise

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        # Always close the writer
        parquet_writer.close()
        logger.info(f"Processing completed. Output saved to {output_parquet_path}")
