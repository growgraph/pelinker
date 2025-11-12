import click
import pathlib
import spacy
import torch
import pandas as pd
import tqdm
import logging

from pelinker.ops import _detect_file_format, _detect_headers_and_columns
from pelinker.util import load_models, extract_and_embed_mentions
from pelinker.onto import WordGrouping
from pelinker.model import LinkerModel
from pelinker.io.parquet import ParquetWriter

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-type",
    type=click.STRING,
    default="biobert",
    help="Backbone model type for token embeddings.",
)
@click.option(
    "--layers-spec",
    type=click.STRING,
    default="1,2",
    help="String of layers; e.g., `1,2,3` corresponds to [-1,-2,-3].",
)
@click.option(
    "--input-text-table-path",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("data/test/mag_sample.tsv.gz"),
    help="Input file (TSV/CSV, optionally gzipped) with pmid and text columns. Headers are auto-detected.",
)
@click.option(
    "--properties-txt-path",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("data/test/props.txt"),
    help="Path to newline-separated list of properties/patterns.",
)
@click.option(
    "--output-parquet-path",
    type=click.Path(path_type=pathlib.Path),
    help="output Parquet file to append to",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    default=False,
    help="Enable GPU acceleration if CUDA is available",
)
@click.option(
    "--chunk-size",
    type=click.INT,
    default=1000,
    help="Chunk size for streaming input. Each chunk is split into batches and serialized.",
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=200,
    help="Embedding batch size inside a chunk.",
)
@click.option(
    "--nlp-model",
    type=click.STRING,
    default="en_core_web_trf",
    help="spaCy model to use for tokenization/lemmas.",
)
@click.option(
    "--head",
    type=click.INT,
    default=None,
    help="Number of chunks to process (skip it for all chunks).",
)
def run(
    model_type,
    layers_spec,
    input_text_table_path,
    properties_txt_path,
    output_parquet_path,
    use_gpu,
    chunk_size,
    batch_size,
    nlp_model,
    head,
):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if use_gpu:
        if torch.cuda.is_available():
            logger.info("Using GPU in upcoming processes")
        else:
            logger.warning("CUDA is not available. Running on CPU instead")
            use_gpu = False

    # Load models
    logger.info(f"Loading models: {model_type} and {nlp_model}")
    tokenizer, model = load_models(model_type, sentence=False)
    if use_gpu:
        model.to("cuda")
        spacy.require_gpu()
    nlp = spacy.load(nlp_model)
    layers = LinkerModel.str2layers(layers_spec)

    # Load properties
    with open(properties_txt_path, "r") as f:
        props = [p for p in f.read().split("\n") if p]

    logger.info(f"Loaded {len(props)} properties")
    logger.info(f"Layers are set to {layers}")

    # Detect file format and headers
    file_format = _detect_file_format(input_text_table_path)
    has_header, pmid_col, text_col = _detect_headers_and_columns(
        input_text_table_path, file_format
    )

    logger.info(f"Detected file format: {file_format.upper()}")
    logger.info(f"Headers detected: {has_header}")
    logger.info(f"Using columns: '{pmid_col}', {text_col}'")

    # Log head parameter usage
    if head is not None:
        logger.info(f"Processing only first {head} chunks (head={head})")
    else:
        logger.info("Processing all chunks")

    # Set up input reader with detected format and headers
    compression = "gzip" if input_text_table_path.suffix.endswith(".gz") else None
    sep = "\t" if file_format == "tsv" else ","

    reader = pd.read_csv(
        input_text_table_path,
        sep=sep,
        header=0 if has_header else None,
        compression=compression,
        chunksize=chunk_size,
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
            if head is not None and i >= head:
                logger.info(f"Reached head limit of {head} chunks, stopping")
                break
            try:
                # Extract texts and pmids using detected column names
                chunk_texts = list(chunk[text_col])
                pmids = [
                    str(pmid) for pmid in chunk[pmid_col]
                ]  # Convert PMIDs to strings

                # Extract mentions and embeddings
                rows_data = extract_and_embed_mentions(
                    props=props,
                    data=chunk_texts,
                    batch_size=batch_size,
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


if __name__ == "__main__":
    run()
