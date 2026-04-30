import click
import pathlib
import logging

from pelinker.config import EmbeddingModelMetadata, EmbeddingTrainingConfig
from pelinker.embedder import embed_kb_corpus

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-type",
    type=click.STRING,
    default="pubmedbert",
    help="Backbone model type for token embeddings (same vocabulary as pelinker.cli.fit).",
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
    "--kb-csv-path",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("data/derived/properties.synthesis.2.csv"),
    help="Path to csv with `label`, `entity_id` columns.",
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
    "--input-buffer-rows",
    "input_buffer_rows",
    type=click.INT,
    default=1000,
    help="Rows per pandas read pass over the text table (I/O only; not encoder batch size).",
)
@click.option(
    "--encoder-batch-size",
    "encoder_batch_size",
    type=click.INT,
    default=200,
    help="Table rows per transformer encoder forward pass (lower if GPU runs out of memory).",
)
@click.option(
    "--nlp-model",
    type=click.STRING,
    default="en_core_web_trf",
    help="spaCy model to use for tokenization/lemmas.",
)
@click.option(
    "--max-input-buffers",
    "max_input_buffers",
    type=click.INT,
    default=None,
    help="Stop after this many text-table read passes (each up to --input-buffer-rows rows).",
)
@click.option(
    "--negatives-per-positive",
    "negatives_per_positive",
    type=click.FLOAT,
    default=0.0,
    help="Sample this many random negative mentions per positive mention.",
)
@click.option(
    "--negative-label",
    "negative_label",
    type=click.STRING,
    default="__NEGATIVE__",
    help="Entity label used for sampled negatives.",
)
@click.option(
    "--negative-seed",
    "negative_seed",
    type=click.INT,
    default=None,
    help="Random seed for deterministic negative sampling.",
)
def run(
    model_type,
    layers_spec,
    input_text_table_path,
    kb_csv_path,
    output_parquet_path,
    use_gpu,
    input_buffer_rows,
    encoder_batch_size,
    nlp_model,
    max_input_buffers,
    negatives_per_positive,
    negative_label,
    negative_seed,
):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    metadata = EmbeddingModelMetadata.from_single(model_type, layers_spec)
    training = EmbeddingTrainingConfig(
        input_text_table_path=input_text_table_path,
        kb_csv_path=kb_csv_path,
        use_gpu=use_gpu,
        input_buffer_rows=input_buffer_rows,
        encoder_batch_size=encoder_batch_size,
        nlp_model=nlp_model,
        max_input_buffers=max_input_buffers,
        negatives_per_positive=negatives_per_positive,
        negative_label=negative_label,
        negative_seed=negative_seed,
    )

    embed_kb_corpus(
        metadata=metadata,
        training=training,
        output_parquet_path=output_parquet_path,
    )


if __name__ == "__main__":
    run()
