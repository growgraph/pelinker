import click
import pathlib
import logging

from pelinker.embedder import embed_kb_corpus

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
    kb_csv_path,
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

    # Call the embedding function
    embed_kb_corpus(
        model_type=model_type,
        layers_spec=layers_spec,
        input_text_table_path=input_text_table_path,
        kb_csv_path=kb_csv_path,
        output_parquet_path=output_parquet_path,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        batch_size=batch_size,
        nlp_model=nlp_model,
        head=head,
    )


if __name__ == "__main__":
    run()
