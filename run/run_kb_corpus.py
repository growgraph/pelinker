import click
import pathlib
import spacy
import pandas as pd
import tqdm
import logging
from typing import List

from pelinker.util import texts_to_vrep, text_to_tokens, load_models
from pelinker.onto import WordGrouping
from pelinker.model import LinkerModel
from pelinker.writer import ParquetWriter

logger = logging.getLogger(__name__)


def _wg_for_property(prop: str) -> WordGrouping | None:
    n = len(prop.split())
    if n in (1, 2, 3, 4):
        return WordGrouping(n)
    return None


def extract_and_embed_mentions(
    props: list[str],
    data: list[str],
    pmids: list[str],
    tokenizer,
    model,
    nlp,
    layers,
    batch_size,
    word_modes=(WordGrouping.W1, WordGrouping.W2, WordGrouping.W3),
) -> List[dict]:
    """
    Modified to return list of dicts instead of DataFrame for better memory management
    and consistent schema handling.
    """
    data_pmids = pmids

    data_batched = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    data_pmids_batched = [
        data_pmids[i : i + batch_size] for i in range(0, len(data), batch_size)
    ]

    # Pre-tokenize properties for lemma matching
    prop_tokens = {p: text_to_tokens(nlp=nlp, text=p) for p in props}

    rows = []
    for ibatch, text_batch in enumerate((pbar := tqdm.tqdm(data_batched))):
        report_batch = texts_to_vrep(
            text_batch,
            tokenizer=tokenizer,
            model=model,
            layers_spec=layers,
            word_modes=list(word_modes),
            nlp=nlp,
        )

        batch_pmids = data_pmids_batched[ibatch]

        # For each property, pick the matching word grouping and aggregate matches
        for p in props:
            pe = prop_tokens[p]
            wg = _wg_for_property(p)
            if wg is None:
                continue
            if wg not in report_batch.available_groupings():
                continue

            expression_container = report_batch[wg]
            for itext, (text, expr_holder) in enumerate(
                zip(report_batch.texts, expression_container.expression_data)
            ):
                expr_lemma_match = expr_holder.filter_on_lemmas(pe)
                if not expr_lemma_match:
                    continue

                offsets = [
                    report_batch.chunk_mapper.map_chunk_to_text(e.itext, e.ichunk)
                    for e, _ in expr_lemma_match
                ]

                for (e, tt), offset in zip(expr_lemma_match, offsets):
                    mention = text[offset + e.a : offset + e.b]
                    # Convert numpy array to list for consistent Parquet schema
                    embed_list = tt.numpy().tolist()
                    rows.append(
                        {
                            "pmid": batch_pmids[itext],
                            "property": p,
                            "mention": mention,
                            "embed": embed_list,  # Now a Python list, not numpy array
                        }
                    )

        pbar.set_description(f"entities added in chunk : {len(rows)}")

    return rows


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
    "--input-path",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("data/jamshid/bio_mag_2M.tsv.gz"),
    help="Input TSV.GZ with two columns: pmid, text.",
)
@click.option(
    "--props-path",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("data/test/uni_props.txt"),
    help="Path to newline-separated list of properties/patterns.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("data/jamshid/bio_2M_res.parquet"),
    help="Output Parquet file to append to.",
)
@click.option(
    "--chunk-size",
    type=click.INT,
    default=2000,
    help="Chunk size for streaming input.",
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=40,
    help="Embedding batch size inside a chunk.",
)
@click.option(
    "--nlp-model",
    type=click.STRING,
    default="en_core_web_sm",
    help="spaCy model to use for tokenization/lemmas.",
)
def run(
    model_type,
    layers_spec,
    input_path,
    props_path,
    output_path,
    chunk_size,
    batch_size,
    nlp_model,
):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load models
    logger.info("Loading models...")
    nlp = spacy.load(nlp_model)
    tokenizer, model = load_models(model_type, sentence=False)
    layers = LinkerModel.str2layers(layers_spec)

    # Load properties
    with open(props_path, "r") as f:
        props = [p for p in f.read().split("\n") if p]

    logger.info(f"Loaded {len(props)} properties")

    # Set up input reader
    reader = pd.read_csv(
        input_path, sep="\t", header=None, compression="gzip", chunksize=chunk_size
    )

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Parquet writer
    parquet_writer = ParquetWriter(output_path)

    try:
        total_processed = 0
        for i, chunk in tqdm.tqdm(
            enumerate(reader), desc="Processing chunks", leave=True, position=0
        ):
            try:
                chunk_texts = list(chunk[1])
                pmids = list(chunk[0])

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

                # Clear memory
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
        logger.info(f"Processing completed. Output saved to {output_path}")


if __name__ == "__main__":
    run()
