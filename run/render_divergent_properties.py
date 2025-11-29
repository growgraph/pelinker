import logging
import pathlib
from typing import Dict, Iterable

import click
import pandas as pd
import torch

from pelinker.model import LinkerModel
from pelinker.ops import _detect_file_format
from pelinker.util import encode, load_models

logger = logging.getLogger(__name__)


def _load_dataframe(table_path: pathlib.Path) -> pd.DataFrame:
    """Load a dataframe from CSV/TSV while mimicking run/embed_kb_corpus.py style."""
    table_path = table_path.expanduser()
    if not table_path.exists():
        raise FileNotFoundError(f"Input table not found at {table_path}")

    file_format = _detect_file_format(table_path)
    compression = "gzip" if table_path.suffix.endswith(".gz") else None
    sep = "\t" if file_format == "tsv" else ","

    return pd.read_csv(table_path, sep=sep, compression=compression)


def _serialize_embeddings(batch_embeddings):
    if torch.is_tensor(batch_embeddings):
        batch_embeddings = batch_embeddings.detach().cpu().numpy()
    return [
        emb.tolist() if hasattr(emb, "tolist") else list(emb)
        for emb in batch_embeddings
    ]


def embed_column_values(
    phrases: Iterable[str],
    tokenizer,
    model,
    layers,
) -> Dict[str, list[float]]:
    phrases_list = [phrase for phrase in (str(p).strip() for p in phrases) if phrase]

    if not phrases_list:
        return {}

    phrase_to_embedding: Dict[str, list[float]] = {}
    batch_embeddings = encode(phrases_list, tokenizer, model, layers)
    for phrase, emb in zip(phrases_list, batch_embeddings):
        phrase_to_embedding[phrase] = emb

    return phrase_to_embedding


@click.command()
@click.option(
    "--input-table-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Path to the dataframe to load (CSV/TSV, optionally gzipped).",
)
@click.option(
    "--column-name",
    type=click.STRING,
    required=True,
    help="Column containing the phrases to embed.",
)
@click.option(
    "--model-type",
    type=click.STRING,
    default="biobert",
    show_default=True,
    help="Backbone model identifier passed to pelinker.util.load_models.",
)
@click.option(
    "--layers-spec",
    type=click.STRING,
    default="sent",
    show_default=True,
    help="Layer spec string (digits for token layers or 'sent' for SentenceTransformer).",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    default=False,
    help="Move the encoder model to CUDA if available.",
)
def run(
    input_table_path,
    column_name,
    model_type,
    layers_spec,
    use_gpu,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    layers = LinkerModel.str2layers(layers_spec)

    logger.info("Loading dataframe from %s", input_table_path)
    df = _load_dataframe(input_table_path)

    if column_name not in df.columns:
        raise click.BadParameter(
            f"Column '{column_name}' not found in dataframe columns {list(df.columns)}",
            param_hint="--column-name",
        )

    tokenizer, model = load_models(model_type)

    if use_gpu:
        if torch.cuda.is_available():
            logger.info("Moving model to CUDA")
            model = model.to("cuda")
        else:
            logger.warning("CUDA not available, falling back to CPU")

    phrase_series = df[column_name].dropna()
    if phrase_series.empty:
        logger.warning("Column '%s' is empty after dropping NA values", column_name)
        return {}

    phrase_embeddings = embed_column_values(
        phrase_series,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
    )

    logger.info(
        "Embedded %d unique phrases from column '%s'",
        len(phrase_embeddings),
        column_name,
    )
    for idx, (phrase, embedding) in enumerate(phrase_embeddings.items()):
        logger.info("Sample #%d '%s' -> dim=%d", idx + 1, phrase[:60], len(embedding))
        if idx >= 2:
            break

    return phrase_embeddings


if __name__ == "__main__":
    run()
