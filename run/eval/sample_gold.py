"""Draw the stratified gold-annotation sample from a held-out abstract table.

Input: a TSV(.gz) with columns ``ids`` (stringified record carrying a pmid URL),
``publication_year``, ``summary`` — the held-out out-of-sample corpus shape. Each
abstract gets a cheap mention-density proxy (KB label lemma-window matches, the same
lemma matching idea stage (A) uses, without any embedding), then the sample is stratified
over publication-year terciles × mention-density bins, with an explicit quota of
zero-match abstracts as negative controls so measured precision is honest.

Outputs, under ``--output-dir``:

- ``sample_manifest.csv`` — one row per selected abstract: ``doc_id``, ``pmid``,
  ``publication_year``, ``n_mentions_proxy``, ``stratum``, ``role``
  (``primary`` / ``double`` / ``reserve``), ``text_sha1``.
- ``sample_texts.jsonl`` — ``{"doc_id", "pmid", "text"}`` per line, so downstream
  annotation never re-reads the full corpus table.

Usage:

    uv run python run/eval/sample_gold.py \
        --input-text-table-path <corpus>.tsv.gz \
        --kb-csv-path data/derived/properties.synthesis.2.csv \
        --output-dir <workdir>/gold
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import click
import pandas as pd
import spacy

logger = logging.getLogger(__name__)

_PMID_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
_MAX_WINDOW = 4


def extract_pmid(ids_field: str) -> str | None:
    """Pull the numeric pmid out of the stringified ids record, if present."""
    m = _PMID_RE.search(str(ids_field))
    return m.group(1) if m else None


def build_label_lemma_index(
    labels: list[str], nlp: spacy.language.Language
) -> set[tuple[str, ...]]:
    """KB labels as lowercase lemma tuples (window lengths 1..4)."""
    index: set[tuple[str, ...]] = set()
    for doc in nlp.pipe(labels):
        lemmas = tuple(t.lemma_.lower() for t in doc if not (t.is_punct or t.is_space))
        if 0 < len(lemmas) <= _MAX_WINDOW:
            index.add(lemmas)
    return index


def count_lemma_window_matches(
    text_lemmas: list[str], index: set[tuple[str, ...]]
) -> int:
    """Sliding lemma windows of length 1..4 that hit the KB label index."""
    n = 0
    for width in range(1, _MAX_WINDOW + 1):
        for i in range(len(text_lemmas) - width + 1):
            if tuple(text_lemmas[i : i + width]) in index:
                n += 1
    return n


def assign_strata(df: pd.DataFrame) -> pd.Series:
    """Stratum label: publication-year tercile × mention-density bin (zero/low/high)."""
    year_bin = pd.qcut(
        df["publication_year"].rank(method="first"), q=3, labels=["y0", "y1", "y2"]
    )
    positive = df["n_mentions_proxy"] > 0
    median_pos = df.loc[positive, "n_mentions_proxy"].median() if positive.any() else 0
    density_bin = pd.Series("zero", index=df.index, dtype=object)
    density_bin[positive & (df["n_mentions_proxy"] <= median_pos)] = "low"
    density_bin[positive & (df["n_mentions_proxy"] > median_pos)] = "high"
    return year_bin.astype(str) + "-" + density_bin


def draw_stratified(
    df: pd.DataFrame,
    *,
    n_total: int,
    zero_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Proportional draw over strata, with the zero-density quota enforced."""
    rng_state = seed
    n_zero = int(round(n_total * zero_fraction))
    zero_pool = df[df["n_mentions_proxy"] == 0]
    pos_pool = df[df["n_mentions_proxy"] > 0]

    n_zero = min(n_zero, len(zero_pool))
    n_pos = n_total - n_zero

    zero_draw = zero_pool.sample(n=n_zero, random_state=rng_state)

    # Proportional allocation over positive strata, with largest-remainder rounding.
    strata = pos_pool.groupby("stratum", sort=True)
    sizes = strata.size()
    exact = sizes / sizes.sum() * n_pos
    alloc = exact.astype(int)
    remainder = (exact - alloc).sort_values(ascending=False)
    for name in remainder.index[: n_pos - int(alloc.sum())]:
        alloc[name] += 1

    parts = [zero_draw]
    for name, group in strata:
        k = min(int(alloc.get(name, 0)), len(group))
        if k > 0:
            parts.append(group.sample(n=k, random_state=rng_state))
    drawn = pd.concat(parts)
    # Top up from the leftover positive pool if capping ever left us short.
    if len(drawn) < n_total:
        leftover = pos_pool.loc[~pos_pool.index.isin(drawn.index)]
        top_up = leftover.sample(
            n=min(n_total - len(drawn), len(leftover)), random_state=rng_state
        )
        drawn = pd.concat([drawn, top_up])
    return drawn.sample(frac=1.0, random_state=rng_state)  # shuffle role assignment


@click.command()
@click.option("--input-text-table-path", required=True, type=click.Path(exists=True))
@click.option("--kb-csv-path", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--n-primary", default=200, show_default=True)
@click.option("--n-double", default=50, show_default=True)
@click.option("--n-reserve", default=50, show_default=True)
@click.option(
    "--zero-fraction",
    default=0.1,
    show_default=True,
    help="Fraction of the sample drawn from zero-proxy-match abstracts (negative controls).",
)
@click.option("--seed", default=13, show_default=True)
@click.option("--nlp-model", default="en_core_web_lg", show_default=True)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Only score the first N corpus rows (smoke tests).",
)
def main(
    input_text_table_path: str,
    kb_csv_path: str,
    output_dir: str,
    n_primary: int,
    n_double: int,
    n_reserve: int,
    zero_fraction: float,
    seed: int,
    nlp_model: str,
    limit: int | None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = pd.read_csv(input_text_table_path, sep="\t")
    required = {"ids", "publication_year", "summary"}
    if not required.issubset(corpus.columns):
        raise click.ClickException(
            f"corpus table must have columns {sorted(required)}, got {list(corpus.columns)}"
        )
    if limit is not None:
        corpus = corpus.head(limit)
    corpus = corpus.reset_index(drop=True)
    corpus["doc_id"] = corpus.index
    corpus["pmid"] = corpus["ids"].map(extract_pmid)
    corpus["summary"] = corpus["summary"].fillna("").astype(str)
    logger.info("Corpus: %d abstracts", len(corpus))

    kb = pd.read_csv(kb_csv_path)
    labels = kb["label"].dropna().astype(str).tolist()
    logger.info("KB: %d labels", len(labels))

    nlp = spacy.load(nlp_model, exclude=["parser", "ner"])
    label_index = build_label_lemma_index(labels, nlp)
    logger.info("Label lemma index: %d windows", len(label_index))

    counts: list[int] = []
    for doc in nlp.pipe(corpus["summary"].tolist(), batch_size=64):
        lemmas = [t.lemma_.lower() for t in doc if not (t.is_punct or t.is_space)]
        counts.append(count_lemma_window_matches(lemmas, label_index))
    corpus["n_mentions_proxy"] = counts
    logger.info(
        "Proxy mention counts: zero=%d median(pos)=%s",
        int((corpus["n_mentions_proxy"] == 0).sum()),
        corpus.loc[corpus["n_mentions_proxy"] > 0, "n_mentions_proxy"].median(),
    )

    corpus["stratum"] = assign_strata(corpus)
    n_total = n_primary + n_double + n_reserve
    drawn = draw_stratified(
        corpus, n_total=n_total, zero_fraction=zero_fraction, seed=seed
    )

    roles = (["primary"] * n_primary + ["double"] * n_double + ["reserve"] * n_reserve)[
        : len(drawn)
    ]
    drawn = drawn.copy()
    drawn["role"] = roles
    drawn["text_sha1"] = drawn["summary"].map(
        lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_cols = [
        "doc_id",
        "pmid",
        "publication_year",
        "n_mentions_proxy",
        "stratum",
        "role",
        "text_sha1",
    ]
    drawn[manifest_cols].to_csv(out / "sample_manifest.csv", index=False)
    with (out / "sample_texts.jsonl").open("w", encoding="utf-8") as fh:
        for _, row in drawn.iterrows():
            fh.write(
                json.dumps(
                    {
                        "doc_id": int(row["doc_id"]),
                        "pmid": row["pmid"],
                        "text": row["summary"],
                    }
                )
            )
            fh.write("\n")
    logger.info(
        "Wrote %d rows (%d primary / %d double / %d reserve) to %s",
        len(drawn),
        (drawn["role"] == "primary").sum(),
        (drawn["role"] == "double").sum(),
        (drawn["role"] == "reserve").sum(),
        out,
    )


if __name__ == "__main__":
    main()
