"""Draw — and grow — the stratified gold-annotation sample from a held-out abstract table.

Input: a TSV(.gz) with columns ``ids`` (a stringified record carrying doi / mag /
openalex / pmcid / pmid), ``publication_year``, ``summary``. By default only
**pmid-bearing** documents are eligible (``--require-pmid``): the gold set is cited in a
biomedical paper and a PubMed id is the identifier a reader resolves without help.

**1. Eligibility** (:mod:`pelinker.eval.sample_quality`). A document is drawn only if it
is a plausible English research abstract — publication year, length and sentence bounds,
English function-word share — not an exact duplicate, and absent from the fit corpus
(``--exclude-corpus``, matched on ``mag`` and on text hash). Every dropped row is counted
by reason in the sampling report.

**2. Mentions** (:class:`pelinker.text.mentions.MentionDetector`). Each eligible abstract
is parsed and its KB mentions detected with the *same* definition stage (A) of the fit
uses: verb-predicate labels from the dependency parse, one label per verb, the rest on
lemma windows, one label per site. A raw lemma-window count would mostly measure abstract
length, because nouns ("control", "increase") match labels too.

**3. Strata** = publication-year tercile × verb-mention density (``none`` / ``low`` /
``high``) × ``tail`` — whether the abstract holds a detected label outside the pool's
``--tail-top-k`` most frequent ones. Tail strata are over-allocated by ``--tail-weight``
so rare relations reach the gold set sooner; the draw stays random *within* each stratum
and every document records its inclusion probability ``p_incl``, so rates can be
reweighted to the pool. Abstracts with no detected mention are drawn as
``--none-fraction`` negative controls (they measure false positives, not "text without
relations").

**4. Growth.** ``--extend`` appends a new batch to an existing sample: already-drawn
documents are excluded, ``doc_id`` values stay stable (row index of the input table),
roles are assigned per batch in the ``--n-primary`` / ``--n-double`` / ``--n-reserve``
proportions, and ``sample_texts.jsonl`` is appended, never rewritten. Grow until the
coverage report (``audit_sample.py``) stops gaining labels.

Outputs, under ``--output-dir``:

- ``sample_manifest.csv`` — one row per drawn abstract: ``doc_id``, ``doc_uid``, every
  identifier, ``publication_year``, ``n_chars``, ``n_mentions``, ``n_verb_mentions``,
  ``labels_detected`` (``|``-joined), ``stratum``, ``p_incl``, ``batch``, ``role``,
  ``text_sha1``.
- ``sample_texts.jsonl`` — ``{"doc_id", "doc_uid", "ids", "text"}`` per line.
- ``sampling_report.batch<k>.json`` — gate counts by reason, training overlap, pool and
  stratum sizes, allocation.
- ``pool_mentions.csv.gz`` — detected mentions of the whole eligible pool, reused by
  ``--extend`` while the KB and the gate are unchanged.

Usage:

    uv run python run/eval/sample_gold.py \
        --input-text-table-path <corpus>.tsv.gz \
        --exclude-corpus <fit corpus>.tsv.gz \
        --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
        --output-dir <workdir>/gold

    # later: add a batch of 120 (80 / 20 / 20)
    uv run python run/eval/sample_gold.py ... --output-dir <workdir>/gold --extend \
        --n-primary 80 --n-double 20 --n-reserve 20
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import spacy
from pelinker.core.paths import ExpandedPath
from pelinker.eval import sample_quality as sq
from pelinker.text.mentions import DETECTOR_VERSION, MentionDetector
from pelinker.text.tokenize import tokens_from_doc

logger = logging.getLogger(__name__)

_PMID_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "doi": re.compile(r"doi=(?:'([^']*)'|\"([^\"]*)\"|None)"),
    "mag": re.compile(r"mag=(?:'([^']*)'|\"([^\"]*)\"|(\d+)|None)"),
    "openalex": re.compile(r"openalex=(?:'([^']*)'|\"([^\"]*)\"|None)"),
    "pmcid": re.compile(r"pmcid=(?:'([^']*)'|\"([^\"]*)\"|None)"),
    "pmid": re.compile(r"pmid=(?:'([^']*)'|\"([^\"]*)\"|None)"),
}

ID_FIELDS: tuple[str, ...] = ("pmid", "doi", "openalex", "mag", "pmcid")
"""Identifier precedence for ``doc_uid``.

``pmid`` leads because the gold set is drawn from pmid-bearing documents only (see
``--require-pmid``): it is the identifier biomedical readers resolve without help. The
rest are recorded too — ``mag`` in particular is what the fit corpus is keyed on, so it
is what a train/test overlap check compares against.
"""


def extract_identifiers(ids_field: object) -> dict[str, str | None]:
    """Parse every identifier out of the stringified ``Row(...)`` record.

    The corpus stores one field holding ``doi``/``mag``/``openalex``/``pmcid``/``pmid``,
    any of which may be ``None``. A bare pmid URL is reduced to its numeric id so the
    value is comparable with the pmid form used elsewhere.
    """
    text = str(ids_field)
    out: dict[str, str | None] = {}
    for name, pattern in _ID_PATTERNS.items():
        match = pattern.search(text)
        value = None
        if match is not None:
            value = next((g for g in match.groups() if g), None)
        if value is not None:
            value = value.strip() or None
        if name == "pmid" and value is not None:
            numeric = _PMID_RE.search(value)
            value = numeric.group(1) if numeric else value
        out[name] = value
    return out


def document_uid(identifiers: dict[str, str | None]) -> str | None:
    """First available identifier in :data:`ID_FIELDS` precedence, or ``None``."""
    for field in ID_FIELDS:
        value = identifiers.get(field)
        if value:
            return value
    return None


DENSITY_NONE = "none"
MANIFEST_COLUMNS = [
    "doc_id",
    "doc_uid",
    *ID_FIELDS,
    "publication_year",
    "n_chars",
    "n_mentions",
    "n_verb_mentions",
    "labels_detected",
    "stratum",
    "p_incl",
    "batch",
    "role",
    "text_sha1",
]


def kb_fingerprint(
    kb_csv_path: str | Path, rules: sq.Eligibility, require_pmid: bool
) -> str:
    """What the pool cache depends on: KB bytes, gate, pmid rule, detector version."""
    h = hashlib.sha1(Path(kb_csv_path).read_bytes())
    h.update(DETECTOR_VERSION.encode())
    h.update(json.dumps(dataclasses.asdict(rules), sort_keys=True).encode())
    h.update(str(require_pmid).encode())
    return h.hexdigest()


def detect_pool_mentions(
    texts: list[str], detector: MentionDetector, nlp, *, batch_size: int = 64
) -> pd.DataFrame:
    """``n_mentions``, ``n_verb_mentions``, ``labels_detected`` per text."""
    rows = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        mentions = detector.detect(tokens_from_doc(doc))
        rows.append(
            {
                "n_mentions": len(mentions),
                "n_verb_mentions": sum(m.is_verbal for m in mentions),
                "labels_detected": "|".join(sorted({m.label for m in mentions})),
            }
        )
    return pd.DataFrame(rows)


def split_labels(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return value.split("|")


def tail_labels(labels_detected: pd.Series, top_k: int) -> frozenset[str]:
    """Labels outside the ``top_k`` most frequent (by number of abstracts) in the pool."""
    freq: dict[str, int] = {}
    for value in labels_detected:
        for label in split_labels(value):
            freq[label] = freq.get(label, 0) + 1
    ranked = sorted(freq, key=lambda k: (-freq[k], k))
    return frozenset(ranked[top_k:])


def assign_strata(df: pd.DataFrame, tail: frozenset[str]) -> pd.Series:
    """``y<tercile>-<none|low|high>-<tail|common>``; negatives carry no tail part."""
    year_bin = pd.qcut(
        df["publication_year"].rank(method="first"), q=3, labels=["y0", "y1", "y2"]
    ).astype(str)
    positive = df["n_verb_mentions"] > 0
    median_pos = df.loc[positive, "n_verb_mentions"].median() if positive.any() else 0
    density = pd.Series(DENSITY_NONE, index=df.index, dtype=object)
    density[positive & (df["n_verb_mentions"] <= median_pos)] = "low"
    density[positive & (df["n_verb_mentions"] > median_pos)] = "high"
    has_tail = df["labels_detected"].map(
        lambda v: any(label in tail for label in split_labels(v))
    )
    tail_part = np.where(has_tail, "tail", "common")
    stratum = year_bin + "-" + density
    return stratum.where(density == DENSITY_NONE, stratum + "-" + tail_part)


def allocate(sizes: pd.Series, n: int, weights: pd.Series) -> pd.Series:
    """Largest-remainder allocation of ``n`` over strata ∝ size × weight, capped at size."""
    alloc = pd.Series(0, index=sizes.index, dtype=int)
    remaining = n
    active = sizes[sizes > 0].index.tolist()
    while remaining > 0 and active:
        mass = sizes[active] * weights[active]
        exact = mass / mass.sum() * remaining
        take = np.minimum(exact.astype(int), sizes[active] - alloc[active])
        rem = (exact - exact.astype(int)).sort_values(ascending=False)
        for name in rem.index:
            if take.sum() >= remaining:
                break
            if alloc[name] + take[name] < sizes[name]:
                take[name] += 1
        if take.sum() == 0:
            break
        alloc[active] += take
        remaining -= int(take.sum())
        active = [s for s in active if alloc[s] < sizes[s]]
    return alloc


def draw_batch(
    pool: pd.DataFrame,
    *,
    n_total: int,
    none_fraction: float,
    tail_weight: float,
    seed: int,
) -> pd.DataFrame:
    """Random draw within strata; returns the drawn rows with ``p_incl`` set."""
    is_none = pool["n_verb_mentions"] == 0
    none_pool, pos_pool = pool[is_none], pool[~is_none]
    n_none = min(int(round(n_total * none_fraction)), len(none_pool))
    n_pos = min(n_total - n_none, len(pos_pool))

    sizes = pos_pool.groupby("stratum").size()
    weights = pd.Series(
        [tail_weight if name.endswith("-tail") else 1.0 for name in sizes.index],
        index=sizes.index,
    )
    alloc = allocate(sizes, n_pos, weights)

    parts = []
    none_sizes = none_pool.groupby("stratum").size()
    none_alloc = allocate(none_sizes, n_none, pd.Series(1.0, index=none_sizes.index))
    for sizes_, alloc_, frame in (
        (none_sizes, none_alloc, none_pool),
        (sizes, alloc, pos_pool),
    ):
        for name, group in frame.groupby("stratum"):
            k = int(alloc_.get(name, 0))
            if k <= 0:
                continue
            picked = group.sample(n=k, random_state=seed).copy()
            picked["p_incl"] = k / int(sizes_[name])
            parts.append(picked)
    if not parts:
        return pool.iloc[0:0].assign(p_incl=pd.Series(dtype=float))
    drawn = pd.concat(parts)
    return drawn.sample(frac=1.0, random_state=seed)  # shuffle before role assignment


def assign_roles(n: int, n_primary: int, n_double: int, n_reserve: int) -> list[str]:
    """Roles for ``n`` drawn rows in the requested proportions (largest remainder)."""
    total = n_primary + n_double + n_reserve
    if total <= 0:
        raise click.ClickException("role counts must sum to a positive number")
    quotas = {
        "primary": n_primary / total * n,
        "double": n_double / total * n,
        "reserve": n_reserve / total * n,
    }
    counts = {k: int(v) for k, v in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - counts[k], reverse=True)[
        : n - sum(counts.values())
    ]:
        counts[k] += 1
    return (
        ["primary"] * counts["primary"]
        + ["double"] * counts["double"]
        + ["reserve"] * counts["reserve"]
    )


def load_existing(out: Path) -> pd.DataFrame | None:
    manifest = out / "sample_manifest.csv"
    if not manifest.exists():
        return None
    return pd.read_csv(manifest, dtype={f: str for f in ID_FIELDS})


@click.command()
@click.option("--input-text-table-path", required=True, type=ExpandedPath(exists=True))
@click.option("--kb-csv-path", required=True, type=ExpandedPath(exists=True))
@click.option("--output-dir", required=True, type=ExpandedPath())
@click.option(
    "--exclude-corpus",
    default=None,
    type=ExpandedPath(exists=True),
    help="Fit corpus TSV (id column, text column); its documents are never drawn.",
)
@click.option("--n-primary", default=200, show_default=True)
@click.option("--n-double", default=50, show_default=True)
@click.option("--n-reserve", default=50, show_default=True)
@click.option(
    "--none-fraction",
    default=0.1,
    show_default=True,
    help="Share of the batch drawn from abstracts with no detected mention.",
)
@click.option("--tail-top-k", default=20, show_default=True)
@click.option(
    "--tail-weight",
    default=2.0,
    show_default=True,
    help="Allocation weight of strata holding a rare (tail) label; 1.0 = proportional.",
)
@click.option("--min-year", default=sq.Eligibility.min_year, show_default=True)
@click.option("--min-chars", default=sq.Eligibility.min_chars, show_default=True)
@click.option("--max-chars", default=sq.Eligibility.max_chars, show_default=True)
@click.option(
    "--min-sentences", default=sq.Eligibility.min_sentences, show_default=True
)
@click.option(
    "--min-english-share", default=sq.Eligibility.min_english_share, show_default=True
)
@click.option(
    "--require-pmid/--no-require-pmid",
    default=True,
    show_default=True,
    help=(
        "Draw only from documents carrying a PubMed id. On by default: the gold set is "
        "cited in a biomedical paper, and a pmid is the identifier a reader resolves "
        "without help."
    ),
)
@click.option(
    "--extend",
    is_flag=True,
    help="Append a new batch to the sample already in --output-dir.",
)
@click.option("--seed", default=13, show_default=True)
@click.option("--nlp-model", default="en_core_web_lg", show_default=True)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Only consider the first N corpus rows (smoke tests).",
)
def main(
    input_text_table_path: str,
    kb_csv_path: str,
    output_dir: str,
    exclude_corpus: str | None,
    n_primary: int,
    n_double: int,
    n_reserve: int,
    none_fraction: float,
    tail_top_k: int,
    tail_weight: float,
    min_year: int,
    min_chars: int,
    max_chars: int,
    min_sentences: int,
    min_english_share: float,
    require_pmid: bool,
    extend: bool,
    seed: int,
    nlp_model: str,
    limit: int | None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(output_dir)
    existing = load_existing(out)
    if existing is not None and not extend:
        raise click.ClickException(
            f"{out} already holds a sample; pass --extend to add a batch, or choose a "
            "new --output-dir"
        )
    if existing is None and extend:
        raise click.ClickException(f"--extend: no sample_manifest.csv in {out}")
    batch = 0 if existing is None else int(existing["batch"].max()) + 1
    batch_seed = seed + batch

    rules = sq.Eligibility(
        min_year=min_year,
        min_chars=min_chars,
        max_chars=max_chars,
        min_sentences=min_sentences,
        min_english_share=min_english_share,
    )
    report: dict = {
        "batch": batch,
        "seed": batch_seed,
        "eligibility": dataclasses.asdict(rules),
    }

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
    identifiers = corpus["ids"].map(extract_identifiers)
    for field in _ID_PATTERNS:
        corpus[field] = identifiers.map(lambda ids, f=field: ids[f])
    corpus["doc_uid"] = identifiers.map(document_uid)
    corpus["summary"] = corpus["summary"].fillna("").astype(str)
    report["corpus_rows"] = len(corpus)

    if require_pmid:
        corpus = corpus.loc[corpus["pmid"].notna()].copy()
        if len(corpus) == 0:
            raise click.ClickException(
                "no corpus rows carry a pmid; pass --no-require-pmid to draw from all "
                "documents and accept weaker provenance"
            )
        report["pmid_rows"] = len(corpus)

    training = sq.load_training_keys(exclude_corpus) if exclude_corpus else None
    reasons = sq.apply_gate(corpus, rules, training=training)
    report["gate"] = sq.reason_counts(reasons)
    report["training_corpus_checked"] = training is not None
    logger.info("Eligibility gate: %s", report["gate"])
    pool = corpus.loc[reasons.isna()].copy()
    pool["text_sha1"] = pool["summary"].map(sq.text_sha1)
    pool["n_chars"] = pool["summary"].str.len()

    # Mentions for the whole eligible pool, cached while the KB and the gate are unchanged.
    fingerprint = kb_fingerprint(kb_csv_path, rules, require_pmid)
    cache_path = out / "pool_mentions.csv.gz"
    cached = pd.read_csv(cache_path) if cache_path.exists() else None
    if (
        cached is not None
        and (cached["fingerprint"] == fingerprint).all()
        and set(pool["doc_id"]) <= set(cached["doc_id"])
    ):
        detected = cached.set_index("doc_id").drop(columns="fingerprint")
        logger.info("Reusing pool mentions from %s", cache_path)
    else:
        kb = pd.read_csv(kb_csv_path)
        labels = kb["label"].dropna().astype(str).tolist()
        symmetric = (
            frozenset(kb.loc[kb["is_symmetric"].fillna(False).astype(bool), "label"])
            if "is_symmetric" in kb.columns
            else frozenset()
        )
        nlp = spacy.load(nlp_model)
        detector = MentionDetector(labels, nlp, symmetric)
        logger.info(
            "Detecting mentions in %d eligible abstracts (%d verb labels)",
            len(pool),
            len(detector.verb_labels),
        )
        detected = detect_pool_mentions(pool["summary"].tolist(), detector, nlp)
        detected.index = pool["doc_id"].to_numpy()
        detected.index.name = "doc_id"
        out.mkdir(parents=True, exist_ok=True)
        detected.assign(fingerprint=fingerprint).reset_index().to_csv(
            cache_path, index=False
        )
    pool = pool.join(detected, on="doc_id")
    pool["labels_detected"] = pool["labels_detected"].fillna("")

    tail = tail_labels(pool["labels_detected"], tail_top_k)
    pool["stratum"] = assign_strata(pool, tail)
    report["pool_size"] = len(pool)
    report["pool_strata"] = pool["stratum"].value_counts().sort_index().to_dict()
    report["tail_labels"] = len(tail)

    if existing is not None:
        taken = set(existing["doc_uid"].astype(str))
        pool = pool.loc[~pool["doc_uid"].astype(str).isin(taken)]
        report["already_drawn"] = len(taken)

    n_total = n_primary + n_double + n_reserve
    drawn = draw_batch(
        pool,
        n_total=n_total,
        none_fraction=none_fraction,
        tail_weight=tail_weight,
        seed=batch_seed,
    ).copy()
    drawn["role"] = assign_roles(len(drawn), n_primary, n_double, n_reserve)
    drawn["batch"] = batch
    report["drawn"] = len(drawn)
    report["drawn_strata"] = drawn["stratum"].value_counts().sort_index().to_dict()

    # An annotated document that cannot be cited is not evidence. Refuse rather than
    # emit a gold row whose provenance is a row index into one local file.
    unidentified = drawn.loc[drawn["doc_uid"].isna()]
    if len(unidentified):
        raise click.ClickException(
            f"{len(unidentified)} drawn documents carry no identifier "
            f"({', '.join(ID_FIELDS)}); first doc_ids: "
            f"{unidentified['doc_id'].head(5).tolist()}"
        )

    out.mkdir(parents=True, exist_ok=True)
    manifest = drawn[MANIFEST_COLUMNS]
    if existing is not None:
        manifest = pd.concat([existing[MANIFEST_COLUMNS], manifest], ignore_index=True)
    manifest.to_csv(out / "sample_manifest.csv", index=False)
    with (out / "sample_texts.jsonl").open("a", encoding="utf-8") as fh:
        for _, row in drawn.iterrows():
            record = {
                "doc_id": int(row["doc_id"]),
                "doc_uid": row["doc_uid"],
                "ids": {
                    field: (None if pd.isna(row[field]) else row[field])
                    for field in ID_FIELDS
                },
                "text": row["summary"],
            }
            fh.write(json.dumps(record))
            fh.write("\n")
    (out / f"sampling_report.batch{batch}.json").write_text(
        json.dumps(report, indent=1, default=int), encoding="utf-8"
    )
    logger.info(
        "Batch %d: wrote %d rows (%d primary / %d double / %d reserve) to %s",
        batch,
        len(drawn),
        int((drawn["role"] == "primary").sum()),
        int((drawn["role"] == "double").sum()),
        int((drawn["role"] == "reserve").sum()),
        out,
    )


if __name__ == "__main__":
    main()
