# Gold evaluation

The clustering metrics (DBCV, ARI) say how coherent the fitted manifold is against the
pipeline's own weak labels. They cannot say whether a linked mention is *right*. This
guide covers the other track: a human-verified gold set of annotated predicate mentions,
and reference baselines scored against it through one harness.

Everything here lives in `run/eval/` (drivers) and `pelinker/eval/` (library code). The
linker itself never calls an LLM; only this evaluation path does.

## Install and credentials

```bash
uv sync --extra dev --extra eval
```

The `eval` extra installs every provider SDK behind `pelinker.eval.llm`. Each provider
reads its own credential from the environment:

| Provider | Credential |
|----------|------------|
| `gemini` (default) | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |

Responses are cached on disk, keyed on provider, model and both prompts. Re-runs, parser
fixes and re-scoring are free, and a finished annotation batch is reproducible from the
cache alone.

## The id space: canonical, with direction carried separately

The property KB holds both members of many converse pairs — "regulates" and "regulated
by", "has part" and "part of" — as separate entries. If an annotator could use either, a
passive mention would have two equally valid encodings, and the gold would record
disagreements that are really just two names for one relation.

So one member of each pair is elected **canonical**, and orientation moves into a separate
`direction` field (`forward`, `inverse`, `symmetric`, `na`). Annotation offers canonical
labels only. Scoring folds every system's answer the same way, so a baseline that answers
"regulated by" is scored on the relation it picked rather than on which name it used.

**Symmetric relations** ("interacts with", "overlaps", "connected to") have no converse:
read either way they are the same relation, so their only valid direction is `symmetric`.
They are flagged `is_symmetric`, marked `symmetric` in the annotation prompt, and an
annotator's oriented answer on one is set to `symmetric` and flagged `direction_coerced`.

This is why the whole pipeline consumes the **pairs** KB, and why the tools refuse a KB
that predates the derivation instead of quietly falling back:

```
properties.synthesis.2.csv          # merged KB
  → properties.synthesis.2.inverse.csv   # + owl:inverseOf, the derivation's INPUT
    → properties.synthesis.2.pairs.csv   # + is_canonical / canonical_entity_id / is_symmetric
```

## Step 0 — derive converse pairs, and review them

The full lineage of the KB files — sources, hand-curated files, the generated pairs KB and
what each edit invalidates — is diagrammed in `data/README.md` in the repository.

```bash
uv run python run/preprocessing/derive_inverse_pairs.py \
  --kb-csv-path data/derived/properties.synthesis.2.inverse.csv \
  --output-path data/derived/properties.synthesis.2.pairs.csv \
  --review-csv data/curated/inverse_pairs.review.csv \
  --ro-csv-path data/derived/properties.ro.csv \
  --symmetric-csv data/curated/symmetric.csv
```

`is_symmetric` is joined from the RO extraction (`extract_properties_ro.py`), which counts
a property as symmetric when RO declares `owl:SymmetricProperty` or defines it by the chain
`inverse(P) ∘ P`, plus the curated `data/curated/symmetric.csv` for entries RO does not
cover. The surface-form rules never pair a symmetric entry.

Pairs come from four rules, recorded per row in `inverse_source`. Only `ro` (a declared
`owl:inverseOf`) is a fact about the ontology; `has_of`, `passive_by` and `passive_prep`
are surface-form proposals, marked `needs_review`.

Unreviewed proposals are written to `<output>.pending_review.csv`. A curator fills in
`verdict`:

- **accept** — confirmed converses; the pair stands and `needs_review` clears.
- **reject** — not converses; the link is removed and each member becomes its own
  canonical form again.

Pass the filled sheet back as `--review-csv` and re-run. Review before annotating, because
a wrong pair merges two distinct relations onto one canonical id and nothing downstream
can see it: prompts, gold and scores all agree with the mistake. `annotate_llm.py` warns
while any pair is still unreviewed.

## Step 1 — sample the documents

```bash
uv run python run/eval/sample_gold.py \
  --input-text-table-path <held-out>.tsv.gz \
  --exclude-corpus <fit corpus>.tsv.gz \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
  --output-dir <workdir>/gold
```

The sampler works in three stages:

1. **Eligibility gate.** Only plausible English research abstracts are drawn:
   - publication year from 1990 (`--min-year`);
   - length within `--min-chars` / `--max-chars`, and at least `--min-sentences`
     sentences, which excludes citation strings, news items and full texts;
   - an English function-word share (`--min-english-share`);
   - no exact duplicates, and no document from the fit corpus (matched on `mag` and on
     text hash).

   Each dropped row is counted by reason in `sampling_report.batch<k>.json`.
2. **Mentions.** Each eligible abstract is parsed, and its KB mentions are detected with
   the same definition the fit uses. Verb labels are matched from the dependency parse,
   one per verb; other labels are matched lexically; each site keeps one label. A raw
   lemma-window count mostly measures abstract length, because nouns ("control",
   "increase") match labels too.
3. **Strata.** Documents are grouped by year tercile × verb-mention density (`none` /
   `low` / `high`) × `tail`, which marks an abstract holding a label outside the pool's
   `--tail-top-k` most frequent.
   - Tail strata are over-allocated by `--tail-weight`, so rare relations reach the gold
     set sooner.
   - The draw is random within each stratum, and each document records its inclusion
     probability `p_incl`, so rates can be reweighted to the pool.
   - About `--none-fraction` of documents are negative controls with no detected mention.

**Grow the sample in batches.** Re-running with `--extend` appends a batch:
- documents already drawn are excluded;
- `doc_id`s stay stable;
- roles are assigned per batch in the `--n-primary` / `--n-double` / `--n-reserve`
  proportions, so the lockbox holds as the set grows;
- the pool's detected mentions are cached in `pool_mentions.csv.gz` while the KB and gate
  are unchanged.

After each batch, and after annotating it, run the audit:

```bash
uv run python run/eval/audit_sample.py --gold-dir <workdir>/gold \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv
```

It re-checks eligibility, reports how strongly density still tracks length, lists
documents and negative controls with no gold hit, and measures label concentration. It
also reports coverage: canonical labels reaching `--min-mentions` gold hits, out of those
the pool can supply, plus a cumulative curve per batch. Stop adding batches when that
curve levels off.

Every document is assigned a role:

| Role | Default size | Purpose |
|------|--------------|---------|
| `primary` | `--n-primary` | the reported gold slice, annotated once |
| `double` | `--n-double` | annotated twice, for the agreement (κ) check |
| `reserve` | `--n-reserve` | held back, not annotated |

A drawn document without any identifier is an error rather than a row: a gold span whose
document cannot be cited is not usable evidence.

## Step 2 — pre-annotate with an LLM

```bash
uv run python run/eval/annotate_llm.py \
  --sample-dir <workdir>/gold \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
  --output-path <workdir>/gold/gold.llm-a.json \
  --roles primary,double
```

Three contracts make the output auditable:

- **The model answers with labels, never ids.** Ids mean nothing to a language model, and
  transcribing them fails silently; an unmatched label is a logged rejection instead.
- **The model never emits character offsets.** It quotes the surface verbatim and gives a
  1-based occurrence index; the driver resolves offsets itself and rejects what does not
  resolve. Rejections land in `<output>.report.json` — they are never dropped silently.
- **A converse label is recovered, not discarded.** The prompt offers canonical labels,
  but the decoder accepts the whole KB: an answer of "expressed in" maps onto the
  canonical id with `direction` flipped and is flagged. Dropping those would remove
  inverse-direction mentions specifically — the population the directionality work needs.

**Scope: verb mentions.** A hit on a verb-predicate label whose span contains no verb
(for example "association" or "inhibition") is rejected as `non_verbal`. The linker is
trained on verb mentions only. Labels that are not verb predicates ("has part") keep
their noun surfaces. The check runs after the response cache, so `--no-verbal-only`
re-decodes without any LLM calls.

**Paraphrases are kept and tagged.** Each hit carries `surface_anchored`:
- `true` when the surface uses the label's own wording ("increased" for `increases`);
- `false` for a paraphrase ("leads to" for `causes`).

Paraphrases are valid gold. The linker's training is anchored on label lemmas, so
paraphrases are scored as a separate slice that measures generalisation. The tag is
computed after the cache.

Use `--dry-run` to see request sizes without spending anything.

**Truncated answers are never cached.** A response the provider cut off at `--max-tokens`
is reported as `truncated` in `<output>.report.json`, and the document is left out of
the output. It is not written as an empty document, because that would read as "the
annotator found nothing". Nothing is cached for it, so re-running with a larger limit
retries only those documents. A mention the model repeats after every occurrence is
already claimed is reported as `duplicate_occurrence`. `surface_not_found` means the
quoted text is not in the abstract at all.

## Step 3 — a second annotator for agreement and recall

Run it again with a different **model family**, over the same roles:

```bash
uv run python run/eval/annotate_llm.py \
  --sample-dir <workdir>/gold \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
  --output-path <workdir>/gold/gold.llm-b.json \
  --provider openai --model <model-id> --reasoning-effort low --max-tokens 16000 \
  --roles primary,double
```

The second annotator does two jobs. It gives κ, and it raises recall: verified gold only
ever contains spans that one of the annotators proposed. Running it over `double` alone
would leave the reporting documents bounded by a single annotator's recall.

A second checkpoint of the same family mostly measures itself; κ between families is the
informative number.

For a reasoning model, pin `--reasoning-effort` and give `--max-tokens` headroom, because
the hidden reasoning counts against the same limit as the answer. The effort is part of
the cache key and of the default annotator tag (`<model>@<effort>`), so each setting is
its own run. Try `--limit 3` first and read `usage` in the cached responses under
`.llm_cache/` to size the limit.

## Step 4 — agreement, review sheet, adjudication

```bash
uv run python run/eval/gold_review_sheet.py agreement \
  --gold <workdir>/gold/gold.llm-a.json --other <workdir>/gold/gold.llm-b.json \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
  --equivalences data/curated/equivalences.csv

uv run python run/eval/gold_review_sheet.py export \
  --gold <workdir>/gold/gold.llm-a.json --other <workdir>/gold/gold.llm-b.json \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
  --equivalences data/curated/equivalences.csv \
  --output <workdir>/gold/review.tsv
```

κ is reported on entity id and on direction, over the documents **both** annotators
covered — `n_docs_shared` says how many. A span only one annotator marked enters the
entity-id κ as an explicit `∅` on the other side, in both directions, so missing a mention
and inventing one are penalized alike.

With `--kb-csv-path`, κ is also reported **folded** (`kappa_entity_id_folded`). Ids are
compared in the reference inventory's tiers 0 and 1: converse members fold onto their
canonical member, and `--equivalences` adds the identities the KB implies without
declaring (`data/curated/equivalences.csv`). Two annotators who name one relation two ways
then agree. The raw κ stays beside it.

The review TSV has one row per candidate span with a marked context window. Rows carry an
`origin`: `gold` from the first annotator, `other` for spans only the second proposed —
without those, verified gold could never exceed the first annotator's recall. On documents
the second annotator never saw, the comparison columns read `not-covered` rather than
looking like a disagreement. Each row shows the other annotator's `other_entity_id`,
`other_label` and `other_direction`. `agrees` compares the two ids in the folded space.
`agrees_direction` is filled only where the relation agrees (a symmetric relation agrees in
any direction), so an `agrees=True`, `agrees_direction=False` row is a pure direction
dispute.

Fill in `verdict` (`accept` / `reject` / `fix`) on **every** row, with `fix_entity_id` and
`fix_direction` where the verdict is `fix`; anything else is an error, so nothing is kept
by omission. Then:

```bash
uv run python run/eval/gold_review_sheet.py import \
  --gold <workdir>/gold/gold.llm-a.json --sheet <workdir>/gold/review.tsv \
  --other <workdir>/gold/gold.llm-b.json \
  --verified-by <name> --output <workdir>/gold/gold.verified.json
```

Kept hits become `source="human"` with the adjudicator as `annotator`, and record their
`verdict`. `--other` is required whenever an `origin=other` row was accepted, since the
hit and its argument spans live in that file.

Import also writes `<output>.verification.json`. It holds the accept / fix / reject
counts and the **correction rate** (fixed plus rejected, over reviewed candidates), split
into anchored and paraphrase slices. `audit_sample.py --gold-file gold.verified.json`
includes it.

**Describing the gold set.** Every candidate is reviewed, but candidates come only from
the LLM annotators. The accurate description is therefore *LLM pre-annotated, every
candidate manually verified*, not *manually annotated*: a mention both annotators missed
is not in gold. Report the annotator κ and the correction rate alongside it. A baseline
LLM linker should come from a model family that did not pre-annotate the gold, or it is
scored against its own kind.

## Step 5 — score the baselines

```bash
uv run python run/eval/run_baselines.py \
  --gold <workdir>/gold/gold.verified.json \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
  --report-dir <workdir>/eval-runs/<run-name> \
  --systems lexical,encoder \
  --model-path <models>/pelinker.pubmedbert.1
```

| System | What it does | Regimes |
|--------|--------------|---------|
| `lexical` | lemma match to KB labels — the floor | end-to-end + linking-only |
| `encoder` | sentence-encoder cosine top-1 over `label: description` | linking-only (end-to-end reuses the lexical spans) |
| `llm` | constrained choice over the KB, disk-cached | linking-only |
| `linker` | a fitted PELinker artifact, via the KB-out → KB-in bridge | end-to-end + linking-only |

Two regimes are reported separately. **End-to-end** scores span detection and linking
together. **Linking-only** gives the system the gold spans, isolating the id decision from
span proposal.

Configure the `llm` baseline with a model that did **not** annotate the gold, or the
comparison is circular. Nothing enforces this.

Results go to `baseline_results.json` / `.csv` under `--report-dir`. Keep that directory
outside the repository: measured numbers belong with the measurement writeup.

## Step 6 — corpus and gold statistics

```bash
uv run python run/eval/dataset_stats.py \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
  --gold <workdir>/gold/gold.verified.json \
  --report-dir <workdir>/eval-runs/dataset-stats
```

Any subset of corpus / KB / fit report / gold may be passed; each contributes its section
to `dataset_stats.json`.

## Reading a result

- `entity_accuracy` is **undefined**, not zero, when `n_id_comparable` is 0 — most often
  because a linker artifact emitted minted KB-out ids and no id bridge was available.
- `accuracy_overall` in the linking-only regime counts abstentions as wrong;
  `accuracy_on_predicted` does not. Report both, or the headline flatters whichever system
  abstains most.
- Direction is annotated in gold but not scored by the baselines: they assign an id and
  nothing else.
