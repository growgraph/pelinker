# Run Scripts Documentation

This directory contains scripts for preprocessing knowledge bases, embedding corpora, analyzing embedding quality, producing **OOV / manifold anomaly** figures from fit reports plus batch-linked mention dumps, and building and scoring the **human-verified gold set** (`eval/`).

For a **single end-to-end train** (corpus embedding → KB filtering/aggregation → PCA/UMAP → optimized HDBSCAN clustering → serialized linker artifact), use the packaged CLI `pelinker.cli.fit` documented in [Fitting the linker model](#fitting-the-linker-model) below.

The published **MkDocs** site mirrors this layout at a high level under [Run scripts & CLIs](https://growgraph.github.io/pelinker/user_guide/run_scripts_and_cli/) (source: `docs/user_guide/run_scripts_and_cli.md`). Package **version** is `version` in the repo root `pyproject.toml`.

## Directory Structure

```
run/
├── README.md                    # This file
├── embed_kb_corpus.py           # Embed a corpus against the KB (fit stage A, standalone)
├── smoke_server.py              # Smoke-test pelinker.cli.server HTTP routes
├── loop.embed.kb.corpus.sh      # Batch embedding (grid over model × layer)
├── loop.fit.sh                  # Batch full fit: same grid, runs pelinker-fit (A+B)
├── preprocessing/               # Property knowledge base generation
│   ├── extract_properties_go.py    # Extract from the GO-CAMs ontology
│   ├── extract_properties_ro.py    # Extract from the Relations Ontology
│   ├── merge_properties.py         # Merge properties from all sources
│   └── derive_inverse_pairs.py     # Link converse pairs, elect a canonical member
├── analysis/                    # Embedding quality, stability & OOV diagnostics
│   ├── cluster_stability.py        # Cluster assignment stability across draws
│   ├── compact_predict_study.py    # Legacy vs ParametricUMAP+MLP quality/size gates
│   ├── direction_diagnostic.py     # Where voice/direction is lost (encoder vs supervision)
│   ├── weak_label_check.py         # Acceptance check for stage-(A) weak labels
│   ├── oov_analysis.py             # Fit report + OOV mention dump → PDF figures
│   ├── replot_fit.py               # Re-render figures from an existing fit report
│   └── select_diverse_entities.py  # Select diverse entity subsets
└── eval/                        # Human-verified gold set & reference baselines
    ├── sample_gold.py              # Eligibility gate + stratified sample, grown in batches
    ├── audit_sample.py             # Sample soundness + KB coverage, after every batch
    ├── annotate_llm.py             # LLM pre-annotation of predicate mentions
    ├── gold_review_sheet.py        # Review TSV round trip + Cohen's κ
    ├── run_baselines.py            # Lexical / encoder / LLM / linker, one harness
    ├── dataset_stats.py            # Corpus, KB, fit-report and gold statistics
    └── prompts/                    # Annotation prompt templates
```

Model selection, dimension search, the scale curve and grid replotting are **console
scripts**, not files under `run/` — see [Hyperparameter search](#hyperparameter-search).

## Preprocessing Scripts

Scripts in the `preprocessing/` directory generate property knowledge base files from various ontology sources.

### `extract_properties_go.py`

Extracts property definitions from the Gene Ontology (GO) Causal Activity Models (GO-CAMs) ontology. 

- **Input**: `data/raw/GO-CAMs.ttl.gz` (Turtle format ontology file)
- **Output**: 
  - `data/derived/properties.go.csv` - Extracted properties with entity IDs, labels, and descriptions
  - `data/derived/properties.go.failed.csv` - Entities that failed to fetch from the OLS API
- **Process**: Queries the GO-CAMs ontology for object properties, then fetches detailed metadata from the EBI OLS API

### `extract_properties_ro.py`

Extracts property definitions from the Relations Ontology (RO).

- **Input**: `data/raw/ro.owl` (OWL format ontology file)
- **Output**: `data/derived/properties.ro.csv` - Extracted properties with entity IDs, labels, descriptions, declared `inverse_entity_id` and `is_symmetric`
- **Process**: Parses the RO OWL file and extracts object properties with their labels and descriptions. A property is symmetric when RO declares `owl:SymmetricProperty` or defines it by the chain `inverse(P) ∘ P`

### `merge_properties.py`

Merges properties from multiple sources (RO, GO, and custom properties) into a unified knowledge base.

- **Inputs**:
  - `data/derived/properties.ro.csv`
  - `data/raw/properties.csv` (custom properties)
  - Latest versioned synthesis file (if exists)
- **Output**: `data/derived/properties.synthesis.{version}.csv` - Merged property knowledge base
- **Process**:
  - Merges RO properties, existing PEL properties, and new custom properties
  - Filters out obsolete or deprecated properties
  - Assigns entity IDs to new properties (PEL.{number} format)
  - Removes duplicates, prioritizing entries with descriptions
  - Only creates a new version if entity IDs have changed

### `derive_inverse_pairs.py`

Links converse pairs across the whole KB and elects a canonical member for each, producing
the KB the gold pipeline consumes. Which KB files are curated, which are generated, and
what to re-run after an edit: [`data/README.md`](../data/README.md) (with a diagram).

- **Input**: `data/derived/properties.synthesis.2.inverse.csv` (the merged KB plus declared
  `owl:inverseOf`)
- **Outputs**: `data/derived/properties.synthesis.2.pairs.csv`, plus
  `<output>.pending_review.csv` listing pairs awaiting a curator verdict
- **Rules**, recorded per row in `inverse_source`: `ro` (declared `owl:inverseOf`,
  authoritative), `has_of` (`has X` ↔ `X of`), `passive_by` (`X-ed by` ↔ the active lemma)
  and `passive_prep` (the same for the remaining prepositions). Only the first is a fact
  about the ontology; the other three are proposals marked `needs_review`.
- **Symmetric relations**: `is_symmetric` is joined from `properties.ro.csv`
  (`--ro-csv-path`), plus `data/curated/symmetric.csv` (`--symmetric-csv`) for entries RO
  does not cover. A symmetric relation has no converse, so the surface-form rules never
  pair it.
- **Canonical member**: the KB carries both members of a pair, so a passive mention would
  otherwise have two equally valid encodings. One member is elected canonical
  (`is_canonical`, `canonical_entity_id`), and orientation moves into the gold's
  `direction` field.
- **Review loop**: fill `verdict` (`accept` / `reject`) in the pending sheet and pass it
  as `--review-csv`. Rejecting unlinks the pair so each member is its own canonical form
  again. Review before annotating — a wrong pair merges two distinct relations onto one
  id, and prompts, gold and scores then all agree with the mistake.

```bash
uv run python run/preprocessing/derive_inverse_pairs.py \
  --kb-csv-path data/derived/properties.synthesis.2.inverse.csv \
  --output-path data/derived/properties.synthesis.2.pairs.csv \
  --review-csv data/curated/inverse_pairs.review.csv \
  --ro-csv-path data/derived/properties.ro.csv \
  --symmetric-csv data/curated/symmetric.csv
```

## Gold evaluation (`eval/`)

The clustering metrics score the manifold against the pipeline's own weak labels; the gold
set is the independent track. Full walkthrough: **[Gold
evaluation](https://growgraph.github.io/pelinker/user_guide/evaluation/)** (source:
`docs/user_guide/evaluation.md`). Needs `uv sync --extra dev --extra eval` and a provider
credential.

| Step | Script | Produces |
|------|--------|----------|
| 0 | `preprocessing/derive_inverse_pairs.py` | the canonical pairs KB |
| 1 | `eval/sample_gold.py` (`--extend` to grow) | `sample_manifest.csv`, `sample_texts.jsonl` (roles: `primary` / `double` / `reserve`), `sampling_report.batch<k>.json` |
| 1a | `eval/audit_sample.py` | `sample_audit.json`: eligibility re-check, empty documents, label coverage per batch |
| 2 | `eval/annotate_llm.py` | `gold.llm-a.json` + a rejection report |
| 3 | `eval/annotate_llm.py` with another model family, `--roles primary,double` | `gold.llm-b.json` |
| 4 | `eval/gold_review_sheet.py` `agreement` / `export` → adjudicate → `import` | κ, `review.tsv`, `gold.verified.json` |
| 5 | `eval/run_baselines.py` | `baseline_results.{json,csv}` |
| 6 | `eval/dataset_stats.py` | `dataset_stats.json` |

Two invariants hold throughout: every step consumes the **pairs** KB (a KB without
`is_canonical` is refused rather than tolerated), and ids are compared in canonical space
on both sides, so a converse-member answer counts as the relation it names. Point
`--report-dir` outside the repository — measured numbers belong with the measurement
writeup.

## Embedding Scripts

### `embed_kb_corpus.py`

Embeds a knowledge base corpus using the same pipeline as **stage (A)** of `pelinker.cli.fit` (both call `pelinker.embed.corpus.embed_kb_corpus`).

- **Purpose**: Stream a text table, find KB property mentions, and write **mention-level** rows (with vectors) to Parquet.
- **Inputs**:
  - `--input-text-table-path`: TSV/CSV (optional gzip) with `pmid` and `text` columns; headers are auto-detected (same as fit).
  - `--kb-csv-path`: Property KB CSV with **`label`** and **`entity_id`** (property labels are the patterns; same role as **`kb_path`** in `pelinker-fit`).
  - `--output-parquet-path`: Destination Parquet (columns include **`property`**, **`embed`**, **`pmid`**, **`mention`**, etc.)—same artifact as **`embeddings_parquet`** in fit.

#### Parameter reference (`embed_kb_corpus.py`)

| Flag | Default | Meaning | Practical note |
|---|---|---|---|
| `--model-type` | `pubmedbert` | Transformer backbone used to produce token embeddings. | Keep consistent with downstream fit/eval for comparable artifacts. |
| `--layers-spec` | `1,2` | Which hidden layers to aggregate (parsed by `str2layers`; `1` means last layer, `1,2` means last two). | More layers can improve signal but increase compute cost. |
| `--input-text-table-path` | `data/test/mag_sample.tsv.gz` | Input corpus table (TSV/CSV, optional gzip) that contains `pmid` and `text`. | Header/column detection is automatic. |
| `--kb-csv-path` | `data/derived/properties.synthesis.2.csv` | KB dictionary with `label` and `entity_id` used for mention matching. | Labels drive matching; IDs are stored for linker vocabulary alignment. |
| `--output-parquet-path` | *(required)* | Output mention-level parquet path. | One row per extracted mention, including embedding vectors. |
| `--use-gpu` | `false` | Move encoder inference to CUDA (if available). | Use this for speed on large corpora. |
| `--input-buffer-rows` | `1000` | Rows per pandas chunk when reading the text table. | I/O chunking only; does not control model forward memory. |
| `--encoder-batch-size` | `200` | Number of table rows encoded per transformer forward pass. | Primary OOM control knob; lower when GPU runs out of memory. |
| `--nlp-model` | `en_core_web_lg` | spaCy pipeline used for tokenization/lemma processing around mention extraction. | Ensure the model is installed in the `uv` env. |
| `--max-input-buffers` | *(unset)* | Stop after this many read chunks (`input_buffer_rows` each, except final partial chunk). | Useful for smoke tests without scanning the full corpus. |
| `--negatives-per-positive` | `0.0` | Number of synthetic negative mentions sampled per positive mention. | `0` disables negatives; `1.0` means roughly one negative per positive. |
| `--negative-label` | `__NEGATIVE__` | Label assigned to sampled negative rows. | Keep this distinct from all real KB labels. |
| `--negative-seed` | *(unset)* | RNG seed for negative sampling. | Set for reproducible test runs and stable comparisons. |

#### Negative sampling behavior

- `--negatives-per-positive` controls **how many** negatives are added, relative to positives.
- `--negative-label` controls **what label** those synthetic negatives carry in output rows.
- `--negative-seed` controls **determinism** of which negatives are sampled.
- Negatives are intended for training/evaluation robustness; for pure extraction runs, keep `--negatives-per-positive=0`.

### `loop.embed.kb.corpus.sh` / `loop.fit.sh`

- **`loop.embed.kb.corpus.sh`**: loops over the same default **`model_type` × `layers_spec`** grid and runs **`embed_kb_corpus.py`** only (Parquet per combo). Run it with `bash`. It **refuses to overwrite**: if any target `res_<model>_<layer>.parquet` exists, it stops before embedding anything, so an earlier grid (for example the "before" arm of a weak-label change) is never replaced. Write each grid to a new `--output-parquet-path` directory. Optional flags:
  - `--models "pubmedbert scibert"` and `--layers "1 2"` narrow the grid;
  - `--max-input-buffers N` caps the input for a smoke run;
  - `--no-gpu` runs on CPU.

  It warns when `--kb-csv-path` has no `is_symmetric` column (pass the pairs KB). Check a smoke parquet with `analysis/weak_label_check.py` before launching the full grid.
- **`loop.fit.sh`**: same grid, but runs **`uv run pelinker-fit`** per combo—**stage (A)** writes `res_<model>_<layer_tag>.parquet` under **`--output-parquet-prefix`**, **stage (B)** writes **`pelinker.<model>.<layer_tag>.gz`** under **`--output-model-prefix`** (`layer_tag` is `layers_spec` with commas replaced by `_` for filenames). Requires four flags: `--input-text-table-path`, `--kb-csv-path`, `--output-parquet-prefix`, **`--output-model-prefix`**. Optional **`--layers`**: **`layers_spec`** list (default `1,2,3`). Comma separates distinct specs; use **semicolons** when one spec contains commas, e.g. `--layers 1,2,3`, `--layers 1`, or `--layers '1,2;3'` (runs `1,2` then `3`).

## Fitting the linker model

Module: **`pelinker.cli.fit`**. It runs the linker training pipeline in **two conceptual stages**:

1. **Stage (A)** — `embed_kb_corpus(...)` (same function as `run/embed_kb_corpus.py`) when **`input_text_table_path`** is set: **`kb_path`** + text table → **`embeddings_parquet`**. Verb-predicate labels are matched from the dependency parse, one label per verb mention chosen by voice (`pelinker/text/predicates.py`); other labels match lemma windows. Each row records `direction` and `surface_rule`. Pass the pairs KB (`properties.synthesis.2.pairs.csv`) so symmetric relations are never given an inverse direction.
2. **Stage (B)** — `Linker.fit(...)` on that Parquet: fusion / negative screener / PCA / UMAP / HDBSCAN at a fixed `min_cluster_size` → fitted linker. Under the default `class_view=reldir`, the catalog's composition is read between canonical relations and every cluster gets a `dominant_direction`, which `predict` and `/link` emit as `direction_predicted`. The linker is serialized via `Linker.dump` (joblib at **`{model_path}.gz`**; the `.gz` suffix is appended automatically). Choose `min_cluster_size` upstream (e.g. `pelinker-model-selection`); this CLI does not run a grid search during fit. It can, however, **read** the upstream choice: pass `selection_report=` for the search winner, or `scale_curve_path=` to extrapolate `min_cluster_size` to this fit's realized row count. In `compact` mode the fit also holds out a `pmid`-grouped slice, scores the MLP entity head against its HDBSCAN teacher, and writes the result to the fit report as `distillation_fidelity`.

**Which stages run is set by `pipeline=`, not inferred from the other options:**

| `pipeline` | Stage (A) | Stage (B) | Use when |
|------------|-----------|-----------|----------|
| `embed_only` (**default**) | yes | no | producing a mention parquet to search over |
| `fit_only` | no | yes | the parquet already exists; passing a text table here is an error |
| `both` | yes | yes | a single end-to-end train |
| `auto` | if a text table is given | yes | scripted runs where the text table may or may not be set |

The default is `embed_only`, so a command that sets `input_text_table_path`, `model_path`
and everything else still writes only the parquet unless you ask for `pipeline=both`.

There are **no implicit path fallbacks**: `model_path` and `report_path` are required for
any pipeline that fits, and the process fails rather than writing to a default location.

**How to run** (use `uv` so dependencies match `uv.lock`):

- `uv run pelinker-fit …` (console script from `pyproject.toml`)
- `uv run python -m pelinker.cli.fit …` (equivalent module invocation)

Configuration uses [Hydra](https://hydra.cc/) **override** syntax: `key=value` arguments after the command. App defaults are composed from `FitCliConfig` in `pelinker/cli/fit.py` via `pelinker/conf/fit.yaml`.

Hydra’s **`hydra.output_subdir`** defaults to **`null`** here (no `.hydra` folder under the run directory; Hydra still creates a timestamped `outputs/…` run dir unless you change `hydra.run.dir`). To use Hydra’s stock layout with a nested config snapshot directory, pass e.g. `hydra.output_subdir=.hydra`. The same default is set for **`pelinker-serves`** (`uv run python -m pelinker.cli.server`, `pelinker/conf/server.yaml`).

### Parameter alignment with `embed_kb_corpus.py`

| `pelinker-fit` (Hydra) | `run/embed_kb_corpus.py` (Click) |
|------------------------|----------------------------------|
| `kb_path` | `--kb-csv-path` |
| `input_text_table_path` | `--input-text-table-path` |
| `embeddings_parquet` | `--output-parquet-path` |
| `model_type`, `layers_spec`, … | `--model-type`, `--layers-spec`, … |

### Required inputs

- **`kb_path`**: Property KB CSV. Must include **`label`** and **`entity_id`** (labels are corpus patterns; IDs map fused properties to linker vocabulary—the same schema as **`--kb-csv-path`** for embedding).
- **`embeddings_parquet`**: Mention-level **Parquet** path—**output** of stage (A) and **input** of stage (B). For stage (B) only, it must already exist and match **`model_type`** / **`layers_spec`**.
- **`input_text_table_path`**: required by `embed_only` and `both`, and **rejected** by `fit_only`. When stage (A) runs it **writes** `embeddings_parquet`; otherwise stage (B) reads it.

### Optional parameters (defaults)

| Override | Default | Meaning |
|----------|---------|---------|
| `model_type` | `pubmedbert` | Embedding backbone (same vocabulary as `embed_kb_corpus.py` / `EmbeddingModelMetadata`). |
| `layers_spec` | `1` | Which layers to use (string parsed by `str2layers`; e.g. comma-separated indices). |
| `pipeline` | `embed_only` | Which stages run: `embed_only`, `fit_only`, `both`, `auto`. |
| `pca_components` | *(unset)* | PCA dimensionality before UMAP; unset takes `selection_report`'s value, else 100. |
| `umap_dim` | *(unset)* | UMAP output dimension for clustering; unset takes `selection_report`'s value, else 8. |
| `predict_mode` | `compact` | `compact` (ParametricUMAP + MLP entity head) or `legacy` (UMAP + HDBSCAN `approximate_predict`). |
| `screener_kind` | `lda` | Negative screener: `lda` or `svm`; persisted on the artifact. |
| `projection_enabled` | `true` | When false, skip the 3D manifold OOV score model, removing that predict-time gate. |
| `model_types` / `layers_specs` | *(unset)* | Per-parquet backbone and layers when fusing several parquets; length 1 broadcasts. Unset, the scalars apply unless the parquet stem matches `..._<model>_<layers>`. |
| `cluster_viz_method` | `pca` | Projection used for the cluster visualization: `pca` or `umap`. |
| `clustering_sample_index` | `0` | Bootstrap index for the clustering subsample; match model selection's `sample_idx` to reproduce its draw. |
| `clustering_sample_rows` | *(unset)* | Max mention rows per clustering bootstrap draw (stratified). Omit to use all loaded rows after filters. |
| `seed` | `13` | Bootstrap seed for clustering subsample draws; default for `mention_cap_seed` and `screener_seed`. |
| `pca_seed` | `13` | Random seed for PCA and cluster-viz PCA. |
| `umap_seed` | *(unset)* | UMAP random seed; omit for parallel UMAP. Set (e.g. `umap_seed=${seed}`) for reproducible production fits. |
| `drop_rare_entities` | `false` | Drop KB entities with fewer than `min_mentions_per_entity` rows before subsampling. |
| `min_mentions_per_entity` | `20` | Floor for `--drop-rare-entities` (negative label exempt). |
| `max_mentions_per_entity` | *(unset)* | Optional seeded cap on mention rows per KB entity before subsampling. |
| `max_mentions_negative` | *(unset)* | Optional cap on synthetic negative rows; omit to leave negatives uncapped. |
| `mention_cap_seed` | `seed` | RNG seed for per-entity mention cap (defaults to `seed`). |
| **`class_view`** | `reldir` | Classes the fit's ARI and catalog composition use (`pelinker/kb/classes.py`): `raw` (matched label), `rel` (canonical relation) or `reldir` (canonical relation + direction). `rel` / `reldir` need the pairs KB; `raw` reproduces earlier fits. See [Class views](#class-views). |
| `class_kb_path` | `kb_path` | Pairs KB the class view is computed from. |
| `min_cluster_size` | *(unset)* | HDBSCAN `min_cluster_size`. Omit to take it from `selection_report`, else `scale_curve_path`, else `20`. An explicit value always wins; the origin is recorded in the fit report under `min_cluster_size_provenance`. |
| **`selection_report`** | *(unset)* | `selected_hyperparameters.json` (or the report dir holding it) from `pelinker-model-selection` / `pelinker-dim-selection`. Fills in `pca_components`, `umap_dim`, `umap_n_neighbors`, `min_cluster_size` when those are not set here — so the search's winner reaches the fit instead of being retyped. |
| **`scale_curve_path`** | *(unset)* | `scale_curve.json` from `pelinker-scale-curve`. When set (and `min_cluster_size` is not), the hyperparameter is extrapolated to this fit's realized manifold row count rather than transferred verbatim from the selection sample size. |
| `umap_n_neighbors` | *(unset)* | UMAP `n_neighbors`; omit for the library default (15). Scale-dependent — 15 neighbours describe very different neighbourhoods at 10k and 2M rows. |
| `entity_head_holdout_fraction` | `0.15` | Rows withheld from entity-head training to measure distillation fidelity. `0.0` trains on every row and skips the measurement (reproduces pre-fidelity artifacts). |
| `entity_head_holdout_group_col` | `pmid` | Column kept whole across the holdout split; mentions from one document are correlated, so a row-level split inflates measured agreement. |
| `distillation_gates_enabled` | `true` | Check the fitted head against `distillation_min_entity_agreement` (0.95) and `distillation_max_emit_rate_rel_delta` (0.10). |
| `distillation_on_failure` | `warn` | `warn` keeps the model and records the breach in the report; `raise` aborts the fit. |
| **`model_path`** | *(required to fit)* | Base path `Linker.dump` writes to; `.gz` is appended for you. |
| **`report_path`** | *(required to fit)* | Directory for `linker_fit.clustering_report.json.gz`, `linker_fit.cluster_composition.json.gz` and `linker_fit.kb_out.json`. |
| `entity_head_hidden_layers` | `[256, 128, 128]` | MLP hidden sizes for `compact` mode. |
| `use_gpu` | `false` | GPU for transformer encoding when embedding the corpus. |
| `input_buffer_rows` | `1000` | Stage (A): rows per pandas read pass over the text table (I/O buffer; does **not** control GPU memory). |
| `encoder_batch_size` | `200` | Stage (A): table rows per encoder forward pass—**lower this if the GPU runs out of memory**. |
| `batch_size` | `1000` | Stage (B): rows per batch when **reading large embedding parquet files**; same role as `pelinker-model-selection --batch-size`. |
| `nlp_model` | `en_core_web_lg` | spaCy pipeline for mention extraction (pinned by the `dev` extra). |
| `max_input_buffers` | *(unset)* | Stage (A): stop after this many text-table read passes (each up to `input_buffer_rows` rows); unrelated to `encoder_batch_size`. |
| **`kb_name`** | stem of `kb_path` | Display name stored in `KBConfig`. |
| **`kb_version`** | `0.1.0` | KB version string stored on the model. |
| **`kb_created_at`** | today | ISO date string (`YYYY-MM-DD`); defaults to **today** if omitted. |
| **`kb_description`** | `""` | Free-form KB description. |
| **`kb_entity_count`** | *(unset)* | Optional; if omitted, may be filled from the fitted vocabulary in `KBConfig`. |

**There is no default output location.** A pipeline that fits requires both `model_path` and `report_path`, and fails rather than writing somewhere implicit. Give `model_path` without the `.gz` suffix — the linker appends it. A fit also refuses to overwrite: `both` and `embed_only` abort when a target parquet already exists.

**Migration (sample size):** `frac` / `eval_max_rows` / `n_embedding_batches` were replaced by `clustering_sample_rows` (absolute cap after load filters). Example: `frac=0.1` on 1M rows ≈ `clustering_sample_rows=100000`. Old `n_embedding_batches=50` with `batch_size=1000` truncated parquet reads before filters; use `clustering_sample_rows=50000` after filters instead.

### Examples

End-to-end: embed a corpus and fit a linker in one run (note `pipeline=both` — the default embeds only):

```bash
uv run pelinker-fit \
  pipeline=both \
  kb_path=data/derived/properties.synthesis.2.pairs.csv \
  input_text_table_path=<corpus>.tsv.gz \
  embeddings_parquet=<workdir>/corpus_pubmedbert_1.parquet \
  model_path=<models>/pelinker.pubmedbert.run1 \
  report_path=<workdir>/reports/run1
```

Embed only, to produce a parquet for the hyperparameter searches:

```bash
uv run pelinker-fit \
  kb_path=data/derived/properties.synthesis.2.pairs.csv \
  input_text_table_path=<corpus>.tsv.gz \
  embeddings_parquet=<workdir>/res_pubmedbert_1.parquet
```

Fit from an existing parquet, taking the search winner rather than retyping it:

```bash
uv run python -m pelinker.cli.fit \
  pipeline=fit_only \
  kb_path=data/derived/properties.synthesis.2.pairs.csv \
  embeddings_parquet=<workdir>/res_pubmedbert_1.parquet \
  selection_report=<workdir>/reports/dim_selection_2 \
  model_path=<models>/pelinker.from_parquet \
  report_path=<workdir>/reports/from_parquet
```

Short GPU smoke test truncating stage (A) after two table read passes (`input_buffer_rows` rows each unless the file ends sooner):

```bash
uv run pelinker-fit \
  kb_path=data/derived/properties.synthesis.2.pairs.csv \
  input_text_table_path=<corpus>.tsv.gz \
  embeddings_parquet=<workdir>/corpus_smoke_trunc.parquet \
  max_input_buffers=2 \
  input_buffer_rows=500 \
  use_gpu=true
```

## Class views

The KB is used as given; its entries and labels are never edited. What the objective and
the catalog count as one class is a *view* of it, computed from the pairs KB's own
declarations (`pelinker/kb/classes.py`):

| View | Class of a mention | Use |
|---|---|---|
| `raw` | the matched label | reproduces fits and searches from before views existed |
| `rel` | the canonical relation | catalog composition (merges between relations) |
| `reldir` | canonical relation + direction relative to it | **default** for the ARI of every search and of the fit |

**Why `reldir` is the default.** The weak-label matcher prefers the label whose own reading
has the mention's voice. A passive mention of a relation with a converse entry is
therefore stored under that entry, while a passive mention of a relation without one is
stored under the active label with `direction=inverse`. Against raw labels, ARI rewards
separating voices for the first kind of relation and penalizes it for the second.
`reldir` makes every passive an inverse class. `rel` would fold the voices together and
reward clusters that merge them, which removes the only direction signal an embedding-only
linker has.

The selection CLIs take `--class-view` and `--class-kb-path`, and the fit takes
`class_view=` and `class_kb_path=` (defaulting to `kb_path`). A view enters the search
checkpoint fingerprint, with the KB by content, so a run cannot resume under a different
view or a re-derived KB.

## Batch linking (`pelinker-link-files`)

Module: **`pelinker.cli.link_files`**. Console script: **`uv run pelinker-link-files`** (same as `uv run python -m pelinker.cli.link_files`).

Runs **`Linker.predict`** on one or more UTF-8 inputs (plain text = one document per file, or JSON objects / lists with a `text` field—see `--help`). Typical flags:

| Flag | Role |
|------|------|
| `-m` / `--model` | Linker artifact path (`Linker.load`; may omit or include `.gz` per loader rules). |
| `--thr-score` | Minimum cluster membership score (same role as server `thr_score`). |
| `-o` / `--output` | Write the full JSON report (entities, scores, optional GT echo). |
| `--dump-mention-anomaly PATH` | Per-mention table with PCA residual / Mahalanobis-style metrics; format from extension: **`.parquet`**, **`.csv`**, **`.jsonl`**. Feeds **`run/analysis/oov_analysis.py`**. |
| `--include-anomaly-metrics` | Attach anomaly fields to **entity** records in the JSON output. |
| `--kb-validation` | Include KB lemma validation-style fields where applicable. |
| `--use-gpu` | CUDA for the encoder path when available. |

## HTTP server smoke tests

After you have a dumped linker (the packaged default, or `model_path` from a fit), you can run the **FastAPI** server from **`pelinker.cli.server`**. Configuration uses Hydra like fit; defaults live in `pelinker/conf/server.yaml`.

**Start the server** (pick one):

- `uv run pelinker-serves` (console script from `pyproject.toml`)
- `uv run python -m pelinker.cli.server` (equivalent module invocation)

Common Hydra overrides: `host`, `port` (default **8599**), `model_file_spec` (linker dump **without** the `.gz` suffix—same rule as `Linker.load`), `thr_score`, `use_gpu`, `cors_allow_origins`. API routes include `GET /health`, `GET /info`, `GET /model`, `POST /link`, and `POST /link/debug`; interactive docs are at **`/docs`** when the server is up.

### `smoke_server.py`

Small **Click** client in this directory to hit those routes while developing. Start the server in one terminal, then:

```bash
uv run python run/smoke_server.py --endpoint health
uv run python run/smoke_server.py --endpoint info
uv run python run/smoke_server.py --endpoint model
uv run python run/smoke_server.py --endpoint link
uv run python run/smoke_server.py --endpoint link-debug
```

Use **`--host`** / **`--port`** so they match the running server (defaults: `localhost` and **8599**). For **`link`** and **`link-debug`**, omit **`--input-path`** to send a built-in two-document `texts` example, or pass a JSON file whose root is an object with **`text`** or **`texts`** (optional keys such as `thr_score`, `use_gpu`, `max_length`; for debug, `include_entity_anomaly_metrics`, `kb_validation`). Plain or **`.json.gz`** files are accepted (via `pelinker.data.load_json_path`). Use **`--output`** to write the JSON response to a file instead of printing; **`--timeout`** defaults to 300 seconds for slow cold starts.

## Analysis Scripts

Scripts in the `analysis/` directory evaluate embedding quality and select diverse entities.

### `pelinker-model-selection`

Implementation: [`pelinker.search.model_selection`](../pelinker/search/model_selection/); CLI `pelinker/cli/model_selection.py`.

Measures the quality of embeddings obtained from stage (A) by evaluating clustering performance. Required: `--input-dir`, `--report-path`.

- **Purpose**: Evaluates how well embeddings cluster semantically similar properties together
- **Input**: Directory containing parquet files (pattern: `res_<model>_<layer>.parquet`)
- **Outputs**:
  - `model_selection.run_report.json.gz` - Standardized run-level model-selection report
  - `model.perf.heatmap.png` - Heatmap of best scores across models
  - `model.ari.heatmap.png` - Heatmap of ARI clustering quality (if available)
  - `{model}_{layer}.png` - Metrics plots for each model/layer
  - `umap_best.html` - Interactive UMAP visualization of the best performing model
- **Key Features**:
  - Evaluates multiple model/layer combinations
  - Optimizes cluster size using various metrics
  - Supports multiple sampling runs for statistical robustness
  - Shared mention-frame load with `pelinker-fit`: optional `--drop-rare-entities`, `--max-mentions-per-entity`, then `--clustering-sample-rows` (omit = all loaded rows)
  - **Optional**: `--selected-labels-kb-path` parameter to evaluate quality over a specific subset of labels from a selected knowledge base CSV file
  - `--class-view` (default `reldir`) and `--class-kb-path` set the classes ARI scores against — see [Class views](#class-views). The same two options exist on `pelinker-dim-selection` and `pelinker-scale-curve`
- **Metrics** (two-level, same as `dim_selection.py`):
  - **MCS** (`min_cluster_size`): HDBSCAN hyperparameter — smallest cluster HDBSCAN will form; searched on an inner grid
  - **Inner** (choose MCS): `grid_objective=dbcv_ari_geomean` — clip mean DBCV and mean ARI at 0, take `sqrt(dbcv*ari)` per bootstrap sample, smooth, then pick the largest MCS within one *paired* standard error of the best (`grid_one_se_k`, default 1.0). The geometric mean ranks grid points identically under any rescaling of either metric, so nothing needs normalizing
  - **Outer** (rank model×layer): at each combo’s pooled MCS, combine mean DBCV + mean ARI with the *same* clipped geometric mean. Each candidate is scored from its own numbers, so adding one cannot reorder the others. Column `best_score` remains mean DBCV (heatmaps); `outer_score` chooses the winner

### `compact_predict_study.py`

Compares **legacy** (UMAP + HDBSCAN `approximate_predict`) vs **compact** (ParametricUMAP + MLP entity head) on one embeddings parquet before trusting the production compact default.

- **Arms**: A legacy, B compact (shipped), C iso-manifold, D iso-head, E LinearSVC underfit control
- **Gates**: entity-id agreement vs A ≥ 0.95, emit-rate within ±10%, size ≤ 15 MB or ≥5× smaller, latency ≤ 1.5× A
- **Split**: 65/15/20 train/tune/holdout, **grouped by `pmid`** via `pelinker.linker.distillation.grouped_holdout_split`. It was a plain row shuffle before, which put mentions of the same document on both sides and made every agreement number optimistic.
- **Outputs**: `arms.csv`, `summary.json` with explicit pass/fail under `--report-dir`
- **Example**: `uv run python run/analysis/compact_predict_study.py --embeddings-parquet … --report-dir …`
- **Run it before trusting `predict_mode=compact` on your data.** The arms are built so a
  failure is attributable: C and D vary the manifold and the head one at a time, so a gap
  between the shipped compact path and legacy can be charged to the ParametricUMAP manifold
  or to the MLP entity head rather than to "compact mode" as a whole, and E is an underfit
  control that should *not* pass. Reports land under `--report-dir`, which is gitignored —
  keep the numbers with the measurement writeup, not in this repo.

For a per-fit number rather than this five-arm audit, `pelinker-fit` now measures held-out student-vs-teacher fidelity on **every** compact fit and writes a `distillation_fidelity` block into `linker_fit.clustering_report.json.gz` — see [Fitting the linker model](#fitting-the-linker-model).

### `pelinker-scale-curve`

Implementation: [`pelinker.search.scale_curve`](../pelinker/search/scale_curve/); CLI `pelinker/cli/scale_curve.py`.

Measures how the chosen `min_cluster_size` moves with the mention-frame size, instead of transferring an integer picked on a subsample straight into a full-corpus fit.

- **Why**: `min_cluster_size` is an absolute row count — and so, by HDBSCAN's default, is `min_samples`. Selection runs at `--clustering-sample-rows`; the fit runs on everything. The plateau solver min–max normalizes *within* each curve, so the choice is driven by the shape of f(MCS) over a fixed absolute grid: change N and the plateau moves while the grid does not.
- **How**: one full inner search per "rung" (a `clustering_sample_rows` value), reusing the production path (`draw_selection_sample` → `evaluate_selection_sample` → pooled MCS), then least squares on `log(MCS*) ~ a + b·log(N)`.
- **Outputs** (under `--report-path`): `scale_curve.json` (consumed by `pelinker-fit scale_curve_path=…`), `scale_curve.{png,pdf}` (log-log; pinned rungs drawn hollow in red), and per-sample grid rows.
- **Reading the exponent `b`**: `~0` means the absolute value transfers fine and today's behaviour was right; `0 < b < 1` sublinear growth (the expected regime); `~1` a constant fraction of N.
- **When not to trust it**: rungs flagged **pinned** (the chosen value sat on a `[--min-scale, --max-scale)` bound) or a low R² mean the grid, not the sample size, decided the answer. Widen the grid and re-run; the CLI prints this warning itself.

```bash
uv run pelinker-scale-curve \
  --input-parquet <workdir>/res_pubmedbert_2.parquet \
  --report-path <workdir>/reports/scale_curve_2 \
  --class-kb-path data/derived/properties.synthesis.2.pairs.csv \
  --rungs 10000,25000,50000,100000 \
  --pca-components 22 --umap-dim 3 --n-sample 3
```

### `pelinker-dim-selection`

Implementation: [`pelinker.search.dim_selection`](../pelinker/search/dim_selection/); CLI `pelinker/cli/dim_selection.py`.

After model selection picks a winning embedding combo, search **`(pca_components, umap_dim)`** on that single parquet with the same clustering metrics stack. Required: `--input-parquet`, `--report-path`.

- **Purpose**: Choose robust PCA and UMAP dimensions for the transform pipeline (defaults today: 100 and 8)
- **Input**: One mention-level parquet (`--input-parquet`); model/layer parsed from the filename (or `--model` / `--layer`)
- **Search**: Coarse grid (default PCA `40,80,120,180` × UMAP `4,6,8,12`), then optional local refine around the winner (`--refine` / `--no-refine`)
- **Sampling**: same mention-frame load as model selection / fit — optional `--drop-rare-entities`, `--max-mentions-per-entity`, then `--clustering-sample-rows` (omit = all loaded rows)
- **Outputs** (under `--report-path`):
  - `dim_selection.results.csv` — per-cell mean DBCV / ARI / `outer_score` / pooled MCS
  - `dim.outer.heatmap.{png,pdf}` / `dim.dbcv.heatmap.{png,pdf}` / `dim.ari.heatmap.{png,pdf}` — PCA × UMAP heatmaps
  - `dim.outer.surface.{png,pdf}` — 3D surface of outer (combined DBCV+ARI) score over PCA × UMAP
  - `dim.metrics.violin.{png,pdf}` — per-bootstrap DBCV / ARI violins across cells (`n_sample` ≥ 2)
  - `dim.dbcv_vs_ari.{png,pdf}` — DBCV vs ARI scatter (one point/ellipse per cell)
  - `dim_selection.summary.json` — chosen dims + metrics documentation (includes MCS glossary) + figure list
  - `dim_selection.state.json.gz` — resumable checkpoint
  - `results_grid_per_sample.csv` — per-sample MCS grid curves (feeds violin / DBCV–ARI scatter)
- **Metrics**: identical two-level stack as model selection — **inner and outer both use DBCV+ARI**; MCS = `min_cluster_size`
- **Example**:

```bash
uv run python -m pelinker.cli.dim_selection \
  --input-parquet <workdir>/res_pubmedbert_2.parquet \
  --report-path <workdir>/reports/dim_selection_2 \
  --class-kb-path data/derived/properties.synthesis.2.pairs.csv \
  --n-sample 3 \
  --clustering-sample-rows 10000
```

### `select_diverse_entities.py`

Selects semantically diverse entities from a knowledge base using clustering-based selection.

- **Purpose**: Identifies a diverse subset of entities that represent the semantic space of the full knowledge base
- **Input**: CSV/TSV file with entity IDs and labels
- **Output**: CSV file with selected diverse entities
- **Process**:
  - Embeds all labels using a transformer model
  - Applies PCA for dimensionality reduction
  - Uses K-means clustering to identify diverse groups
  - Selects the most representative entity from each cluster (preferring generic/simple terms)
- **Use Case**: Useful for creating evaluation sets or reducing knowledge base size while maintaining semantic coverage

### `oov_analysis.py`

Publication-style **figures** for pre-classifier anomaly space: compares training (fit) mentions with **KB-validated OOV** vs **unconfirmed OOV** using PCA residual and Mahalanobis-style distances from the fit-B clustering report, plus ROC/PR and decision-boundary sweeps.

- **Inputs**:
  - **`--fit-report`**: gzipped JSON clustering report with `pca_residuals` / `pca_mahalanobis` (training anchor distribution).
  - **`--oov-csv`**: Mention-level table from **`pelinker-link-files --dump-mention-anomaly`** (or equivalent columns).
  - **`--out-dir`**: Directory for output PDFs.
- **Run**: `uv run python run/analysis/oov_analysis.py --help` for the full CLI (composite “paper” figure, alignment with the negative screener, etc.).
- **Dependencies**: Uses **`matplotlib`** / **`seaborn`**; install optional **dev** extras if needed (`uv sync --extra dev` so the plotting stack matches `pyproject.toml`).

### `pelinker-replot`

Regenerates the model-selection figures — including the **DBCV vs ARI** scatter — from the
artifacts already under a report directory, with no re-embedding and no re-clustering.

```bash
uv run pelinker-replot <report-dir>
```

Takes the report directory as its one positional argument. Optional: `--checkpoint`,
`--grid-one-se-k` (re-solve the grid with a different one-SE width),
`--grid-cluster-count-reward`, `--grid-n-entities`, `--all-pca-pairgrid-samples`.

### `replot_fit.py`

Visualises cluster composition from a **fit** report (as opposed to a selection report):
composition bars and pies, an interactive cluster view, and an entity→cluster Sankey.

```bash
uv run python run/analysis/replot_fit.py <report-dir>
```

Reads `linker_fit.clustering_report.json.gz`, `linker_fit.cluster_composition.json.gz` and
`linker_fit.kb_out.json`. Optional: `--top-n` (3), `--max-clusters`, `--max-entities`,
`--pmid-text-table`, `--sankey-min-frac` (0.0), `--cluster-label` (`display` | `short`),
`--viz-all-kb`, `--show`.

### `cluster_stability.py`

Asks whether clusters are the same *objects* from one bootstrap draw to the next, rather
than merely the same in number — a count can be stable while membership churns.

```bash
uv run python run/analysis/cluster_stability.py \
  --labels-parquet <report-dir>/sample_cluster_labels.parquet \
  --report-dir <workdir>/stability
```

The input comes from `pelinker-dim-selection --persist-labels` (off by default). Writes
`stability_summary.json`, `stability_pairs.csv` and `stability_per_cluster.csv`; optional
`--jaccard-floor` (0.5) sets what counts as the same cluster across draws.

### `weak_label_check.py`

Acceptance check for the weak labels in a stage-(A) mention parquet. Run it on a smoke
run before a full embedding grid. It reports how labels were assigned (`surface_rule`,
`direction`) and fails, with a non-zero exit, on any of the following:
- a span carrying two labels;
- a labelled span nested inside another;
- a span labelled with both a relation and its converse;
- a symmetric relation with `direction=inverse`;
- a label the KB does not hold (the class views refuse it).

For a pairs KB it also reports the number of classes each view yields and the mention
mass by direction relative to the canonical relation; `classes_reldir` is the number of
classes the selection objective scores against.

```bash
uv run python run/analysis/weak_label_check.py \
  --parquet <dir>/res_pubmedbert_1.parquet \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv
```

### `direction_diagnostic.py`

Sizes the directionality problem before any model is built: how far converse pairs collide
in the learned space, reported as collision rate, co-membership and separation over the
KB's declared pairs.

```bash
uv run python run/analysis/direction_diagnostic.py \
  --kb-csv-path data/derived/properties.synthesis.2.pairs.csv \
  --model-path <models>/pelinker.pubmedbert.run1 \
  --report-dir <workdir>/direction
```

Takes exactly one source of assignments: `--model-path`, `--fit-report` or
`--assignments-parquet`. It deliberately refuses the cluster-composition artifact, which is
truncated to the top N entities per cluster and would understate collisions.
