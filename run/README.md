# Run Scripts Documentation

This directory contains scripts for preprocessing knowledge bases, embedding corpora, analyzing embedding quality, and producing **OOV / manifold anomaly** figures from fit reports plus batch-linked mention dumps.

For a **single end-to-end train** (corpus embedding → KB filtering/aggregation → PCA/UMAP → optimized HDBSCAN clustering → serialized linker artifact), use the packaged CLI `pelinker.cli.fit` documented in [Fitting the linker model](#fitting-the-linker-model) below.

The published **MkDocs** site mirrors this layout at a high level under [Run scripts & CLIs](https://growgraph.github.io/pelinker/user_guide/run_scripts_and_cli/) (source: `docs/user_guide/run_scripts_and_cli.md`). Package **version** is `version` in the repo root `pyproject.toml`.

## Directory Structure

```
run/
├── README.md                    # This file
├── embed_kb_corpus.py          # Embed knowledge base corpus
├── test_server.py              # Smoke-test pelinker.cli.server HTTP routes
├── loop.embed.kb.corpus.sh     # Batch embedding (grid over model × layer)
├── loop.fit.sh                 # Batch full fit: same grid, runs pelinker-fit (A+B)
├── preprocessing/               # Property knowledge base generation
│   ├── extract_properties_go.py    # Extract from GO-CAMs ontology
│   ├── extract_properties_ro.py    # Extract from Relations Ontology
│   └── merge_properties.py         # Merge properties from all sources
├── analysis/                    # Embedding quality & OOV diagnostics
│   ├── model_selection.py          # Model selection over embedding combinations
│   ├── oov_analysis.py             # Fit report + OOV mention dump → PDF figures
│   ├── replot_dbcv_ari_scatter.py  # DBCV vs ARI scatter from existing grid CSV
│   └── select_diverse_entities.py  # Select diverse entity subsets
└── obsolete/                    # Deprecated scripts (not actively maintained)
    ├── analysis/
    ├── experiments/
    ├── preprocessing/
    └── testing/
```

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
- **Output**: `data/derived/properties.ro.csv` - Extracted properties with entity IDs, labels, and descriptions
- **Process**: Parses the RO OWL file and extracts object properties with their labels and descriptions

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

## Embedding Scripts

### `embed_kb_corpus.py`

Embeds a knowledge base corpus using the same pipeline as **stage (A)** of `pelinker.cli.fit` (both call `pelinker.embedder.embed_kb_corpus`).

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
| `--nlp-model` | `en_core_web_trf` | spaCy pipeline used for tokenization/lemma processing around mention extraction. | Ensure the model is installed in the `uv` env. |
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

- **`loop.embed.kb.corpus.sh`**: loops over the same default **`model_type` × `layers_spec`** grid and runs **`embed_kb_corpus.py`** only (Parquet per combo).
- **`loop.fit.sh`**: same grid, but runs **`uv run pelinker-fit`** per combo—**stage (A)** writes `res_<model>_<layer_tag>.parquet` under **`--output-parquet-prefix`**, **stage (B)** writes **`pelinker.<model>.<layer_tag>.gz`** under **`--output-model-prefix`** (`layer_tag` is `layers_spec` with commas replaced by `_` for filenames). Requires four flags: `--input-text-table-path`, `--kb-csv-path`, `--output-parquet-prefix`, **`--output-model-prefix`**. Optional **`--layers`**: **`layers_spec`** list (default `1,2,3`). Comma separates distinct specs; use **semicolons** when one spec contains commas, e.g. `--layers 1,2,3`, `--layers 1`, or `--layers '1,2;3'` (runs `1,2` then `3`).

## Fitting the linker model

Module: **`pelinker.cli.fit`**. It runs the linker training pipeline in **two conceptual stages**:

1. **Stage (A)** — `embed_kb_corpus(...)` (same function as `run/embed_kb_corpus.py`) when **`input_text_table_path`** is set: **`kb_path`** + text table → **`embeddings_parquet`**.
2. **Stage (B)** — `Linker.fit(...)` on that Parquet: fusion / negative screener / PCA / UMAP / HDBSCAN at a fixed `min_cluster_size` → fitted linker; serialized via `Linker.dump` (joblib at **`{output_path}.gz`**; the `.gz` suffix is appended automatically). Choose `min_cluster_size` upstream (e.g. `pelinker.model_selection`); this CLI does not run a grid search during fit.

If you omit **`input_text_table_path`**, only **stage (B)** runs (you must already have **`embeddings_parquet`** on disk, e.g. from a prior `embed_kb_corpus.py` run).

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
- **`input_text_table_path`** (optional): If set, stage (A) runs and **writes** **`embeddings_parquet`** via `embed_kb_corpus`. If omitted, stage (B) **reads** the existing file.

### Optional parameters (defaults)

| Override | Default | Meaning |
|----------|---------|---------|
| `model_type` | `pubmedbert` | Embedding backbone (same vocabulary as `embed_kb_corpus.py` / `EmbeddingModelMetadata`). |
| `layers_spec` | `1` | Which layers to use (string parsed by `str2layers`; e.g. comma-separated indices). |
| `pca_components` | `100` | PCA dimensionality before UMAP. |
| `umap_dim` | `8` | UMAP output dimension for clustering. |
| `clustering_sample_rows` | *(unset)* | Max mention rows per clustering bootstrap draw (stratified). Omit to use all loaded rows after filters. |
| `seed` | `13` | Bootstrap seed for clustering subsample draws; default for `mention_cap_seed` and `screener_seed`. |
| `pca_seed` | `13` | Random seed for PCA and cluster-viz PCA. |
| `umap_seed` | *(unset)* | UMAP random seed; omit for parallel UMAP. Set (e.g. `umap_seed=${seed}`) for reproducible production fits. |
| `drop_rare_entities` | `false` | Drop KB entities with fewer than `min_mentions_per_entity` rows before subsampling. |
| `min_mentions_per_entity` | `20` | Floor for `--drop-rare-entities` (negative label exempt). |
| `max_mentions_per_entity` | *(unset)* | Optional seeded cap on mention rows per KB entity before subsampling. |
| `max_mentions_negative` | *(unset)* | Optional cap on synthetic negative rows; omit to leave negatives uncapped. |
| `mention_cap_seed` | `seed` | RNG seed for per-entity mention cap (defaults to `seed`). |
| `min_cluster_size` | `20` | HDBSCAN `min_cluster_size` for stage (B); set from analysis / grid search outside this CLI. |
| `output_path` | *(see below)* | Where `linker.dump` writes the artifact. |
| `use_gpu` | `false` | GPU for transformer encoding when embedding the corpus. |
| `input_buffer_rows` | `1000` | Stage (A): rows per pandas read pass over the text table (I/O buffer; does **not** control GPU memory). |
| `encoder_batch_size` | `200` | Stage (A): table rows per encoder forward pass—**lower this if the GPU runs out of memory**. |
| `batch_size` | `1000` | Stage (B): rows per batch when **reading large embedding parquet files**; same role as `model_selection.py --batch-size`. |
| `nlp_model` | `en_core_web_trf` | spaCy pipeline for mention extraction (`uv run spacy download en_core_web_trf`). |
| `max_input_buffers` | *(unset)* | Stage (A): stop after this many text-table read passes (each up to `input_buffer_rows` rows); unrelated to `encoder_batch_size`. |
| **`kb_name`** | stem of `kb_path` | Display name stored in `KBConfig`. |
| **`kb_version`** | `0.1.0` | KB version string stored on the model. |
| **`kb_created_at`** | today | ISO date string (`YYYY-MM-DD`); defaults to **today** if omitted. |
| **`kb_description`** | `""` | Free-form KB description. |
| **`kb_entity_count`** | *(unset)* | Optional; if omitted, may be filled from the fitted vocabulary in `KBConfig`. |

**Default output location**: if `output_path` is not set, the model is written under the `pelinker.store` package resources as `pelinker.model.{model_type}.{layers_str}` (e.g. `pelinker.model.pubmedbert.1` → **`pelinker.model.pubmedbert.1.gz`** next to that package resource). Set `output_path` to an explicit filesystem path (without adding `.gz` yourself) for reproducible artifacts.

**Migration (sample size):** `frac` / `eval_max_rows` / `n_embedding_batches` were replaced by `clustering_sample_rows` (absolute cap after load filters). Example: `frac=0.1` on 1M rows ≈ `clustering_sample_rows=100000`. Old `n_embedding_batches=50` with `batch_size=1000` truncated parquet reads before filters; use `clustering_sample_rows=50000` after filters instead.

### Examples

Embed a corpus from the synthesized KB and save to a known path:

```bash
uv run pelinker-fit \
  kb_path=data/derived/properties.synthesis.1.csv \
  input_text_table_path=data/corpus/articles.tsv.gz \
  embeddings_parquet=outputs/corpus_pubmedbert_1.parquet \
  output_path=models/pelinker.pubmedbert.run1
```

Reuse parquet output from a prior `embed_kb_corpus.py` run and tune UMAP:

```bash
uv run python -m pelinker.cli.fit \
  kb_path=data/derived/properties.synthesis.1.csv \
  embeddings_parquet=outputs/res_pubmedbert_1.parquet \
  umap_dim=12 \
  output_path=models/pelinker.from_parquet
```

Short GPU smoke test truncating stage (A) after two table read passes (`input_buffer_rows` rows each unless the file ends sooner):

```bash
uv run pelinker-fit \
  kb_path=data/derived/properties.synthesis.1.csv \
  input_text_table_path=data/corpus/articles.tsv.gz \
  embeddings_parquet=outputs/corpus_smoke_trunc.parquet \
  max_input_buffers=2 \
  input_buffer_rows=500 \
  use_gpu=true
```

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

After you have a dumped linker (packaged default or `output_path` from fit), you can run the **FastAPI** server from **`pelinker.cli.server`**. Configuration uses Hydra like fit; defaults live in `pelinker/conf/server.yaml`.

**Start the server** (pick one):

- `uv run pelinker-serves` (console script from `pyproject.toml`)
- `uv run python -m pelinker.cli.server` (equivalent module invocation)

Common Hydra overrides: `host`, `port` (default **8599**), `model_file_spec` (linker dump **without** the `.gz` suffix—same rule as `Linker.load`), `thr_score`, `use_gpu`, `cors_allow_origins`. API routes include `GET /health`, `GET /info`, `GET /model`, `POST /link`, and `POST /link/debug`; interactive docs are at **`/docs`** when the server is up.

### `test_server.py`

Small **Click** client in this directory to hit those routes while developing. Start the server in one terminal, then:

```bash
uv run python run/test_server.py --endpoint health
uv run python run/test_server.py --endpoint info
uv run python run/test_server.py --endpoint model
uv run python run/test_server.py --endpoint link
uv run python run/test_server.py --endpoint link-debug
```

Use **`--host`** / **`--port`** so they match the running server (defaults: `localhost` and **8599**). For **`link`** and **`link-debug`**, omit **`--input-path`** to send a built-in two-document `texts` example, or pass a JSON file whose root is an object with **`text`** or **`texts`** (optional keys such as `thr_score`, `use_gpu`, `max_length`; for debug, `include_entity_anomaly_metrics`, `kb_validation`). Plain or **`.json.gz`** files are accepted (via `pelinker.io.load_json_path`). Use **`--output`** to write the JSON response to a file instead of printing; **`--timeout`** defaults to 300 seconds for slow cold starts.

## Analysis Scripts

Scripts in the `analysis/` directory evaluate embedding quality and select diverse entities.

### `model_selection.py`

Implementation: [`pelinker.model_selection`](../../pelinker/model_selection/) (this script is a thin shim).

Measures the quality of embeddings obtained from `embed_kb_corpus.py` by evaluating clustering performance.

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
- **Metrics**: Best cluster size, number of properties, clustering score, adjusted Rand index (ARI)

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

### `replot_dbcv_ari_scatter.py`

Rebuilds the **DBCV vs ARI** scatter PNG from an existing **`results_grid_per_sample.csv`** produced by `model_selection.py` (no re-embedding).

```bash
uv run python run/analysis/replot.py path/to/results_grid_per_sample.csv
```

Optional **`-o` / `--output`** overrides the default path next to the CSV (`model.dbcv_vs_ari.png`).
