# Run Scripts Documentation

This directory contains scripts for preprocessing knowledge bases, embedding corpora, and analyzing embedding quality.

For a **single end-to-end train** (corpus embedding → KB filtering/aggregation → PCA/UMAP → optimized HDBSCAN clustering → serialized linker artifact), use the packaged CLI `pelinker.cli.fit` documented in [Fitting the linker model](#fitting-the-linker-model) below.

## Directory Structure

```
run/
├── README.md                    # This file
├── embed_kb_corpus.py          # Embed knowledge base corpus
├── loop.embed.kb.corpus.sh     # Batch embedding (grid over model × layer)
├── loop.fit.sh                 # Batch full fit: same grid, runs pelinker-fit (A+B)
├── preprocessing/               # Property knowledge base generation
│   ├── extract_properties_go.py    # Extract from GO-CAMs ontology
│   ├── extract_properties_ro.py    # Extract from Relations Ontology
│   └── merge_properties.py         # Merge properties from all sources
├── analysis/                    # Embedding quality evaluation
│   ├── clustering_quality.py       # Measure clustering quality of embeddings
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
- **Other useful flags**: `--model-type`, `--layers-spec`, `--use-gpu`, `--input-buffer-rows`, `--encoder-batch-size`, `--nlp-model`, `--max-input-buffers` (mirror `pelinker.cli.fit` defaults where applicable; default **`--model-type`** is `pubmedbert` to match fit). **`--encoder-batch-size`** is the transformer forward batch (GPU memory); **`--input-buffer-rows`** is only how many table rows pandas reads per pass.

### `loop.embed.kb.corpus.sh` / `loop.fit.sh`

- **`loop.embed.kb.corpus.sh`**: loops over the same default **`model_type` × `layers_spec`** grid and runs **`embed_kb_corpus.py`** only (Parquet per combo).
- **`loop.fit.sh`**: same grid, but runs **`uv run pelinker-fit`** per combo—**stage (A)** writes `res_<model>_<layer_tag>.parquet` under **`--output-parquet-prefix`**, **stage (B)** writes **`pelinker.<model>.<layer_tag>.gz`** under **`--output-model-prefix`** (`layer_tag` is `layers_spec` with commas replaced by `_` for filenames). Requires four flags: `--input-text-table-path`, `--kb-csv-path`, `--output-parquet-prefix`, **`--output-model-prefix`**. Optional **`--layers`**: **`layers_spec`** list (default `1,2,3`). Comma separates distinct specs; use **semicolons** when one spec contains commas, e.g. `--layers 1,2,3`, `--layers 1`, or `--layers '1,2;3'` (runs `1,2` then `3`).

## Fitting the linker model

Module: **`pelinker.cli.fit`**. It runs the linker training pipeline in **two conceptual stages**:

1. **Stage (A)** — `embed_kb_corpus(...)` (same function as `run/embed_kb_corpus.py`) when **`input_text_table_path`** is set: **`kb_path`** + text table → **`embeddings_parquet`**.
2. **Stage (B)** — `Linker.fit(...)` on that Parquet: fusion / PCA / UMAP / optimized HDBSCAN → fitted linker; serialized via `Linker.dump` (joblib at **`{output_path}.gz`**; the `.gz` suffix is appended automatically).

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
| `min_class_size` | `20` | Lower bound for clustering optimization / HDBSCAN (see `ClusteringOptimizationConfig`). |
| `max_scale` | `120` | Upper bound for the optimization grid over `min_cluster_size`. |
| `output_path` | *(see below)* | Where `linker.dump` writes the artifact. |
| `use_gpu` | `false` | GPU for transformer encoding when embedding the corpus. |
| `input_buffer_rows` | `1000` | Stage (A): rows per pandas read pass over the text table (I/O buffer; does **not** control GPU memory). |
| `encoder_batch_size` | `200` | Stage (A): table rows per encoder forward pass—**lower this if the GPU runs out of memory**. |
| `frac` | `1.0` | Fraction of rows to keep when sampling the mention parquet for clustering optimization (stage B); same idea as `clustering_quality.py --frac`. |
| `batch_size` | `1000` | Stage (B): rows per batch when **reading large embedding parquet files**; same role as `clustering_quality.py --batch-size`. |
| `nlp_model` | `en_core_web_trf` | spaCy pipeline for mention extraction (`uv run spacy download en_core_web_trf`). |
| `max_input_buffers` | *(unset)* | Stage (A): stop after this many text-table read passes (each up to `input_buffer_rows` rows); unrelated to `encoder_batch_size`. |
| `head` | *(unset)* | Stage B only: max number of parquet read batches (see `pelinker.analysis` / `clustering_quality.py --head`). |
| **`kb_name`** | stem of `kb_path` | Display name stored in `KBConfig`. |
| **`kb_version`** | `0.1.0` | KB version string stored on the model. |
| **`kb_created_at`** | today | ISO date string (`YYYY-MM-DD`); defaults to **today** if omitted. |
| **`kb_description`** | `""` | Free-form KB description. |
| **`kb_entity_count`** | *(unset)* | Optional; if omitted, may be filled from the fitted vocabulary in `KBConfig`. |

**Default output location**: if `output_path` is not set, the model is written under the `pelinker.store` package resources as `pelinker.model.{model_type}.{layers_str}` (e.g. `pelinker.model.pubmedbert.1` → **`pelinker.model.pubmedbert.1.gz`** next to that package resource). Set `output_path` to an explicit filesystem path (without adding `.gz` yourself) for reproducible artifacts.

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

## Analysis Scripts

Scripts in the `analysis/` directory evaluate embedding quality and select diverse entities.

### `clustering_quality.py`

Measures the quality of embeddings obtained from `embed_kb_corpus.py` by evaluating clustering performance.

- **Purpose**: Evaluates how well embeddings cluster semantically similar properties together
- **Input**: Directory containing parquet files (pattern: `res_<model>_<layer>.parquet`)
- **Outputs**:
  - `results.csv` - Summary table with metrics for each model/layer combination
  - `model.perf.heatmap.png` - Heatmap of best scores across models
  - `model.ari.heatmap.png` - Heatmap of ARI clustering quality (if available)
  - `{model}_{layer}.png` - Metrics plots for each model/layer
  - `umap_best.html` - Interactive UMAP visualization of the best performing model
- **Key Features**:
  - Evaluates multiple model/layer combinations
  - Optimizes cluster size using various metrics
  - Supports multiple sampling runs for statistical robustness
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
