# Run Scripts Documentation

This directory contains scripts for preprocessing knowledge bases, embedding corpora, and analyzing embedding quality.

## Directory Structure

```
run/
├── README.md                    # This file
├── embed_kb_corpus.py          # Embed knowledge base corpus
├── loop.embed.kb.corpus.sh     # Batch embedding script
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

Embeds a knowledge base corpus using transformer models for downstream analysis and linking tasks.

- **Purpose**: Processes text corpora and extracts mentions of properties, generating embeddings for each mention
- **Inputs**:
  - `--input-text-table-path`: TSV/CSV file with `pmid` and `text` columns (optionally gzipped)
  - `--properties-txt-path`: Newline-separated list of properties/patterns to search for
- **Output**: Parquet file containing extracted mentions with their embeddings
- **Features**:
  - Supports multiple model types (biobert, pubmedbert, scibert, etc.)
  - Configurable layer selection for embeddings
  - GPU acceleration support
  - Streaming processing with chunking for large datasets
  - Extracts mentions at multiple word grouping levels (W1, W2, W3)

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
