# Run scripts and CLIs

This page is the **high-level map** for training, serving, batch linking, and offline analysis. Detailed flags, tables, and preprocessing steps live in the repository’s [`run/README.md`](https://github.com/growgraph/pelinker/blob/main/run/README.md) (kept next to the scripts).

## Environment

From the repository root, use **`uv`** so dependencies match `uv.lock`:

```bash
uv sync --all-groups
uv run spacy download en_core_web_trf
```

Documentation site builds (optional):

```bash
uv sync --extra docs
uv run mkdocs serve
```

## Packaged commands

| Command | Module | Role |
|--------|--------|------|
| `uv run pelinker-fit` | `pelinker.cli.fit` | Corpus embedding (optional) + `Linker.fit` → serialized artifact (`.gz`). Hydra overrides; defaults in `pelinker/conf/fit.yaml`. |
| `uv run pelinker-serves` | `pelinker.cli.server` | FastAPI server: `/health`, `/info`, `/model`, `/link`, `/link/debug`. Defaults in `pelinker/conf/server.yaml`. |
| `uv run pelinker-link-files` | `pelinker.cli.link_files` | Batch `Linker.predict` on UTF-8 files or JSON documents; optional JSON report and **mention-level anomaly dump** for OOV workflows. |

Equivalent module invocations: `uv run python -m pelinker.cli.fit`, `pelinker.cli.server`, `pelinker.cli.link_files`.

### Batch linking (`pelinker-link-files`)

- **Model**: `-m` / `--model` — path to the linker dump (`Linker.load` rules; `.gz` is resolved like elsewhere).
- **Threshold**: `--thr-score` — same idea as the server’s score threshold.
- **Outputs**: `-o` / `--output` — full prediction JSON; `--dump-mention-anomaly PATH` — per-mention rows with PCA residual / Mahalanobis-style metrics (extension selects **`.parquet`**, **`.csv`**, or **`.jsonl`**).
- **Extras**: `--include-anomaly-metrics` and `--kb-validation` mirror server/debug style fields on entities; `--use-gpu` for CUDA when available.

Plain text files are one document per file; JSON inputs support `text` plus optional `ground_truth` hits (see `--help` on the module).

### OOV and anomaly figures

1. Fit a model and retain the clustering report from training (see fit reporting / `clustering_quality` checkpoints in code).
2. Run **`pelinker-link-files`** with **`--dump-mention-anomaly`** to produce an OOV-oriented mention table.
3. Run **`run/analysis/oov_analysis.py`** with `--fit-report`, `--oov-csv`, and `--out-dir` to generate PDF figures (marginals, ROC/PR, decision boundary sweeps, alignment with the negative screener). The script docstring lists the full argument set.

## `run/` directory (scripts)

| Area | Contents |
|------|-----------|
| **Root** | `embed_kb_corpus.py`, `test_server.py`, `loop.embed.kb.corpus.sh`, `loop.fit.sh` |
| **`preprocessing/`** | GO / RO property extraction and merge → synthesis KB CSVs |
| **`analysis/`** | `clustering_quality.py` (embedding grid + metrics), `select_diverse_entities.py`, **`oov_analysis.py`** (fit report + OOV dump → figures), **`replot_dbcv_ari_scatter.py`** (PNG from an existing `results_grid_per_sample.csv`) |
| **`obsolete/`** | Deprecated experiments (not maintained) |

Always invoke scripts with **`uv run python …`** (see project rules) so the locked environment is used.

## See also

- **[Vector representations](vector_representation.md)** — encoder + spaCy window path (`texts_to_vrep`).
- **API Reference** — generated module pages (`pelinker.model`, `pelinker.analysis`, …).
