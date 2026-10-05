# Run scripts and CLIs

This page is the **high-level map** for training, serving, batch linking, and offline analysis. Detailed flags, tables, and preprocessing steps live in the repository’s [`run/README.md`](https://github.com/growgraph/pelinker/blob/main/run/README.md) (kept next to the scripts).

## Environment

From the repository root, use **`uv`** so dependencies match `uv.lock`:

```bash
uv sync --extra dev
```

The `dev` extra pins the spaCy `en_core_web_lg` model as a dependency, so a separate
`spacy download` is unnecessary — and would not survive the next `uv sync`, which prunes
anything not declared. These are **extras**, not dependency groups: name every one you
want in a single command (`uv sync --extra dev --extra eval`), because the ones you leave
out are uninstalled.

Documentation site builds (optional):

```bash
uv sync --extra docs
uv run mkdocs serve
```

## Packaged commands

| Command | Module | Role |
|--------|--------|------|
| `uv run pelinker-fit` | `pelinker.cli.fit` | Corpus embedding (optional) + `Linker.fit` → serialized artifact (`.gz`). Hydra overrides; defaults in `pelinker/conf/fit.yaml`. Mention load flags (`drop_rare_entities`, `max_mentions_per_entity`, `clustering_sample_rows`) align with model selection. |
| `uv run pelinker-serves` | `pelinker.cli.server` | FastAPI server: `/health`, `/info`, `/model`, `/link`, `/link/debug`. Defaults in `pelinker/conf/server.yaml`. |
| `uv run pelinker-link-files` | `pelinker.cli.link_files` | Batch `Linker.predict` on UTF-8 files or JSON documents; optional JSON report and **mention-level anomaly dump** for OOV workflows. |
| `uv run pelinker-model-selection` | `pelinker.cli.model_selection` | Grid search over embedding backbone × layers on a directory of mention parquets; writes the run report and heatmaps. Like the other searches and `pelinker-fit`, it scores ARI in a *class view* of the KB — `--class-view` (default `reldir`: canonical relation + direction) with `--class-kb-path` pointing at the pairs KB; `raw` scores against the matched labels. See `run/README.md` § Class views. |
| `uv run pelinker-dim-selection` | `pelinker.cli.dim_selection` | `(pca_components, umap_dim)` search on one parquet, same metric stack. `--persist-labels` also writes the per-draw labels that cluster-stability analysis needs. |
| `uv run pelinker-scale-curve` | `pelinker.cli.scale_curve` | Measures how the chosen `min_cluster_size` moves with mention-frame size; writes `scale_curve.json` for `pelinker-fit scale_curve_path=`. |
| `uv run pelinker-replot` | `pelinker.cli.replot` | Regenerates model-selection figures from an existing report directory — no re-embedding, no re-clustering. |

Each has an equivalent module invocation, e.g. `uv run python -m pelinker.cli.fit`.

The hyperparameters chosen by the searches reach a fit through `selection_report=` and
`scale_curve_path=` rather than being retyped; `pelinker-fit` records which source won
under `min_cluster_size_provenance`. Note also that `pelinker-fit` defaults to
`pipeline=embed_only` — pass `pipeline=both` for an end-to-end train.

### Batch linking (`pelinker-link-files`)

- **Model**: `-m` / `--model` — path to the linker dump (`Linker.load` rules; `.gz` is resolved like elsewhere).
- **Threshold**: `--thr-score` — same idea as the server’s score threshold.
- **Outputs**: `-o` / `--output` — full prediction JSON; `--dump-mention-anomaly PATH` — per-mention rows with PCA residual / Mahalanobis-style metrics (extension selects **`.parquet`**, **`.csv`**, or **`.jsonl`**).
- **Extras**: `--include-anomaly-metrics` and `--kb-validation` mirror server/debug style fields on entities; `--use-gpu` for CUDA when available.

Plain text files are one document per file; JSON inputs support `text` plus optional `ground_truth` hits (see `--help` on the module).

### OOV and anomaly figures

1. Fit a model and retain the clustering report from training (see fit reporting / `model_selection` checkpoints in code).
2. Run **`pelinker-link-files`** with **`--dump-mention-anomaly`** to produce an OOV-oriented mention table.
3. Run **`run/analysis/oov_analysis.py`** with `--fit-report`, `--oov-csv`, and `--out-dir` to generate PDF figures (marginals, ROC/PR, decision boundary sweeps, alignment with the negative screener). The script docstring lists the full argument set.

## `run/` directory (scripts)

| Area | Contents |
|------|-----------|
| **Root** | `embed_kb_corpus.py` (standalone stage A), `smoke_server.py` (HTTP smoke tests), `loop.embed.kb.corpus.sh`, `loop.fit.sh` |
| **`preprocessing/`** | GO / RO property extraction and merge → synthesis KB CSVs, then `derive_inverse_pairs.py` → the canonical pairs KB |
| **`analysis/`** | `compact_predict_study.py` (legacy vs compact predict arms), `cluster_stability.py` (cluster identity across draws), `direction_diagnostic.py` (converse-pair collisions), `oov_analysis.py` (fit report + OOV dump → figures), `replot_fit.py` (figures from a fit report), `select_diverse_entities.py` |
| **`eval/`** | The gold pipeline: sampling, LLM pre-annotation, review round trip with κ, reference baselines — see [Gold evaluation](evaluation.md) |

Always invoke scripts with **`uv run python …`** (see project rules) so the locked environment is used.

## See also

- **[Vector representations](vector_representation.md)** — encoder + spaCy window path (`texts_to_vrep`).
- **[Gold evaluation](evaluation.md)** — the human-verified gold set and the reference baselines.
- **API Reference** — generated module pages (`pelinker.model`, `pelinker.search`, …).
