# PELinker — Claude Code Instructions

Property/entity linker for BERT-like models. The pipeline embeds corpus mentions, clusters
them (PCA → UMAP → HDBSCAN), and serializes a `Linker` artifact. Batch linking and a
FastAPI server expose inference.

## Package layout

| Area | Path | Role |
|------|------|------|
| Config / paths | `pelinker/core/` | Hydra-facing config, KB metadata, scaling |
| Text / embeddings | `pelinker/text/`, `pelinker/embed/` | Tokenization, transformer loading, corpus embedding |
| Data | `pelinker/data/` | Parquet/JSON frames, fusion |
| Clustering | `pelinker/clustering/` | Fit pipeline, grid metrics, transforms |
| Hyperparam search | `pelinker/search/` | Model/dim selection, scale curve, grid solver |
| Linker training | `pelinker/linker/` | Cluster training, distillation, KB lemma |
| CLIs | `pelinker/cli/` | `pelinker-fit`, `pelinker-serves`, etc. |
| Scripts | `run/` | Preprocessing, analysis — see @run/README.md |

## Environment (uv only)

This project uses **uv** for all Python environment management. Do not use Poetry, bare
`python`/`python3`, `pip install`, or bare `pytest`.

```bash
# First-time / refresh env
uv sync --extra dev          # CI + local dev (pytest, pre-commit, en_core_web_lg)
uv sync --extra docs         # mkdocs build
uv sync --extra gpu          # optional CuPy

# Run code
uv run python run/script.py
uv run pelinker-fit ...
uv run pelinker-serves
uv run pytest test
uv run pre-commit run --all-files
uv run pre-commit install

# After dependency edits
uv lock && uv sync --extra dev && uv pip check
```

Prefix git commits with `uv run` so pre-commit hooks use the correct environment, e.g.
`uv run git commit -m "message"`.

## Linting and formatting

Pre-commit runs Ruff check + format (see `.pre-commit-config.yaml`). Ruff config in
`pyproject.toml` selects `E4`, `E7`, `E9`, `F` and ignores `E722`.

## Testing

- Run: `uv run pytest test`
- Tests marked `heavy` skip when spaCy `en_core_web_lg` or HF weights are absent.
- The `dev` extra pins `en_core_web_lg` so CI runs tokenization tests without a separate
  `spacy download`.

## Gotchas

- `pelinker/__init__.py` imports `torch`/`triton` before TensorFlow to avoid a native
  segfault when ParametricUMAP loads both runtimes.
- Use `uv sync --extra dev`, not `--all-groups` (no `[dependency-groups]` in
  `pyproject.toml`).

## Conventions

Scoped rules live in `.claude/rules/` (Python style, tests, deps, docs, git workflow).
For pipeline/CLI details see @run/README.md.
