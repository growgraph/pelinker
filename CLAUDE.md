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
| KB | `pelinker/kb/` | Ground-truth spans + scoring, KB-out catalog and id bridge, class views (`classes.py`) |
| Screener | `pelinker/screener/` | Negative / manifold-OOV screening |
| Evaluation | `pelinker/eval/` | Gold harness, reference baselines, canonical KB view, LLM provider seam |
| Reports | `pelinker/reports/` | Report schemas, paths, summaries |
| Store | `pelinker/store/` | Packaged model resources |
| CLIs | `pelinker/cli/` | `pelinker-fit`, `pelinker-serves`, etc. |
| Scripts | `run/` | Preprocessing, analysis, `run/eval/` gold pipeline — see @run/README.md |

## Environment (uv only)

This project uses **uv** for all Python environment management. Do not use Poetry, bare
`python`/`python3`, `pip install`, or bare `pytest`.

```bash
# First-time / refresh env
uv sync --extra dev          # CI + local dev (pytest, pre-commit, en_core_web_lg)
uv sync --extra docs         # mkdocs build
uv sync --extra eval         # gold pipeline: LLM provider SDKs (run/eval/*)
uv sync --extra gpu          # optional CuPy
# extras are exclusive: name them together (--extra dev --extra eval) or the omitted
# ones are uninstalled

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

## Gold evaluation

`run/eval/` is the human-verified gold pipeline; `docs/user_guide/evaluation.md` is the
guide. Two invariants:

- Every step consumes the **pairs** KB (`data/derived/properties.synthesis.2.pairs.csv`,
  built by `run/preprocessing/derive_inverse_pairs.py`). A KB without `is_canonical` is
  refused, not tolerated — a fallback would change what the gold means.
- Ids are compared in canonical space on both sides (`pelinker.eval.kb.canonical_id_map`),
  so a converse-member answer scores as the relation it names.

Measured numbers belong in the measurement writeup, not in this repo: point
`--report-dir` outside the working tree.

## Gotchas

- `pelinker/__init__.py` imports `torch`/`triton` before TensorFlow to avoid a native
  segfault when ParametricUMAP loads both runtimes.
- Use `uv sync --extra dev`, not `--all-groups` (no `[dependency-groups]` in
  `pyproject.toml`).

## Conventions

Scoped rules live in `.claude/rules/` (Python style, tests, deps, docs, git workflow).
For pipeline/CLI details see @run/README.md.
