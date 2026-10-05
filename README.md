# PELinker <img src="https://raw.githubusercontent.com/growgraph/pelinker/refs/heads/main/static/favicon.ico" alt="PELinker logo" style="height: 32px; width:32px;"/>

**Property Entity Linker** — links relation mentions in scientific text to the properties
of a knowledge base, using BERT-like encoders.

![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue?style=flat-square)](LICENSE)
[![pre-commit](https://github.com/growgraph/pelinker/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/growgraph/pelinker/actions/workflows/pre-commit.yml)
[![Docs](https://img.shields.io/badge/docs-growgraph.github.io-blue)](https://growgraph.github.io/pelinker/)

## How it works

1. **Embed** — find KB property mentions in a corpus (lemma windows, plus voice-aware
   matching for verb predicates) and pool encoder activations into one vector per mention.
2. **Fit** — reduce the vectors (PCA → UMAP), cluster them (HDBSCAN), and serialize the
   result as a `Linker` artifact.
3. **Link** — embed new text the same way and assign each mention to a cluster, with a
   score and a predicted direction.

The KB is a CSV of properties (`entity_id`, `label`, …) with converse pairs linked, so a
passive mention resolves to the relation it names. [`data/README.md`](data/README.md)
describes which KB files are curated and which are derived.

## Install

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync --extra dev
```

`dev` includes pytest, pre-commit and the spaCy `en_core_web_lg` model. Other extras:

| Extra | Adds |
|-------|------|
| `docs` | MkDocs site |
| `eval` | LLM provider SDKs for the gold pipeline (`run/eval/`) |
| `preprocess` | ontology parsing for KB regeneration |
| `gpu-cu13` / `gpu-cu12` | CUDA torch + CuPy; mutually exclusive |

`uv sync` uninstalls every extra it is not given, so name them all in one command, e.g.
`uv sync --extra dev --extra gpu-cu13`.

**GPU.** Pick the extra by the "CUDA Version" that `nvidia-smi` reports — the driver's
limit; an installed CUDA toolkit is not used: 13.x → `gpu-cu13`, 12.x → `gpu-cu12`. Check:

```bash
uv run python -c "import torch, cupy, cupy.cublas; print(torch.cuda.is_available(), cupy.cuda.runtime.getDeviceCount())"
```

## Usage

Prefix every command with `uv run`.

### Fit a linker

```bash
uv run pelinker-fit \
  pipeline=both \
  kb_path=data/derived/properties.synthesis.2.pairs.csv \
  input_text_table_path=<corpus>.tsv.gz \
  embeddings_parquet=<workdir>/corpus_pubmedbert_1.parquet \
  model_path=<models>/pelinker.pubmedbert.run1 \
  report_path=<workdir>/reports/run1
```

The corpus is a TSV/CSV of `pmid` and `text` (optionally gzipped). `pipeline` defaults to
`embed_only`; `both` embeds and fits. Stages, parameters, and how the hyperparameter
searches feed the fit: [`run/README.md`](run/README.md).

### Link files

```bash
uv run pelinker-link-files -m <models>/pelinker.pubmedbert.run1 -o out.json input.txt
```

Inputs are plain text (one document per file) or JSON with a `text` field.

### Serve

```bash
uv run pelinker-serves model_file_spec=<models>/pelinker.pubmedbert.run1
```

FastAPI on port 8599; interactive docs at `/docs`. Without `model_file_spec` the server
loads the packaged model. Routes and request fields: [HTTP API](docs/user_guide/api.md).

## Evaluation

**Against a file's own ground truth.** When an input carries a `ground_truth` block
(`{"text": ..., "ground_truth": [{"itext", "a", "b", "entity_id"}, ...]}`, see
`data/ground_truth/`), `pelinker-link-files` adds a `ground_truth_score` block:

- **Detection** — precision / recall / F1 over character spans, matched by overlap within
  a document.
- **Entity accuracy** — over matched spans whose ids are comparable. The linker predicts
  minted cluster ids (`kb::C0007`) while gold files carry input KB ids (`PEL.000032`), so
  `entity_accuracy` may be `null`: undefined, not zero.

`--kb-validation` adds `kb_lemma_validation`: how often a mention's predicted entity agrees
with the entity its own lemma resolves to in the KB — a consistency check, not end-task
accuracy.

**Human-verified gold set.** Canonical predicate vocabulary, LLM pre-annotation, annotator
agreement, and reference baselines scored through one harness: see the
[gold evaluation guide](https://growgraph.github.io/pelinker/user_guide/evaluation/) and
`run/eval/`.

## Development

```bash
uv run pre-commit install            # ruff check + format on commit
uv run pre-commit run --all-files
uv run pytest test                   # `heavy` tests skip when model weights are absent
uv sync --extra docs && uv run mkdocs serve
```

Commit with `uv run git commit …` so the hooks run in the project environment.
