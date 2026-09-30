# PELinker (Property Entity LINKER) <img src="https://raw.githubusercontent.com/growgraph/pelinker/refs/heads/main/static/favicon.ico" alt="PELinkder Logo" style="height: 32px; width:32px;"/>

### Entity linking for BERT-like models

![Python](https://img.shields.io/badge/python-3.10.6%2B-blue.svg) 
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue?style=flat-square)](https://img.shields.io/badge/license-BSD--3--Clause-blue?style=flat-square)
[![pre-commit](https://github.com/growgraph/pelinker/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/growgraph/pelinker/actions/workflows/pre-commit.yml)

---

## Overview


Entity linking for BERT-like models


## Developer notes

1. Make sure there is an available version of python specified in `pyproject.toml`, for example installed using pyenv.
2. Install `uv` : `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Run `uv sync --extra dev` to create a local environment with project dependencies specified in `uv.lock`. The `dev` extra pins the spaCy `en_core_web_lg` model, so no separate `spacy download` is needed. Other extras: `docs` (MkDocs), `eval` (LLM SDKs for the gold pipeline), `preprocess` (ontology parsing), `gpu` (CuPy). Name every extra you want in one command — `uv sync` uninstalls the ones it is not given.
4. Set up `pre-commit` hooks:  `uv run pre-commit install`.
5. To run `pre-commit` independently from `git commit`, run `uv run pre-commit run --all-files`
6. To run tests run `uv run pytest test`


NB.
1. To run python scripts prefix the command with `uv run`, e.g. `uv run python script.py`
2. To git commit also `uv run` prefix, e.g. `uv run git commit -m "first commit"` to make sure `pre-commit` hooks are used from the correct python environement. 


## Testing against ground truth

Ground truth lives in `data/ground_truth` as `{"text": ..., "ground_truth": [{"itext",
"a", "b", "entity_id"}, ...]}`. `pelinker-link-files` scores against it automatically
whenever the input carries a `ground_truth` block:

```commandline
uv run pelinker-link-files -m models/pelinker.pubmedbert.run1 \
  -o reports/gt_score.json data/ground_truth/sample.0.gt.json
```

The output JSON gains a `ground_truth_score` block:

- **Detection** — `precision` / `recall` / `f1` over character spans, matched by overlap
  within a document. Unambiguous and comparable across model versions.
- **Entity accuracy** — over matched spans only, and only where the ids are comparable.
  Since the KB-out work the linker predicts *minted cluster ids* (`kb::C0007`) while the
  gold file carries *input KB* ids (`PEL.000032`), so `n_id_comparable` may be 0 and
  `entity_accuracy` `null`. That means undefined, not zero — read the detection numbers.

Add `--kb-validation` for a `kb_lemma_validation` block: the rate at which a mention's
predicted entity agrees with the entity its own lemma resolves to in the KB. That is a
distant-supervision consistency check, not end-task accuracy.

Programmatic entry points: `pelinker.kb.ground_truth.score_predictions_against_ground_truth`
and `pelinker.linker.kb_lemma.aggregate_kb_lemma_validation`.

## Gold evaluation

The scoring above compares against whatever ground truth a file happens to carry. For the
human-verified gold set — canonical predicate vocabulary, LLM pre-annotation, annotator
agreement, and reference baselines (lexical / encoder k-NN / LLM / the linker itself)
scored through one harness — see the [gold evaluation
guide](https://growgraph.github.io/pelinker/user_guide/evaluation/) and the drivers in
`run/eval/`.

## Fit a model

Train a linker on a corpus and serialize the artifact:

```commandline
uv run pelinker-fit \
  pipeline=both \
  kb_path=data/derived/properties.synthesis.2.csv \
  input_text_table_path=<corpus>.tsv.gz \
  embeddings_parquet=<workdir>/corpus_pubmedbert_1.parquet \
  model_path=<models>/pelinker.pubmedbert.run1 \
  report_path=<workdir>/reports/run1
```

See [`run/README.md`](run/README.md) for the stages, the parameters and how hyperparameters
reach the fit from the selection searches.

### Run server

- `uv run pelinker-serves` (FastAPI; default port 8599, `/docs` for the interactive API)

## Container
1. Build image: `docker buildx build -t gg/pelinker:<current_version> --ssh default=$SSH_AUTH_SOCK . 2>&1 | tee build.log`
2. Run container: `docker run --name pelinker --env THR_SCORE=0.5 gg/pelinker:latest`
