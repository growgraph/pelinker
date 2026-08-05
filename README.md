# PELinker (Property Entity LINKER) <img src="https://raw.githubusercontent.com/growgraph/pelinker/refs/heads/main/static/favicon.ico" alt="PELinkder Logo" style="height: 32px; width:32px;"/>

### Entity linking for BERT-like models

![Python](https://img.shields.io/badge/python-3.10-blue.svg) 
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue?style=flat-square)](https://img.shields.io/badge/license-BSD--3--Clause-blue?style=flat-square)
[![pre-commit](https://github.com/growgraph/pelinker/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/growgraph/pelinker/actions/workflows/pre-commit.yml)

---

## Overview


Entity linking for BERT-like models


## Developer notes

1. Make sure there is an available version of python specified in `pyproject.toml`, for example installed using pyenv.
2. Install `uv` : `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Run `uv sync --all-groups` to create a local environment with project dependencies specified in `uv.lock`
4. Add a spacy language model `uv run spacy download en_core_web_lg`
5. Set up `pre-commit` hooks:  `uv run pre-commit install`.
6. To run `pre-commit` independently from `git commit`, run `uv run pre-commit run --all-files`
7. To run tests run `pytest test`


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

Programmatic entry points: `pelinker.ground_truth.score_predictions_against_ground_truth`
and `pelinker.linker_kb_lemma.aggregate_kb_lemma_validation`.

## Serialize Model

"Train" a model on a corpus


- `uv run python run/save_model.py`

### Run server

- `poetry run python run/serve`

## Container
1. Build image: `docker buildx build -t gg/pelinker:<current_version> --ssh default=$SSH_AUTH_SOCK . 2>&1 | tee build.log`
2. Run container: `docker run --name pelinker --env THR_SCORE=0.5 gg/pelinker:latest`
