# PELinker (Property Entity LINKER)

Entity linking for BERT-like models


## Developer notes

1. Make sure there is an available version of python specified in `pyproject.toml`, for example installed using pyenv.
2. Install `uv` : `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Run `uv sync --all-groups` to create a local environment with project dependencies specified in `uv.lock`
4. Add a spacy language model `uv run spacy download en_core_web_trf`
5. Set up `pre-commit` hooks:  `uv run pre-commit install`.
6. To run `pre-commit` independently from `git commit`, run `uv run pre-commit run --all-files`
7. To run tests run `pytest test`


NB.
1. To run python scripts prefix the command with `uv run`, e.g. `uv run python script.py`
2. To git commit also `uv run` prefix, e.g. `uv run git commit -m "first commit"` to make sure `pre-commit` hooks are used from the correct python environement. 


## Testing against ground truth

Ground truth dataset is stored in `data/ground_truth`, so run the following to obtain the accuracy of the model in `./reports` 

```commandline
python run/testing/run_pel_test.py --text-path ./data/ground_truth/sample.0.gt.json --model-type biobert-stsb --layers-spec sent --extra-context
```

## Serialize Model

"Train" a model on a corpus


- `uv run python run/save_model.py`

### Run server

- `poetry run python run/serve`

## Container
1. Build image: `docker buildx build -t gg/pelinker:<current_version> --ssh default=$SSH_AUTH_SOCK . 2>&1 | tee build.log`
2. Run container: `docker run --name pelinker --env THR_SCORE=0.5 gg/pelinker:latest`
