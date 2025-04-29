# Developer notes

1. Make sure you have the version of python specified in `pyproject.toml`.
   1. For example, using [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#automatic-installer) install `pyenv install 3.10.14`
   2. Then select local python version `pyenv local 3.10.14`
   3. Check using `which python` or `python --version`

2. Setting up poetry env 
   1. Install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
   2. Optionally set the virtual envs to be local `poetry config virtualenvs.in-project true`
   3. Inside project folder : `poetry env use 3.10.14`
   4. Install env with development dependencies: `poetry install --with dev`
   5. Add a spacy language model `python -m spacy download en_core_web_trf`
3. In the project folder before committing
   1. To lint run `pre-commit run --all-files`
   2. To run tests run `pytest test`


## Data Preparation
- `run/preprocessing/extract_properties_ro`
- `run/preprocessing/extract_properties_go`

### Merge properties/relations into a table

Uniformize and trim data incoming from different sources
 
- `run/preprocessing/merge_properties`

## Testing against ground truth

Ground truth dataset is stored in `data/ground_truth`, so run the following to obtain the accuracy of the model in `./reports` 

```commandline
python run/testing/run_pel_test.py --text-path ./data/ground_truth/sample.0.gt.json --model-type biobert-stsb --layers-spec sent --extra-context
```

## Serialize Model

"Train" a model on a corpus


- `poetry run python run/save_model.py`

### Run server

- `poetry run python run/serve`

## Container
1. Build image: `docker buildx build -t gg/pelinker:<current_version> --ssh default=$SSH_AUTH_SOCK . 2>&1 | tee build.log`
2. Run container: `docker run --name pelinker --env THR_SCORE=0.5 gg/pelinker:latest`



### Algo flow

```mermaid
flowchart TD
    A["[text]"] -->|"split_text_into_batches"| B["[[batched text]]"]
    subgraph S1["elementary tensors and word bounds"]
        direction LR
        C ~~~ D
    end

    B -->|"get_word_boundaries"| C["[[word group bounds]]"]
    B -->|"process_text"| D["token tensor"]
    subgraph S2["tensors ~ words of interest"]
        direction LR
        E["ll_tt_stacked"] ~~~ F["mapping_table"]
    end

    S1 -->|"render_elementary_tensor_table"| S2
```

[![](https://mermaid.ink/img/pako:eNqNUj1vwyAU_CuIORnS0UOlJmmndmk6NY4QNs82CgYEz0qjJP7tfdjOx9ChTPi4O-6eOfHSKeAZr4w7lI0MyL7WuWW0XrY53yL84C7nOzafP59zHr3RKBIotEUnCollAzHnZ7ZM9O0IKDboknD0il1RB-kbtlkQDQy0YFGGI_FsdCEyaRU7uKBY4Tqr4k2YltIBStTOsvfPO7pifd-zKStYldtxu5yi1oAiOYrBUQY9xlwNMYer6uA6P134kPVq4IMrIcahbVKuSYluD3YK_Ue5p0SZGvVDn8hcxWhUECDiP1q9koMxAlFElOUeVJp9KvpGB630XttaoCwM3Mweum8WU_ZAIARxH7QYY12lZ8rKZ7yF0Eqt6PefkgGFb0iQ84y2CirZGcqc2wtRZYduc7QlzzB0MOOdVxJhrSVVb3lWSRMJ9dJ-O3f_BqXRhY_xiQ0vbcZp6nUzMS6_rEDWAA?type=png)](https://mermaid.live/edit#pako:eNqNUj1vwyAU_CuIORnS0UOlJmmndmk6NY4QNs82CgYEz0qjJP7tfdjOx9ChTPi4O-6eOfHSKeAZr4w7lI0MyL7WuWW0XrY53yL84C7nOzafP59zHr3RKBIotEUnCollAzHnZ7ZM9O0IKDboknD0il1RB-kbtlkQDQy0YFGGI_FsdCEyaRU7uKBY4Tqr4k2YltIBStTOsvfPO7pifd-zKStYldtxu5yi1oAiOYrBUQY9xlwNMYer6uA6P134kPVq4IMrIcahbVKuSYluD3YK_Ue5p0SZGvVDn8hcxWhUECDiP1q9koMxAlFElOUeVJp9KvpGB630XttaoCwM3Mweum8WU_ZAIARxH7QYY12lZ8rKZ7yF0Eqt6PefkgGFb0iQ84y2CirZGcqc2wtRZYduc7QlzzB0MOOdVxJhrSVVb3lWSRMJ9dJ-O3f_BqXRhY_xiQ0vbcZp6nUzMS6_rEDWAA)


## Analysis

An essential part of analysis is to identify patterns in text and study their embeddings vectors.

To run pattern matching over different models and patterns, and plot them to `figs` folder, where the texts are taken from a csv file with a column named `abstract`:
```shell
cd run
./test.pat.align.sh ./test.pat.align.sh --pattern pat_a --pattern pat_b --plot-path figs --input-path data/test/sample.csv.gz
```
