# Developer notes

Within env `poetry install`.
Add spacy LMs `python -m spacy download en_core_web_trf`.

## Data Preparation
- `run/preprocessing/extract_properties_ro`
- `run/preprocessing/extract_properties_go`

### Merge properties/relations into a table

Uniformize and trim data incoming from different sources
 
- `run/preprocessing/merge_properties`


## Serialize Model serialization

"Train" a model on a corpus


- `poetry run python run/save_model.py`

### Run server

- `poetry run python run/serve`

## Container
1. Build image: `docker buildx build -t gg/pelinker:<current_version> --ssh default=$SSH_AUTH_SOCK . 2>&1 | tee build.log`
2. Run container: `docker run --name pelinker --env THR_SCORE=0.5 gg/pelinker:latest`