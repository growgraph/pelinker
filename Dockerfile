FROM python:3.10-slim-bullseye AS builder

RUN apt update -y && apt upgrade -y && apt install curl git -y
RUN curl -sSL https://install.python-poetry.org | python - --version 1.8.3

ENV PATH="${PATH}:/root/.local/bin"

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_EXPERIMENTAL_SYSTEM_GIT_CLIENT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

COPY README.md logging.conf ./
COPY pelinker ./pelinker
COPY run ./run

RUN poetry install --no-interaction -vvv --without dev
RUN poetry run python -m spacy download en_core_web_sm

RUN rm -rf /tmp/poetry_cache ./.venv/lib/python3.10/site-packages/nvidi*

CMD ["poetry", "run", "python", "run/serve.py", "--model-type", "biobert-stsb", "--port", "8599", "--thr-score", "0.5", "--thr-dif", "0.025"]

