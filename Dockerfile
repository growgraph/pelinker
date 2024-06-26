FROM python:3.10-slim-bullseye AS builder

RUN apt update -y && apt upgrade -y && apt install curl git -y
RUN curl -sSL https://install.python-poetry.org | python - --version 1.7.1

ENV PATH="${PATH}:/root/.local/bin"

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_EXPERIMENTAL_SYSTEM_GIT_CLIENT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

COPY pelinker ./pelinker

RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh poetry install --no-interaction -vvv --without dev
COPY run ./run
COPY README.md logging.conf logging.debug.conf ./

CMD ["poetry", "run", "python", "run/serve.py", "--model-type", "biobert-stsb"]
