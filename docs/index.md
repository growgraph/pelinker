# PELinker

**PELinker** is a bio / science property–entity linker: encoder-based mention vectors, fused property clusters, and a small HTTP API for linking text to KB entities. This site hosts user-oriented notes and the Python API reference.

The current **package version** is defined in `pyproject.toml` (see the `[project]` `version` field). Release artifacts are the **`pelinker`** wheel/sdist and the documented CLIs below.

## Documentation map

- **[Run scripts & CLIs](user_guide/run_scripts_and_cli.md)** — `pelinker-fit`, `pelinker-serves`, `pelinker-link-files`, and how they connect to scripts under `run/` (including OOV / anomaly analysis).
- **[Vector representations](user_guide/vector_representation.md)** — how `texts_to_vrep` turns text and transformer layers into pooled embeddings for sliding token windows.
- **API Reference** — auto-generated from package docstrings (`pelinker`).
- **Repository run guide** — long-form tables and preprocessing live in [`run/README.md`](https://github.com/growgraph/pelinker/blob/main/run/README.md) on GitHub.

## Development

Build the docs locally (from the repository root, with optional dependencies installed):

```bash
uv sync --extra docs
uv run mkdocs serve
```

Run the test suite with the same environment:

```bash
uv run pytest test
```
