# Pelinker

**Pelinker** is a bio / science property–entity linker. This site hosts technical notes and the Python API reference.

## Documentation map

- **[Vector representations](user_guide/vector_representation.md)** — how `texts_to_vrep` turns text and transformer layers into pooled embeddings for sliding token windows.
- **API Reference** — auto-generated from package docstrings (`pelinker`).

## Development

Build the docs locally (from the repository root, with optional dependencies installed):

```bash
uv sync --extra docs
uv run mkdocs serve
```
