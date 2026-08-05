"""Machine-readable handoff from the search stages to ``pelinker-fit``.

Model selection picks a ``(model, layer)`` and a pooled ``min_cluster_size``; dimension
selection picks ``(pca_components, umap_dim)``. Both wrote their winners only into
human-readable reports, so the values reached the fit by being retyped into Hydra
overrides. That is how a real run could choose ``pca=22, umap_dim=3`` while
``pelinker-fit`` went on defaulting to ``100`` and ``8``.

This module defines the small typed record both searches emit and the fit consumes, so
the transfer is checked rather than remembered. The realized row count travels with it —
``min_cluster_size`` is an absolute count and means nothing without the N it was chosen
at (see :mod:`pelinker.scaling`).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

SELECTED_HYPERPARAMETERS_BASENAME = "selected_hyperparameters.json"
SELECTED_HYPERPARAMETERS_SCHEMA = "pelinker.selected_hyperparameters.v1"


@dataclass(frozen=True)
class SelectedHyperparameters:
    """What a search stage chose, and the conditions it chose them under."""

    source: str
    """Which stage wrote this (``model_selection`` / ``dim_selection``)."""
    model: str
    layer: str
    pca_components: int
    umap_dim: int
    min_cluster_size: int
    manifold_kind: str
    """Coordinates the search ran on. ``pelinker-fit`` refuses a fit whose
    ``predict_mode`` implies a different manifold, rather than silently transferring
    ``min_cluster_size`` across a coordinate-system change."""
    n_rows_realized: int | None = None
    """Mention rows the winning configuration was actually scored on."""
    umap_n_neighbors: int | None = None
    run_fingerprint: str | None = None
    outer_score: float | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be a non-empty string")
        if self.pca_components < 1:
            raise ValueError("pca_components must be >= 1")
        if self.umap_dim < 2:
            raise ValueError("umap_dim must be >= 2")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if self.manifold_kind not in ("umap", "parametric"):
            raise ValueError(
                f"manifold_kind must be 'umap' or 'parametric', got {self.manifold_kind!r}"
            )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": SELECTED_HYPERPARAMETERS_SCHEMA,
            "source": self.source,
            "model": self.model,
            "layer": self.layer,
            "pca_components": int(self.pca_components),
            "umap_dim": int(self.umap_dim),
            "min_cluster_size": int(self.min_cluster_size),
            "manifold_kind": self.manifold_kind,
            "n_rows_realized": (
                None if self.n_rows_realized is None else int(self.n_rows_realized)
            ),
            "umap_n_neighbors": (
                None if self.umap_n_neighbors is None else int(self.umap_n_neighbors)
            ),
            "run_fingerprint": self.run_fingerprint,
            "outer_score": (
                None if self.outer_score is None else float(self.outer_score)
            ),
        }

    @staticmethod
    def from_jsonable(data: dict[str, Any]) -> SelectedHyperparameters:
        schema = data.get("schema")
        if schema != SELECTED_HYPERPARAMETERS_SCHEMA:
            raise ValueError(
                f"expected schema {SELECTED_HYPERPARAMETERS_SCHEMA!r}, got {schema!r}"
            )
        return SelectedHyperparameters(
            source=str(data.get("source", "")),
            model=str(data["model"]),
            layer=str(data["layer"]),
            pca_components=int(data["pca_components"]),
            umap_dim=int(data["umap_dim"]),
            min_cluster_size=int(data["min_cluster_size"]),
            manifold_kind=str(data["manifold_kind"]),
            n_rows_realized=(
                None
                if data.get("n_rows_realized") is None
                else int(data["n_rows_realized"])
            ),
            umap_n_neighbors=(
                None
                if data.get("umap_n_neighbors") is None
                else int(data["umap_n_neighbors"])
            ),
            run_fingerprint=data.get("run_fingerprint"),
            outer_score=(
                None if data.get("outer_score") is None else float(data["outer_score"])
            ),
        )


def write_selected_hyperparameters(
    selected: SelectedHyperparameters, report_path: pathlib.Path
) -> pathlib.Path:
    """Write ``selected_hyperparameters.json`` under ``report_path`` (atomic)."""
    report_path = pathlib.Path(report_path).expanduser()
    report_path.mkdir(parents=True, exist_ok=True)
    out = report_path / SELECTED_HYPERPARAMETERS_BASENAME
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(selected.to_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(out)
    return out


def load_selected_hyperparameters(
    path: pathlib.Path | str,
) -> SelectedHyperparameters:
    """Read a handoff file, or the file of that name inside a report directory."""
    p = pathlib.Path(path).expanduser()
    if p.is_dir():
        p = p / SELECTED_HYPERPARAMETERS_BASENAME
    return SelectedHyperparameters.from_jsonable(
        json.loads(p.read_text(encoding="utf-8"))
    )
