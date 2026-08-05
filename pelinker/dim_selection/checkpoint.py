"""Structured checkpoint I/O for ``pelinker.dim_selection`` runs.

Shared machinery (version, fingerprint, atomic gzip-aware I/O, the fingerprint fields
common to every search) lives in :mod:`pelinker.checkpoint`; only the cell-keyed
schema is here.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

from pelinker.checkpoint import (
    CHECKPOINT_VERSION,
    compute_run_fingerprint,
    read_checkpoint_json,
    require_checkpoint_version,
    shared_fingerprint_fields,
    utc_now_iso,
    write_checkpoint_json_atomic,
)
from pelinker.onto import NEGATIVE_LABEL

__all__ = [
    "CHECKPOINT_VERSION",
    "DEFAULT_CHECKPOINT_NAME",
    "DimSelectionCheckpoint",
    "FailureRecord",
    "compute_run_fingerprint",
    "fingerprint_config_from_cli",
    "load_checkpoint",
    "mark_cell_done",
    "new_checkpoint",
    "record_failure",
    "save_checkpoint_atomic",
    "utc_now_iso",
]

DEFAULT_CHECKPOINT_NAME = "dim_selection.state.json.gz"


@dataclass
class FailureRecord:
    """One recorded failure for a (pca, umap) cell."""

    cell_key: str
    error: str
    at: str


@dataclass
class DimSelectionCheckpoint:
    """On-disk checkpoint for resumable PCA/UMAP dimension selection runs."""

    version: int = CHECKPOINT_VERSION
    run_fingerprint: str = ""
    created_at: str = ""
    updated_at: str = ""
    completed_cells: list[str] = field(default_factory=list)
    summaries_by_key: dict[str, dict[str, str | float | int | None]] = field(
        default_factory=dict
    )
    stages: dict[str, str] = field(default_factory=dict)
    failures: list[FailureRecord] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_fingerprint": self.run_fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_cells": sorted(self.completed_cells),
            "summaries_by_key": {
                k: dict(v) for k, v in sorted(self.summaries_by_key.items())
            },
            "stages": dict(sorted(self.stages.items())),
            "failures": [
                {"cell_key": f.cell_key, "error": f.error, "at": f.at}
                for f in self.failures
            ],
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> DimSelectionCheckpoint:
        require_checkpoint_version(data)
        failures_raw = data.get("failures") or []
        failures = [
            FailureRecord(
                cell_key=str(fr["cell_key"]),
                error=str(fr["error"]),
                at=str(fr["at"]),
            )
            for fr in failures_raw
        ]
        return DimSelectionCheckpoint(
            version=int(data["version"]),
            run_fingerprint=str(data["run_fingerprint"]),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            completed_cells=list(data.get("completed_cells") or []),
            summaries_by_key=dict(data.get("summaries_by_key") or {}),
            stages=dict(data.get("stages") or {}),
            failures=failures,
        )


def fingerprint_config_from_cli(
    *,
    input_parquet: pathlib.Path,
    model: str,
    layer: str,
    pca_grid: tuple[int, ...],
    umap_grid: tuple[int, ...],
    refine: bool,
    cluster_viz_method: str,
    min_class_size: int,
    seed: int,
    pca_seed: int,
    umap_seed: int | None,
    clustering_sample_rows: int | None,
    batch_size: int,
    n_sample: int,
    selected_labels_kb_path: pathlib.Path | None,
    max_scale: int,
    min_scale: int | None = None,
    clustering_grid_step: int = 5,
    negative_label: str = NEGATIVE_LABEL,
    screener_kind: str = "lda",
    drop_rare_entities: bool = False,
    min_mentions_per_entity: int = 20,
    max_mentions_per_entity: int | None = None,
    max_mentions_negative: int | None = None,
    mention_cap_seed: int = 13,
) -> dict[str, Any]:
    return {
        **shared_fingerprint_fields(
            cluster_viz_method=cluster_viz_method,
            min_class_size=min_class_size,
            seed=seed,
            pca_seed=pca_seed,
            umap_seed=umap_seed,
            clustering_sample_rows=clustering_sample_rows,
            batch_size=batch_size,
            n_sample=n_sample,
            selected_labels_kb_path=selected_labels_kb_path,
            max_scale=max_scale,
            min_scale=min_scale,
            clustering_grid_step=clustering_grid_step,
            negative_label=negative_label,
            screener_kind=screener_kind,
            drop_rare_entities=drop_rare_entities,
            min_mentions_per_entity=min_mentions_per_entity,
            max_mentions_per_entity=max_mentions_per_entity,
            max_mentions_negative=max_mentions_negative,
            mention_cap_seed=mention_cap_seed,
        ),
        "input_parquet": str(input_parquet.expanduser().resolve()),
        "layer": layer,
        "model": model,
        "pca_grid": list(pca_grid),
        "refine": refine,
        "umap_grid": list(umap_grid),
    }


def load_checkpoint(path: pathlib.Path) -> DimSelectionCheckpoint:
    return DimSelectionCheckpoint.from_json_dict(read_checkpoint_json(path))


def save_checkpoint_atomic(
    path: pathlib.Path, checkpoint: DimSelectionCheckpoint
) -> None:
    checkpoint.updated_at = utc_now_iso()
    write_checkpoint_json_atomic(path, checkpoint.to_json_dict())


def new_checkpoint(fingerprint: str) -> DimSelectionCheckpoint:
    now = utc_now_iso()
    return DimSelectionCheckpoint(
        version=CHECKPOINT_VERSION,
        run_fingerprint=fingerprint,
        created_at=now,
        updated_at=now,
        stages={"coarse": "pending", "refine": "pending"},
    )


def mark_cell_done(
    ckpt: DimSelectionCheckpoint,
    ckpt_path: pathlib.Path,
    *,
    cell_key: str,
    summary_flat: dict[str, str | float | int | None],
) -> None:
    if cell_key not in ckpt.completed_cells:
        ckpt.completed_cells.append(cell_key)
    ckpt.summaries_by_key[cell_key] = dict(summary_flat)
    save_checkpoint_atomic(ckpt_path, ckpt)


def record_failure(
    ckpt: DimSelectionCheckpoint,
    ckpt_path: pathlib.Path,
    *,
    cell_key: str,
    message: str,
) -> None:
    ckpt.failures.append(
        FailureRecord(cell_key=cell_key, error=message, at=utc_now_iso())
    )
    save_checkpoint_atomic(ckpt_path, ckpt)
