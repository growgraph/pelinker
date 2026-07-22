"""Structured checkpoint I/O for ``pelinker.dim_selection`` runs."""

from __future__ import annotations

import gzip
import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from pelinker.io.json_files import is_gzip_file_path, load_json_path
from pelinker.onto import NEGATIVE_LABEL

CHECKPOINT_VERSION = 1
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
        if int(data.get("version", -1)) != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version: {data.get('version')!r}; "
                f"expected {CHECKPOINT_VERSION}"
            )
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_run_fingerprint(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(blob).hexdigest()


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
    kb = None
    if selected_labels_kb_path is not None:
        kb = str(selected_labels_kb_path.expanduser().resolve())
    resolved_min_scale = (
        min_scale if min_scale is not None else max(1, min_class_size // 2)
    )
    return {
        "batch_size": batch_size,
        "clustering_grid_step": clustering_grid_step,
        "clustering_sample_rows": clustering_sample_rows,
        "cluster_viz_method": cluster_viz_method,
        "drop_rare_entities": drop_rare_entities,
        "input_parquet": str(input_parquet.expanduser().resolve()),
        "layer": layer,
        "max_mentions_negative": max_mentions_negative,
        "max_mentions_per_entity": max_mentions_per_entity,
        "max_scale": max_scale,
        "mention_cap_seed": mention_cap_seed,
        "min_class_size": min_class_size,
        "min_mentions_per_entity": min_mentions_per_entity,
        "min_scale": resolved_min_scale,
        "model": model,
        "n_sample": n_sample,
        "negative_label": negative_label,
        "pca_grid": list(pca_grid),
        "pca_seed": pca_seed,
        "refine": refine,
        "screener_kind": screener_kind,
        "seed": seed,
        "selected_labels_kb_path": kb,
        "umap_grid": list(umap_grid),
        "umap_seed": umap_seed,
    }


def load_checkpoint(path: pathlib.Path) -> DimSelectionCheckpoint:
    data = load_json_path(path)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be a JSON object")
    return DimSelectionCheckpoint.from_json_dict(data)


def save_checkpoint_atomic(
    path: pathlib.Path, checkpoint: DimSelectionCheckpoint
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.updated_at = utc_now_iso()
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(
        checkpoint.to_json_dict(), indent=2, sort_keys=True, ensure_ascii=False
    )
    text = payload + "\n"
    if is_gzip_file_path(path):
        with gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as gz:
            gz.write(text)
    else:
        tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


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
