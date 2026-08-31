"""Shared machinery for resumable search checkpoints.

``model_selection``, ``dim_selection`` and ``scale_curve`` each keep their own
checkpoint dataclass, because their keys genuinely differ (a fused ``combination``, a
``(pca, umap)`` cell, a sample-size rung) and their on-disk field names must stay
stable for already-written state files. Everything that does *not* depend on those
names lives here: the version constant, the timestamp, the run fingerprint, the
atomic gzip-aware read/write, and the fingerprint fields all three searches share.
"""

from __future__ import annotations

import gzip
import json
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from pelinker.data.json_files import is_gzip_file_path, load_json_path
from pelinker.core.onto import NEGATIVE_LABEL

CHECKPOINT_VERSION = 2
"""Bumped to 2: outer DBCV+ARI scores switched from ``0.5*(dbcv+ari)`` to the clipped
geometric mean. ``singleton_scores_by_key`` persists those values, so resuming a v1
checkpoint would compare old-scale and new-scale scores against each other."""


@dataclass
class FailureRecord:
    """One recorded failure, keyed by whatever the search iterates over."""

    key: str
    error: str
    at: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_run_fingerprint(config: dict[str, Any]) -> str:
    """sha256 of the canonicalized run config; a change invalidates the checkpoint."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(blob).hexdigest()


def require_checkpoint_version(data: dict[str, Any]) -> None:
    if int(data.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {data.get('version')!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )


def read_checkpoint_json(path: pathlib.Path) -> dict[str, Any]:
    """Load a checkpoint payload (plain or gzipped) and assert it is an object."""
    data = load_json_path(path)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be a JSON object")
    return data


def write_checkpoint_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write via ``*.tmp`` + ``replace`` so a crash never leaves a torn checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if is_gzip_file_path(path):
        with gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as gz:
            gz.write(text)
    else:
        tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def shared_fingerprint_fields(
    *,
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
    """The fingerprint fields common to every search; callers add their own on top.

    ``min_scale`` is resolved here to the same default
    :meth:`~pelinker.core.config.ClusteringOptimizationConfig.resolved_min_scale` uses, so
    an explicit ``min_scale`` equal to the default does not change the fingerprint.
    """
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
        "max_mentions_negative": max_mentions_negative,
        "max_mentions_per_entity": max_mentions_per_entity,
        "max_scale": max_scale,
        "mention_cap_seed": mention_cap_seed,
        "min_class_size": min_class_size,
        "min_mentions_per_entity": min_mentions_per_entity,
        "min_scale": resolved_min_scale,
        "n_sample": n_sample,
        "negative_label": negative_label,
        "pca_seed": pca_seed,
        "screener_kind": screener_kind,
        "seed": seed,
        "selected_labels_kb_path": kb,
        "umap_seed": umap_seed,
    }
