"""Structured checkpoint I/O for ``run/analysis/clustering_quality.py`` runs."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal

CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_NAME = "clustering_quality.state.json"

StageState = Literal["pending", "in_progress", "complete", "skipped"]
RunMode = Literal["single", "fusion2", "fusion3", "all"]


@dataclass
class FailureRecord:
    """One recorded failure for a combination."""

    combination_key: str
    error: str
    at: str


@dataclass
class ClusteringQualityCheckpoint:
    """On-disk checkpoint for resumable clustering quality runs."""

    version: int = CHECKPOINT_VERSION
    run_fingerprint: str = ""
    created_at: str = ""
    updated_at: str = ""
    completed_combinations: list[str] = field(default_factory=list)
    summaries_by_key: dict[str, dict[str, str | float | None]] = field(
        default_factory=dict
    )
    singleton_scores_by_key: dict[str, float] = field(default_factory=dict)
    stages: dict[str, StageState] = field(default_factory=dict)
    failures: list[FailureRecord] = field(default_factory=list)
    # Last --fusion-pairs / --fusion-triples reflected in fusion summaries; if CLI
    # differs, arity-2/3 checkpoint rows are dropped (singletons keep the same fp).
    checkpoint_fusion_pairs: int | None = None
    checkpoint_fusion_triples: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_fingerprint": self.run_fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "checkpoint_fusion_pairs": self.checkpoint_fusion_pairs,
            "checkpoint_fusion_triples": self.checkpoint_fusion_triples,
            "completed_combinations": sorted(self.completed_combinations),
            "summaries_by_key": {
                k: dict(v) for k, v in sorted(self.summaries_by_key.items())
            },
            "singleton_scores_by_key": {
                k: float(v) for k, v in sorted(self.singleton_scores_by_key.items())
            },
            "stages": dict(sorted(self.stages.items())),
            "failures": [
                {"combination_key": f.combination_key, "error": f.error, "at": f.at}
                for f in self.failures
            ],
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> ClusteringQualityCheckpoint:
        if int(data.get("version", -1)) != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version: {data.get('version')!r}; "
                f"expected {CHECKPOINT_VERSION}"
            )
        failures_raw = data.get("failures") or []
        failures = [
            FailureRecord(
                combination_key=str(fr["combination_key"]),
                error=str(fr["error"]),
                at=str(fr["at"]),
            )
            for fr in failures_raw
        ]
        cfp = data.get("checkpoint_fusion_pairs")
        cft = data.get("checkpoint_fusion_triples")
        return ClusteringQualityCheckpoint(
            version=int(data["version"]),
            run_fingerprint=str(data["run_fingerprint"]),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            completed_combinations=list(data.get("completed_combinations") or []),
            summaries_by_key=dict(data.get("summaries_by_key") or {}),
            singleton_scores_by_key={
                str(k): float(v)
                for k, v in (data.get("singleton_scores_by_key") or {}).items()
            },
            stages=dict(data.get("stages") or {}),
            failures=failures,
            checkpoint_fusion_pairs=int(cfp) if cfp is not None else None,
            checkpoint_fusion_triples=int(cft) if cft is not None else None,
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def combination_key_from_members(members: list[tuple[str, str]]) -> str:
    """
    Canonical key for a combination of (model, layer) embeddings.

    Members are sorted lexicographically by (model, layer). Arity is ``len(members)``.
    """
    if not members:
        raise ValueError("members must be non-empty")
    arity = len(members)
    sorted_m = sorted(members, key=lambda t: (t[0], t[1]))
    inner = "+".join(f"{m}/{layer}" for m, layer in sorted_m)
    return f"{arity}:{inner}"


def compute_run_fingerprint(config: dict[str, Any]) -> str:
    """Stable SHA256 over a JSON-serializable config (sorted keys)."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(blob).hexdigest()


def fingerprint_config_from_cli(
    *,
    input_dir: pathlib.Path,
    umap_dim: int,
    pca_components: int,
    min_class_size: int,
    seed: int,
    frac: float,
    head: int | None,
    batch_size: int,
    prefix: str,
    n_sample: int,
    selected_labels_kb_path: pathlib.Path | None,
    max_scale: int,
) -> dict[str, Any]:
    """Parameters that must match between checkpoint and resume.

    ``--fusion-pairs`` / ``--fusion-triples`` are excluded: they only affect which
    fusion jobs run; changing them invalidates fusion rows via
    :func:`reconcile_fusion_checkpoint_params` instead of the run fingerprint.

    ``--mode`` is excluded so you can resume with a different mode (e.g. singletons
    under ``all``, then ``--mode fusion2``) on the same data fingerprint.
    """
    kb = None
    if selected_labels_kb_path is not None:
        kb = str(selected_labels_kb_path.expanduser().resolve())
    return {
        "batch_size": batch_size,
        "frac": frac,
        "head": head,
        "input_dir": str(input_dir.expanduser().resolve()),
        "max_scale": max_scale,
        "min_class_size": min_class_size,
        "n_sample": n_sample,
        "pca_components": pca_components,
        "prefix": prefix,
        "seed": seed,
        "selected_labels_kb_path": kb,
        "umap_dim": umap_dim,
    }


def _is_fusion_combination_key(key: str) -> bool:
    if ":" not in key:
        return False
    arity_str, _body = key.split(":", 1)
    try:
        arity = int(arity_str)
    except ValueError:
        return False
    return arity >= 2


def reconcile_fusion_checkpoint_params(
    ckpt: ClusteringQualityCheckpoint,
    *,
    fusion_pairs: int,
    fusion_triples: int,
) -> int:
    """
    If fusion CLI counts changed, drop arity >= 2 checkpoint state so fusion is
    recomputed while singletons stay cacheable.

    Returns how many distinct fusion combination keys were removed from completed /
    summaries (0 if fusion params already matched this run).
    """
    if (
        ckpt.checkpoint_fusion_pairs == fusion_pairs
        and ckpt.checkpoint_fusion_triples == fusion_triples
    ):
        return 0

    removed_keys = {
        k for k in ckpt.completed_combinations if _is_fusion_combination_key(k)
    } | {k for k in ckpt.summaries_by_key if _is_fusion_combination_key(k)}

    ckpt.completed_combinations = [
        k for k in ckpt.completed_combinations if not _is_fusion_combination_key(k)
    ]
    ckpt.summaries_by_key = {
        k: v
        for k, v in ckpt.summaries_by_key.items()
        if not _is_fusion_combination_key(k)
    }
    ckpt.failures = [
        f for f in ckpt.failures if not _is_fusion_combination_key(f.combination_key)
    ]
    for name in ("fusion2", "fusion3"):
        ckpt.stages[name] = "pending"
    ckpt.checkpoint_fusion_pairs = fusion_pairs
    ckpt.checkpoint_fusion_triples = fusion_triples
    return len(removed_keys)


def load_checkpoint(path: pathlib.Path) -> ClusteringQualityCheckpoint:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be a JSON object")
    return ClusteringQualityCheckpoint.from_json_dict(data)


def save_checkpoint_atomic(
    path: pathlib.Path, checkpoint: ClusteringQualityCheckpoint
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.updated_at = utc_now_iso()
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(
        checkpoint.to_json_dict(), indent=2, sort_keys=True, ensure_ascii=False
    )
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(path)


def new_checkpoint(fingerprint: str) -> ClusteringQualityCheckpoint:
    now = utc_now_iso()
    return ClusteringQualityCheckpoint(
        version=CHECKPOINT_VERSION,
        run_fingerprint=fingerprint,
        created_at=now,
        updated_at=now,
        stages={
            "single": "pending",
            "fusion2": "pending",
            "fusion3": "pending",
        },
    )


def model_layer_from_singleton_key(key: str) -> tuple[str, str]:
    """Parse ``1:model/layer`` into (model, layer)."""
    if ":" not in key:
        raise ValueError(f"invalid combination key: {key!r}")
    arity_str, body = key.split(":", 1)
    if int(arity_str) != 1:
        raise ValueError(f"expected arity 1 key, got {key!r}")
    if "/" not in body:
        raise ValueError(f"invalid singleton key body: {body!r}")
    model, layer = body.split("/", 1)
    return model, layer


def score_by_model_layer_from_checkpoint(
    singleton_scores_by_key: dict[str, float],
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for k, score in singleton_scores_by_key.items():
        if not k.startswith("1:"):
            continue
        ml = model_layer_from_singleton_key(k)
        out[ml] = float(score)
    return out
