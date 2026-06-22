"""Artifact I/O for model-selection runs (grid CSV, fine metadata, screener eval)."""

from __future__ import annotations

import pathlib

import pandas as pd

from pelinker.grid_export import GRID_EXPORT_ID_COLUMNS, grid_export_column_order
from pelinker.model_selection_checkpoint import (
    FailureRecord,
    ModelSelectionCheckpoint,
    model_layer_from_singleton_key,
    save_checkpoint_atomic,
    score_by_model_layer_from_checkpoint,
    utc_now_iso,
)
from pelinker.reporting import (
    ClusteringSearchSummaryRow,
    PerDatapointScores,
    clustering_search_summary_row_from_flat_dict,
)

id_columns = list(GRID_EXPORT_ID_COLUMNS)


def dedupe_per_sample_grid(df: pd.DataFrame) -> pd.DataFrame:
    grid_cols = grid_export_column_order()
    ordered = [c for c in grid_cols if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    out = df[ordered + tail]
    out = out.drop_duplicates(subset=id_columns, keep="last")
    return out


def read_optional_csv(path: pathlib.Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def merge_new_frames_into_per_sample_grid_csv(
    detail_path: pathlib.Path,
    new_frames: list[pd.DataFrame],
) -> None:
    """Append grid rows to ``results_grid_per_sample.csv`` (merge + dedupe, atomic replace)."""
    if not new_frames:
        return
    new_df = pd.concat(new_frames, ignore_index=True)
    if new_df.empty:
        return
    prior = read_optional_csv(detail_path)
    if prior is not None and not prior.empty:
        merged = pd.concat([prior, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = dedupe_per_sample_grid(merged)
    tmp = detail_path.with_suffix(detail_path.suffix + ".tmp")
    merged.to_csv(tmp, index=False)
    tmp.replace(detail_path)


def fine_metadata_dedupe_subset(df: pd.DataFrame) -> list[str]:
    wanted = [
        "model",
        "layer",
        "sample_idx",
        "pmid",
        "mention",
        "entity",
        "cluster",
    ]
    return [c for c in wanted if c in df.columns]


def dedupe_fine_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = fine_metadata_dedupe_subset(df)
    if cols:
        return df.drop_duplicates(subset=cols, keep="last")
    return df


def read_optional_jsonl_gzip(path: pathlib.Path) -> pd.DataFrame | None:
    """Load gzipped JSON Lines (pandas); return None if missing or unreadable."""
    if not path.exists():
        return None
    try:
        df = pd.read_json(path, lines=True, compression="gzip")
        if df.empty:
            return None
        return df
    except Exception:
        return None


def merge_new_frames_into_fine_metadata_jsonl(
    fine_metadata_path: pathlib.Path,
    new_frames: list[pd.DataFrame],
) -> None:
    if not new_frames:
        return
    new_df = pd.concat(new_frames, ignore_index=True)
    if new_df.empty:
        return
    prior = read_optional_jsonl_gzip(fine_metadata_path)
    merged = (
        pd.concat([prior, new_df], ignore_index=True)
        if prior is not None
        else new_df.copy()
    )
    merged = dedupe_fine_metadata_df(merged)
    tmp = fine_metadata_path.with_name(fine_metadata_path.name + ".tmp")
    merged.to_json(
        tmp,
        orient="records",
        lines=True,
        compression={"method": "gzip", "compresslevel": 9},
    )
    tmp.replace(fine_metadata_path)


def dedupe_fine_screener_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    dup_subset = ["combo_key", "sample_idx", "orig_idx"]
    have = [c for c in dup_subset if c in df.columns]
    if len(have) >= 3:
        return df.drop_duplicates(subset=have, keep="last")
    return df


def merge_new_frames_into_screener_eval_jsonl(
    path: pathlib.Path,
    new_frames: list[pd.DataFrame],
) -> None:
    if not new_frames:
        return
    new_df = pd.concat(new_frames, ignore_index=True)
    if new_df.empty:
        return
    prior = read_optional_jsonl_gzip(path)
    merged = (
        pd.concat([prior, new_df], ignore_index=True)
        if prior is not None
        else new_df.copy()
    )
    merged = dedupe_fine_screener_eval_df(merged)
    tmp = path.with_name(path.name + ".tmp")
    merged.to_json(
        tmp,
        orient="records",
        lines=True,
        compression={"method": "gzip", "compresslevel": 9},
    )
    tmp.replace(path)


def per_datapoint_scores_df(
    scores: PerDatapointScores,
    *,
    combo_key: str,
    model: str,
    layer: str,
    sample_idx: int,
) -> pd.DataFrame:
    n = len(scores.orig_idx)
    if n == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "combo_key": [combo_key] * n,
            "model": [model] * n,
            "layer": [layer] * n,
            "sample_idx": [sample_idx] * n,
            "orig_idx": list(scores.orig_idx),
            "entity": list(scores.entity),
            "y_true": list(scores.y_true),
            "screener_lda_score": list(scores.screener_lda_score),
            "screener_svm_score": list(scores.screener_svm_score),
            "screener_best_score": list(scores.screener_best_score),
            "oov_score": list(scores.oov_score),
            "combined_score": list(scores.combined_score),
        }
    )


def mark_combination_done(
    ckpt: ModelSelectionCheckpoint,
    ckpt_path: pathlib.Path,
    *,
    combination_key: str,
    summary: ClusteringSearchSummaryRow,
    singleton_score_key: str | None,
) -> None:
    if combination_key not in ckpt.completed_combinations:
        ckpt.completed_combinations.append(combination_key)
    flat = summary.to_flat_dict()
    ckpt.summaries_by_key[combination_key] = dict(flat)
    if singleton_score_key is not None:
        ckpt.singleton_scores_by_key[singleton_score_key] = float(summary.dbcv.mean)
    save_checkpoint_atomic(ckpt_path, ckpt)


def record_failure(
    ckpt: ModelSelectionCheckpoint,
    ckpt_path: pathlib.Path,
    *,
    combination_key: str,
    message: str,
) -> None:
    ckpt.failures.append(
        FailureRecord(combination_key=combination_key, error=message, at=utc_now_iso())
    )
    save_checkpoint_atomic(ckpt_path, ckpt)


def singleton_score_by_model_layer_from_checkpoint(
    ckpt: ModelSelectionCheckpoint,
) -> dict[tuple[str, str], float]:
    """Mean DBCV per (model, layer) for fusion proxy (singletons only)."""
    out = score_by_model_layer_from_checkpoint(ckpt.singleton_scores_by_key)
    if out:
        return out
    for key, row in ckpt.summaries_by_key.items():
        if not key.startswith("1:"):
            continue
        ml = model_layer_from_singleton_key(key)
        score = row.get("best_score")
        if score is not None:
            out[ml] = float(score)
    return out


def results_from_checkpoint(
    ckpt: ModelSelectionCheckpoint,
) -> list[ClusteringSearchSummaryRow]:
    return [
        clustering_search_summary_row_from_flat_dict(dict(row))
        for _k, row in sorted(ckpt.summaries_by_key.items(), key=lambda item: item[0])
    ]
