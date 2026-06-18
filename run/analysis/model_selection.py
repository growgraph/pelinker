import click
import gc
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.random import RandomState
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from pelinker.analysis import pooled_min_cluster_size_from_metrics_dfs
from pelinker.grid_export import (
    GRID_COL_CHOSEN_MIN_CLUSTER_SIZE,
    GRID_EXPORT_ID_COLUMNS,
    apply_chosen_min_cluster_size_to_grid,
    grid_export_column_order,
    grid_export_rows_from_report,
    per_combo_metrics_from_grid,
    write_grid_chosen_hyperparameters,
)
from pelinker.sampling import draw_selection_sample
from pelinker.selection import (
    evaluate_selection_from_paths,
    evaluate_selection_sample,
    load_selection_frame,
)
from pelinker.clustering_fusion_ranking import (
    singleton_items_by_dbcv_score,
    top_k_fusion_candidates_by_dbcv_proxy,
)
from pelinker.model_selection_checkpoint import (
    DEFAULT_CHECKPOINT_NAME,
    FailureRecord,
    ModelSelectionCheckpoint,
    RunMode,
    combination_key_from_members,
    compute_run_fingerprint,
    fingerprint_config_from_cli,
    load_checkpoint,
    model_layer_from_singleton_key,
    new_checkpoint,
    reconcile_fusion_checkpoint_params,
    save_checkpoint_atomic,
    score_by_model_layer_from_checkpoint,
    utc_now_iso,
)
from pelinker.config import ClusteringOptimizationConfig, NegativeScreenerConfig
from pelinker.onto import NEGATIVE_LABEL
from pelinker.ops import parse_model_filename
from pelinker.plotting import (
    plot_dbcv_vs_ari_from_grid,
    solve_pooled_grid_by_combo_from_grid,
    plot_heatmap,
    plot_metrics,
    plot_metrics_with_error_bars,
    plot_pca_quality_pairgrid,
    plot_roc_comparison,
    plot_screener_oov_bar,
    plot_cluster_viz,
)
from pelinker.reporting import (
    MODEL_SELECTION_RUN_REPORT_BASENAME,
    MODEL_SELECTION_RUN_REPORT_SCHEMA,
    MODEL_SELECTION_SUMMARY_JSON_SCHEMA,
    CLUSTERING_SEARCH_FINE_METADATA_BASENAME,
    FINE_SCREENER_EVAL_BASENAME,
    CLUSTERING_SEARCH_GRID_CHOSEN_JSON_BASENAME,
    CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME,
    ModelSelectionRunReport,
    ModelSelectionReport,
    ClusteringSearchSummaryRow,
    model_selection_run_report_path,
    model_selection_summary_json_path,
    clustering_search_summary_row_from_flat_dict,
    PerDatapointScores,
    summarize_clustering_reports_for_search,
    write_model_selection_run_report_json,
    write_model_selection_summary_json,
)
from pelinker.transform import TransformConfig

id_columns = list(GRID_EXPORT_ID_COLUMNS)


def _path_by_model_layer(
    valid_files: list[tuple[pathlib.Path, str, str]],
) -> dict[tuple[str, str], pathlib.Path]:
    return {(m, layer): fp for fp, m, layer in valid_files}


def _clustering_optimization_config_for_run(
    *,
    min_class_size: int,
    max_scale: int,
    min_scale: int | None,
    clustering_grid_step: int,
    seed: int,
    frac: float,
    eval_max_rows: int | None,
    n_embedding_batches: int | None,
    batch_size: int,
    negative_label: str,
    screener_kind: str,
) -> ClusteringOptimizationConfig:
    kind: Literal["lda", "svm"] = "svm" if screener_kind == "svm" else "lda"
    ns = NegativeScreenerConfig(kind=kind, negative_label=negative_label)
    return ClusteringOptimizationConfig(
        min_class_size=min_class_size,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
        rns=RandomState(seed=seed),
        base_seed=seed,
        frac=frac,
        eval_max_rows=eval_max_rows,
        n_embedding_batches=n_embedding_batches,
        batch_size=batch_size,
        optimization_method="mean",
        ambient_screener=ns,
    )


def _parse_fusion_members(layer_label: str) -> list[tuple[str, str]]:
    members: list[tuple[str, str]] = []
    for part in layer_label.split("+"):
        p = part.strip()
        model, _, layer = p.partition("/")
        members.append((model, layer))
    return sorted(members, key=lambda t: (t[0], t[1]))


def _ordered_paths_for_fusion(
    path_by_ml: dict[tuple[str, str], pathlib.Path],
    members: list[tuple[str, str]],
) -> list[pathlib.Path]:
    return [path_by_ml[t] for t in members]


def _update_leaderboard_fixed(
    summary_row: ClusteringSearchSummaryRow,
    *,
    best_overall_score: float | None,
    best_overall_model: str | None,
    best_overall_layer: str | None,
    best_per_model: dict[str, float],
) -> tuple[float | None, str | None, str | None, dict[str, float]]:
    mean_dbcv = summary_row.dbcv.mean
    model, layer = summary_row.model, summary_row.layer
    if not model.startswith("fusion"):
        if best_overall_score is None or mean_dbcv > best_overall_score:
            best_overall_score = mean_dbcv
            best_overall_model = model
            best_overall_layer = layer
        if model not in best_per_model or mean_dbcv > best_per_model[model]:
            best_per_model[model] = mean_dbcv
    return best_overall_score, best_overall_model, best_overall_layer, best_per_model


def _recompute_leaderboard_from_results(
    results: list[ClusteringSearchSummaryRow],
) -> tuple[float | None, str | None, str | None, dict[str, float]]:
    best_overall_score = None
    best_overall_model = None
    best_overall_layer = None
    best_per_model: dict[str, float] = {}
    for r in results:
        best_overall_score, best_overall_model, best_overall_layer, best_per_model = (
            _update_leaderboard_fixed(
                r,
                best_overall_score=best_overall_score,
                best_overall_model=best_overall_model,
                best_overall_layer=best_overall_layer,
                best_per_model=best_per_model,
            )
        )
    return best_overall_score, best_overall_model, best_overall_layer, best_per_model


def _validated_oov_label_series(
    df: pd.DataFrame,
    *,
    source_name: str,
) -> pd.Series:
    if "oov_label" not in df.columns:
        raise ValueError(f"{source_name}: missing required 'oov_label' column")

    raw = pd.to_numeric(df["oov_label"], errors="coerce")
    if raw.isna().any():
        raise ValueError(f"{source_name}: oov_label contains non-numeric or NaN values")

    vals = raw.astype(np.int64)
    uniq = set(vals.unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(
            f"{source_name}: oov_label must be binary {{0,1}}; got {sorted(uniq)}"
        )

    n_neg = int((vals == 1).sum())
    n_pos = int((vals == 0).sum())
    if n_neg == 0 or n_pos == 0:
        raise ValueError(
            f"{source_name}: trivial oov_label mask (n_negative={n_neg}, n_positive={n_pos})"
        )
    return vals


def _with_validated_class_label(
    df: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    out = df.copy()
    oov = _validated_oov_label_series(out, source_name=source_name)
    out["class_label"] = np.where(oov == 1, "negative", "positive")
    return out


def _safe_combo_plot_stem(model: str, layer: str) -> str:
    safe_layer = str(layer).replace("/", "_").replace("+", "__")
    return f"{model}_{safe_layer}"


def _pca_pairgrid_output_path(
    report_path: pathlib.Path,
    *,
    model: str,
    layer: str,
    sample_idx: int,
) -> pathlib.Path:
    stem = _safe_combo_plot_stem(model, layer)
    return report_path / f"{stem}_sample{int(sample_idx)}_pca_quality_pairgrid.png"


def _fine_metadata_one_sample_per_combo(fm: pd.DataFrame) -> pd.DataFrame:
    """Keep all rows for the lowest ``sample_idx`` per (model, layer)."""
    pick = fm.groupby(["model", "layer"], sort=False)["sample_idx"].min()
    keyed = fm.merge(
        pick.rename("_pick_sample_idx"),
        left_on=["model", "layer"],
        right_index=True,
    )
    return keyed.loc[keyed["sample_idx"] == keyed["_pick_sample_idx"]].drop(
        columns=["_pick_sample_idx"]
    )


def _write_pca_quality_pairgrids_from_fine_metadata(
    fm: pd.DataFrame,
    report_path: pathlib.Path,
    *,
    source_name: str,
    all_samples: bool = False,
) -> tuple[list[pathlib.Path], list[str]]:
    """
    Write one PCA quality pair grid per (model, layer, sample_idx).

    When ``all_samples`` is false (default), only the lowest ``sample_idx`` per
    (model, layer) is plotted. Each slice uses a single transform fit; rows from
    different combos are never mixed.
    """
    if fm.empty:
        return [], []

    group_cols = ["model", "layer", "sample_idx"]
    missing_group = [c for c in group_cols if c not in fm.columns]
    if missing_group:
        return [], [
            "PCA quality pair grid: fine metadata missing " + ", ".join(missing_group)
        ]

    if not all_samples:
        fm = _fine_metadata_one_sample_per_combo(fm)

    needed = {
        "pca_residual",
        "pca_spectral_entropy",
        "pca_mahalanobis",
        "oov_label",
    }
    if not needed.issubset(fm.columns):
        return [], [
            "PCA quality pair grid: missing "
            + ", ".join(sorted(needed - set(fm.columns)))
        ]

    written: list[pathlib.Path] = []
    skipped: list[str] = []

    for keys, combo_df in fm.groupby(group_cols, sort=False):
        model, layer, sample_idx = keys
        label = f"{model}/{layer} sample {sample_idx}"
        combo_source = f"{source_name} [{label}]"
        try:
            combo_plot = _with_validated_class_label(combo_df, source_name=combo_source)
        except ValueError as exc:
            skipped.append(f"PCA quality pair grid ({label}): {exc}")
            continue

        out_path = _pca_pairgrid_output_path(
            report_path,
            model=str(model),
            layer=str(layer),
            sample_idx=int(sample_idx),
        )
        if plot_pca_quality_pairgrid(
            combo_plot,
            out_path,
            class_col="class_label",
            subtitle=label,
        ):
            written.append(out_path)
        else:
            skipped.append(
                f"PCA quality pair grid ({label}): no plottable rows after dropna"
            )

    if not written and not skipped:
        skipped.append("PCA quality pair grid: no (model, layer, sample_idx) groups")

    return written, skipped


def _materialize_best_report(
    top: ClusteringSearchSummaryRow,
    *,
    valid_files: list[tuple[pathlib.Path, str, str]],
    path_by_ml: dict[tuple[str, str], pathlib.Path],
    transform_config: TransformConfig,
    optimization_config: ClusteringOptimizationConfig,
    selected_labels: set[str] | None,
) -> ModelSelectionReport | None:
    if top.model.startswith("fusion"):
        members = _parse_fusion_members(top.layer)
        try:
            ordered_paths = _ordered_paths_for_fusion(path_by_ml, members)
        except KeyError:
            return None
        return evaluate_selection_from_paths(
            transform_config=transform_config,
            optimization_config=optimization_config,
            file_paths=ordered_paths,
            selected_labels=selected_labels,
            all_metrics_dfs=None,
            show_embedding_read_progress=sys.stdout.isatty(),
        )
    key = (top.model, top.layer)
    path = path_by_ml.get(key)
    if path is None:
        return None
    return evaluate_selection_from_paths(
        transform_config=transform_config,
        optimization_config=optimization_config,
        file_path=path,
        selected_labels=selected_labels,
        all_metrics_dfs=None,
        show_embedding_read_progress=sys.stdout.isatty(),
    )


def _fine_clustering_metadata_df(
    report: ModelSelectionReport,
    *,
    model: str,
    layer: str,
    sample_idx: int,
) -> pd.DataFrame:
    """Per-sample clustering assignments and PCA quality for downstream analysis."""
    cols = ["model", "layer", "sample_idx", "entity", "cluster"]
    optional_cols = ["pmid", "mention"]
    quality_cols = [
        "pca_residual",
        "pca_mahalanobis",
        "pca_spectral_entropy",
        "oov_label",
    ]

    mq = report.mention_quality
    if mq is not None and not mq.empty:
        present_optional = [c for c in optional_cols if c in mq.columns]
        keep = [
            c
            for c in ["entity", "cluster", *present_optional, *quality_cols]
            if c in mq.columns
        ]
        if "entity" not in keep or "cluster" not in keep:
            return pd.DataFrame(columns=cols + present_optional + quality_cols)
        out = mq[keep].copy()
        out.insert(0, "sample_idx", sample_idx)
        out.insert(0, "layer", layer)
        out.insert(0, "model", model)
        return out

    present_optional = [c for c in optional_cols if c in report.assignments.columns]
    keep = [
        c
        for c in ["entity", "cluster", *present_optional]
        if c in report.assignments.columns
    ]
    out0 = report.assignments
    if "entity" not in keep or "cluster" not in keep:
        return pd.DataFrame(columns=cols + present_optional)
    out = out0[keep].copy()
    out.insert(0, "sample_idx", sample_idx)
    out.insert(0, "layer", layer)
    out.insert(0, "model", model)
    n = len(out)
    if len(report.pca_residuals) == n and len(report.pca_mahalanobis) == n:
        out["pca_residual"] = np.asarray(report.pca_residuals, dtype=np.float64)
        out["pca_mahalanobis"] = np.asarray(report.pca_mahalanobis, dtype=np.float64)
    if len(report.pca_spectral_entropy) == n:
        out["pca_spectral_entropy"] = np.asarray(
            report.pca_spectral_entropy, dtype=np.float64
        )
    if len(report.oov_label) != n:
        raise ValueError(
            f"fine metadata labels length mismatch: len(oov_label)={len(report.oov_label)} vs n_rows={n}"
        )
    out["oov_label"] = np.asarray(report.oov_label, dtype=np.int64)
    return out


def _dedupe_per_sample_grid(df: pd.DataFrame) -> pd.DataFrame:
    grid_cols = grid_export_column_order()
    ordered = [c for c in grid_cols if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    out = df[ordered + tail]
    out = out.drop_duplicates(subset=id_columns, keep="last")
    return out


def _read_optional_csv(path: pathlib.Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _merge_new_frames_into_per_sample_grid_csv(
    detail_path: pathlib.Path,
    new_frames: list[pd.DataFrame],
) -> None:
    """Append grid rows to ``results_grid_per_sample.csv`` (merge + dedupe, atomic replace)."""
    if not new_frames:
        return
    new_df = pd.concat(new_frames, ignore_index=True)
    if new_df.empty:
        return
    prior = _read_optional_csv(detail_path)
    if prior is not None and not prior.empty:
        merged = pd.concat([prior, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = _dedupe_per_sample_grid(merged)
    tmp = detail_path.with_suffix(detail_path.suffix + ".tmp")
    merged.to_csv(tmp, index=False)
    tmp.replace(detail_path)


def _fine_metadata_dedupe_subset(df: pd.DataFrame) -> list[str]:
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


def _dedupe_fine_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = _fine_metadata_dedupe_subset(df)
    if cols:
        return df.drop_duplicates(subset=cols, keep="last")
    return df


def _read_optional_jsonl_gzip(path: pathlib.Path) -> pd.DataFrame | None:
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


def _merge_new_frames_into_fine_metadata_jsonl(
    fine_metadata_path: pathlib.Path,
    new_frames: list[pd.DataFrame],
) -> None:
    if not new_frames:
        return
    new_df = pd.concat(new_frames, ignore_index=True)
    if new_df.empty:
        return
    prior = _read_optional_jsonl_gzip(fine_metadata_path)
    merged = (
        pd.concat([prior, new_df], ignore_index=True)
        if prior is not None
        else new_df.copy()
    )
    merged = _dedupe_fine_metadata_df(merged)
    tmp = fine_metadata_path.with_name(fine_metadata_path.name + ".tmp")
    merged.to_json(
        tmp,
        orient="records",
        lines=True,
        compression={"method": "gzip", "compresslevel": 9},
    )
    tmp.replace(fine_metadata_path)


def _dedupe_fine_screener_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    dup_subset = ["combo_key", "sample_idx", "orig_idx"]
    have = [c for c in dup_subset if c in df.columns]
    if len(have) >= 3:
        return df.drop_duplicates(subset=have, keep="last")
    return df


def _merge_new_frames_into_screener_eval_jsonl(
    path: pathlib.Path,
    new_frames: list[pd.DataFrame],
) -> None:
    if not new_frames:
        return
    new_df = pd.concat(new_frames, ignore_index=True)
    if new_df.empty:
        return
    prior = _read_optional_jsonl_gzip(path)
    merged = (
        pd.concat([prior, new_df], ignore_index=True)
        if prior is not None
        else new_df.copy()
    )
    merged = _dedupe_fine_screener_eval_df(merged)
    tmp = path.with_name(path.name + ".tmp")
    merged.to_json(
        tmp,
        orient="records",
        lines=True,
        compression={"method": "gzip", "compresslevel": 9},
    )
    tmp.replace(path)


def _per_datapoint_scores_df(
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


def _combo_key_for_results_row(series: pd.Series) -> str:
    m_str = str(series["model"])
    ly_str = str(series["layer"])
    if m_str.startswith("fusion"):
        return combination_key_from_members(_parse_fusion_members(ly_str))
    return combination_key_from_members([(m_str, ly_str)])


def _mark_combination_done(
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


def _record_failure(
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


def _singleton_score_by_model_layer_from_checkpoint(
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


def _results_from_checkpoint(
    ckpt: ModelSelectionCheckpoint,
) -> list[ClusteringSearchSummaryRow]:
    return [
        clustering_search_summary_row_from_flat_dict(dict(row))
        for _k, row in sorted(ckpt.summaries_by_key.items(), key=lambda item: item[0])
    ]


@dataclass(frozen=True)
class SummaryFigureRenderResult:
    """Outputs from :func:`render_model_selection_summary_figures`."""

    written_paths: tuple[pathlib.Path, ...]
    skipped_messages: tuple[str, ...]
    chosen_hyperparameters_path: pathlib.Path | None = None
    chosen_by_combo: tuple[tuple[str, str, int], ...] = ()
    summary_json_path: pathlib.Path | None = None


def _json_float_metric(val: object) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _summary_combo_entry_from_row(row: pd.Series) -> dict[str, Any]:
    """One leaderboard row for ``model_selection.summary.json``."""
    out: dict[str, Any] = {
        "combo_key": str(row["combo_key"]),
        "model": str(row["model"]),
        "layer": str(row["layer"]),
        "best_score": _json_float_metric(row.get("best_score")),
        "best_score_std": _json_float_metric(row.get("best_score_std")),
        "best_size": _json_float_metric(row.get("best_size")),
        "best_size_std": _json_float_metric(row.get("best_size_std")),
        "ari": _json_float_metric(row.get("ari")),
        "ari_std": _json_float_metric(row.get("ari_std")),
        "screener_auc_mean": _json_float_metric(row.get("screener_auc_mean")),
        "screener_auc_std": _json_float_metric(row.get("screener_auc_std")),
        "screener_lda_auc_mean": _json_float_metric(row.get("screener_lda_auc_mean")),
        "screener_svm_auc_mean": _json_float_metric(row.get("screener_svm_auc_mean")),
        "combined_auc_mean": _json_float_metric(row.get("combined_auc_mean")),
        "combined_auc_std": _json_float_metric(row.get("combined_auc_std")),
        "oov_auc_mean": _json_float_metric(row.get("oov_auc_mean")),
        "oov_auc_std": _json_float_metric(row.get("oov_auc_std")),
    }
    sbk = row.get("screener_best_kind")
    if isinstance(sbk, str) and sbk:
        out["screener_best_kind"] = sbk
    owk = row.get("oov_winner_kind")
    if isinstance(owk, str) and owk:
        out["oov_winner_kind"] = owk
    return out


def _ranked_summary_entries(
    df: pd.DataFrame,
    metric: str,
    *,
    n: int = 3,
) -> list[dict[str, Any]]:
    if metric not in df.columns or df.empty:
        return []
    ranked = df.dropna(subset=[metric]).nlargest(n, metric)
    entries: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        entry = _summary_combo_entry_from_row(row)
        entry["rank"] = rank
        entry["rank_metric"] = metric
        entry["rank_value"] = _json_float_metric(row[metric])
        entries.append(entry)
    return entries


def build_model_selection_summary_payload(
    report_path: pathlib.Path,
    df_results: pd.DataFrame,
    *,
    chosen_by_combo: tuple[tuple[str, str, int], ...] = (),
    chosen_hyperparameters_path: pathlib.Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """
    Compact top-level summary for replot: top screener/combined AUC combos and best DBCV.
    """
    df_heatmap = df_results[~df_results["model"].isin(["fusion2", "fusion3"])].copy()
    top_screener = _ranked_summary_entries(df_heatmap, "screener_auc_mean", n=3)
    top_combined = _ranked_summary_entries(df_heatmap, "combined_auc_mean", n=3)

    best_dbcv: dict[str, Any] | None = None
    if not df_heatmap.empty and "best_score" in df_heatmap.columns:
        dbcv_ranked = df_heatmap.dropna(subset=["best_score"]).nlargest(1, "best_score")
        if not dbcv_ranked.empty:
            best_dbcv = _summary_combo_entry_from_row(dbcv_ranked.iloc[0])

    best_combined: dict[str, Any] | None = None
    if top_combined:
        best_combined = dict(top_combined[0])
        best_combined.pop("rank", None)
        best_combined.pop("rank_metric", None)
        best_combined.pop("rank_value", None)

    grid_block: dict[str, Any] | None = None
    if chosen_by_combo or chosen_hyperparameters_path is not None:
        grid_block = {
            "chosen_min_cluster_size_by_combo": [
                {"model": m, "layer": ly, "min_cluster_size": int(mcs)}
                for m, ly, mcs in chosen_by_combo
            ],
            "chosen_hyperparameters_path": (
                str(chosen_hyperparameters_path)
                if chosen_hyperparameters_path is not None
                else None
            ),
        }

    return {
        "schema": MODEL_SELECTION_SUMMARY_JSON_SCHEMA,
        "generated_at": generated_at or utc_now_iso(),
        "report_dir": str(report_path.resolve()),
        "n_combinations": int(len(df_results)),
        "n_singleton_combinations": int(len(df_heatmap)),
        "rankings": {
            "top_by_screener_auc": top_screener,
            "top_by_combined_auc": top_combined,
            "best_by_dbcv": best_dbcv,
        },
        "best_combined_auc": best_combined,
        "grid": grid_block,
    }


def _write_model_selection_summary_from_results(
    report_path: pathlib.Path,
    df_results: pd.DataFrame,
    *,
    chosen_by_combo: tuple[tuple[str, str, int], ...] = (),
    chosen_hyperparameters_path: pathlib.Path | None = None,
) -> pathlib.Path:
    out_path = model_selection_summary_json_path(report_path)
    payload = build_model_selection_summary_payload(
        report_path,
        df_results,
        chosen_by_combo=chosen_by_combo,
        chosen_hyperparameters_path=chosen_hyperparameters_path,
    )
    write_model_selection_summary_json(out_path, payload)
    return out_path


def _summary_plot_path(report_path: pathlib.Path, stem: str) -> pathlib.Path:
    """Canonical summary figure path (PNG stem); plotting writes PNG+PDF siblings."""
    return report_path / f"{stem}.png"


def _per_combo_metrics_from_grid(
    df_grid: pd.DataFrame,
) -> dict[tuple[str, str], tuple[list[pd.DataFrame], float | None]]:
    """
    Reconstruct per-combination metrics lists from the persisted per-sample grid CSV.

    Returns a mapping of (model, layer) → (metrics_list, chosen_min_cluster_size) where
    ``metrics_list`` is a list of DataFrames (one per sample_idx), each containing
    ``min_cluster_size``, ``dbcv``, ``ari``, ``n_clusters`` — exactly what
    ``plot_metrics`` / ``plot_metrics_with_error_bars`` expect.
    """
    raw = per_combo_metrics_from_grid(df_grid)
    out: dict[tuple[str, str], tuple[list[pd.DataFrame], float | None]] = {}
    for combo, metrics_list in raw.items():
        mcs_val: float | None = None
        if GRID_COL_CHOSEN_MIN_CLUSTER_SIZE in df_grid.columns:
            model, layer = combo
            sub = df_grid[
                (df_grid["model"].astype(str) == model)
                & (df_grid["layer"].astype(str) == layer)
            ]
            vals = sub[GRID_COL_CHOSEN_MIN_CLUSTER_SIZE].dropna()
            if not vals.empty:
                mcs_val = float(vals.iloc[0])
        out[combo] = (metrics_list, mcs_val)
    return out


def render_model_selection_summary_figures(
    report_path: pathlib.Path,
    *,
    checkpoint_path: pathlib.Path | None = None,
    grid_csv_path: pathlib.Path | None = None,
    fine_screener_eval_path: pathlib.Path | None = None,
    fine_metadata_path: pathlib.Path | None = None,
    optimization_config: ClusteringOptimizationConfig | None = None,
    all_pca_pairgrid_samples: bool = False,
) -> SummaryFigureRenderResult:
    """
    Regenerate aggregate model-selection figures from on-disk artifacts (no parquet re-load).

    Uses the checkpoint summaries for score heatmaps and AUC panels, the optional
    per-sample grid CSV for the DBCV vs ARI scatter, gzipped JSON Lines
    ``fine_screener_eval.jsonl.gz`` for ROC comparison, and
    ``fine_metadata.jsonl.gz`` for the PCA quality pair grid.

    Writes the same PNG filenames as ``model_selection`` end-of-run and corresponding
    PDF siblings: ``model.perf.heatmap.png``, ``model.*.heatmap.png`` (scores, ARI,
    screener LDA/SVM/best, OOV, combined), ``model.screener_oov_auc.png``,
    ``model.roc_curves.png``, ``model.roc_best.png`` (best combined-AUC combo),
    ``model.dbcv_vs_ari.png``, ``model_selection.summary.json``, and per-combination
    ``{model}_{layer}_sample{n}_pca_quality_pairgrid.png`` from fine metadata (lowest
    sample index per combo unless ``all_pca_pairgrid_samples`` is true).
    """
    report_path = report_path.expanduser()
    ckpt_file = (
        checkpoint_path.expanduser()
        if checkpoint_path
        else report_path / DEFAULT_CHECKPOINT_NAME
    )
    detail_path = (
        grid_csv_path.expanduser()
        if grid_csv_path
        else report_path / CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME
    )
    screener_path = (
        fine_screener_eval_path.expanduser()
        if fine_screener_eval_path
        else report_path / FINE_SCREENER_EVAL_BASENAME
    )
    fm_path = (
        fine_metadata_path.expanduser()
        if fine_metadata_path
        else report_path / CLUSTERING_SEARCH_FINE_METADATA_BASENAME
    )

    written: list[pathlib.Path] = []
    skipped: list[str] = []
    chosen_hyperparameters_path: pathlib.Path | None = None
    chosen_by_combo_flat: tuple[tuple[str, str, int], ...] = ()
    summary_json_path: pathlib.Path | None = None

    df_grid_detail = _read_optional_csv(detail_path)
    if df_grid_detail is not None and not df_grid_detail.empty:
        df_grid_detail = _dedupe_per_sample_grid(df_grid_detail)

        combo_metrics = _per_combo_metrics_from_grid(df_grid_detail)
        solver_config = optimization_config or ClusteringOptimizationConfig()
        solved_by_combo = solve_pooled_grid_by_combo_from_grid(
            df_grid_detail, solver_config
        )
        chosen_by_combo = {
            combo: result.chosen_min_cluster_size
            for combo, result in solved_by_combo.items()
        }
        if chosen_by_combo:
            df_grid_detail = apply_chosen_min_cluster_size_to_grid(
                df_grid_detail, chosen_by_combo
            )
            tmp_grid = detail_path.with_suffix(detail_path.suffix + ".tmp")
            df_grid_detail.to_csv(tmp_grid, index=False)
            tmp_grid.replace(detail_path)
            written.append(detail_path)

            chosen_hyperparameters_path = (
                report_path / CLUSTERING_SEARCH_GRID_CHOSEN_JSON_BASENAME
            )
            write_grid_chosen_hyperparameters(
                chosen_hyperparameters_path, solved_by_combo, solver_config
            )
            written.append(chosen_hyperparameters_path)
            chosen_by_combo_flat = tuple(
                (model, layer, int(mcs))
                for (model, layer), mcs in sorted(chosen_by_combo.items())
            )

        scatter_path = _summary_plot_path(report_path, "model.dbcv_vs_ari")
        if plot_dbcv_vs_ari_from_grid(df_grid_detail, scatter_path):
            written.append(scatter_path)
        else:
            skipped.append(
                "DBCV vs ARI scatter: insufficient grid columns or no ARI data"
            )

        # Per-combination metric plots (DBCV / ARI / n_clusters vs min_cluster_size).
        for (model, layer), (metrics_list, _stored_mcs) in combo_metrics.items():
            combo = (model, layer)
            solve_result = solved_by_combo.get(combo)
            chosen_mcs = float(chosen_by_combo.get(combo, _stored_mcs or float("nan")))
            safe_layer = layer.replace("/", "_").replace("+", "__")
            if len(metrics_list) > 1:
                out_p = report_path / f"{model}_{safe_layer}_error_bars.png"
                plot_metrics_with_error_bars(
                    metrics_list,
                    out_p,
                    chosen_min_cluster_size=chosen_mcs,
                    grid_solve=solve_result,
                )
            else:
                out_p = report_path / f"{model}_{safe_layer}.png"
                plot_metrics(metrics_list[0], out_p)
            written.append(out_p)
    else:
        skipped.append(f"Grid CSV missing or empty: {detail_path}")

    if not ckpt_file.exists():
        skipped.append(f"Checkpoint missing (heatmaps/bar/ROC need it): {ckpt_file}")
        return SummaryFigureRenderResult(
            written_paths=tuple(written),
            skipped_messages=tuple(skipped),
            chosen_hyperparameters_path=chosen_hyperparameters_path,
            chosen_by_combo=chosen_by_combo_flat,
            summary_json_path=summary_json_path,
        )

    try:
        ckpt = load_checkpoint(ckpt_file)
    except (OSError, ValueError) as e:
        skipped.append(f"Checkpoint unreadable ({ckpt_file}): {e}")
        return SummaryFigureRenderResult(
            written_paths=tuple(written),
            skipped_messages=tuple(skipped),
            chosen_hyperparameters_path=chosen_hyperparameters_path,
            chosen_by_combo=chosen_by_combo_flat,
            summary_json_path=summary_json_path,
        )

    results = _results_from_checkpoint(ckpt)
    if not results:
        skipped.append("Checkpoint has no summaries_by_key rows")
        return SummaryFigureRenderResult(
            written_paths=tuple(written),
            skipped_messages=tuple(skipped),
            chosen_hyperparameters_path=chosen_hyperparameters_path,
            chosen_by_combo=chosen_by_combo_flat,
            summary_json_path=summary_json_path,
        )

    df_results = pd.DataFrame([r.to_flat_dict() for r in results])
    df_results.insert(
        0,
        "combo_key",
        [_combo_key_for_results_row(row) for _, row in df_results.iterrows()],
    )
    df_results = df_results.sort_values(["model", "layer"])
    try:
        summary_json_path = _write_model_selection_summary_from_results(
            report_path,
            df_results,
            chosen_by_combo=chosen_by_combo_flat,
            chosen_hyperparameters_path=chosen_hyperparameters_path,
        )
        written.append(summary_json_path)
    except OSError as e:
        skipped.append(f"Summary JSON: write failed: {e}")

    df_heatmap = df_results[~df_results["model"].isin(["fusion2", "fusion3"])].copy()

    if len(df_heatmap) > 0:
        heatmap_path = _summary_plot_path(report_path, "model.perf.heatmap")
        plot_heatmap(
            df_heatmap, heatmap_path, metric="best_score", metric_label="Best Score"
        )
        written.append(heatmap_path)
    else:
        skipped.append("Score heatmap: no singleton (non-fusion2/3) rows")

    if len(df_heatmap) > 0:
        if "ari" in df_heatmap.columns and df_heatmap["ari"].notna().any():
            ari_heatmap_path = _summary_plot_path(report_path, "model.ari.heatmap")
            plot_heatmap(
                df_heatmap,
                ari_heatmap_path,
                metric="ari",
                metric_label="ARI",
            )
            written.append(ari_heatmap_path)
        else:
            skipped.append("ARI heatmap: missing or all-NaN ARI column")

    auc_heat_specs: list[tuple[str, str, pathlib.Path]] = [
        (
            "screener_lda_auc_mean",
            "Screener LDA AUC",
            pathlib.Path("model.screener_lda_auc.heatmap.png"),
        ),
        (
            "screener_svm_auc_mean",
            "Screener SVM AUC",
            pathlib.Path("model.screener_svm_auc.heatmap.png"),
        ),
        (
            "screener_auc_mean",
            "Screener AUC (best)",
            pathlib.Path("model.screener_auc.heatmap.png"),
        ),
        ("oov_auc_mean", "OOV AUC", pathlib.Path("model.oov_auc.heatmap.png")),
        (
            "combined_auc_mean",
            "Combined AUC",
            pathlib.Path("model.combined_auc.heatmap.png"),
        ),
    ]
    if len(df_heatmap) > 0:
        for col_auc, label_auc, fn_auc in auc_heat_specs:
            if col_auc not in df_heatmap.columns:
                skipped.append(
                    f"{label_auc} heatmap: column {col_auc!r} not in summaries (older checkpoint?)"
                )
                continue
            if not df_heatmap[col_auc].notna().any():
                skipped.append(f"{label_auc} heatmap: all NaN")
                continue
            out_p = report_path / fn_auc
            plot_heatmap(
                df_heatmap,
                out_p,
                metric=col_auc,
                metric_label=label_auc,
                secondary_metric=None,
            )
            written.append(out_p)
    else:
        skipped.append(
            "Screener LDA/SVM/best, OOV, combined AUC heatmaps: no singleton rows"
        )

    bar_path = _summary_plot_path(report_path, "model.screener_oov_auc")
    if plot_screener_oov_bar(df_heatmap, bar_path):
        written.append(bar_path)
    else:
        skipped.append(
            "Screener/OOV/combined bar: missing metrics or empty after fusion filter"
        )

    if screener_path.exists():
        roc_df = _read_optional_jsonl_gzip(screener_path)
        if (
            roc_df is not None
            and not roc_df.empty
            and "combined_auc_mean" in df_results.columns
        ):
            top_df = df_results.dropna(subset=["combined_auc_mean"]).nlargest(
                3,
                "combined_auc_mean",
            )
            top_keys_top = top_df["combo_key"].astype(str).drop_duplicates().tolist()
            roc_out = _summary_plot_path(report_path, "model.roc_curves")
            if plot_roc_comparison(roc_df, roc_out, combo_keys=top_keys_top):
                written.append(roc_out)
            else:
                skipped.append(
                    "ROC curves: plot_roc_comparison returned False (columns or data)"
                )
            best_key = str(top_df["combo_key"].iloc[0])
            roc_best_out = _summary_plot_path(report_path, "model.roc_best")
            if plot_roc_comparison(roc_df, roc_best_out, combo_keys=[best_key]):
                written.append(roc_best_out)
            else:
                skipped.append(
                    "Best-case ROC: plot_roc_comparison returned False (columns or data)"
                )
        else:
            skipped.append(
                "ROC curves: empty screener eval or no combined_auc_mean in summaries"
            )
    else:
        skipped.append(f"ROC curves: file missing {screener_path}")

    fm = _read_optional_jsonl_gzip(fm_path)
    if fm is not None and not fm.empty:
        pca_written, pca_skipped = _write_pca_quality_pairgrids_from_fine_metadata(
            fm,
            report_path,
            source_name=f"summary figures ({fm_path})",
            all_samples=all_pca_pairgrid_samples,
        )
        written.extend(pca_written)
        skipped.extend(pca_skipped)
    else:
        skipped.append(
            f"PCA quality pair grid: fine metadata missing or empty: {fm_path}"
        )

    return SummaryFigureRenderResult(
        written_paths=tuple(written),
        skipped_messages=tuple(skipped),
        chosen_hyperparameters_path=chosen_hyperparameters_path,
        chosen_by_combo=chosen_by_combo_flat,
        summary_json_path=summary_json_path,
    )


def _json_ready_flat_row(row: ClusteringSearchSummaryRow) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in row.to_flat_dict().items():
        if isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Directory containing parquet files",
)
@click.option(
    "--report-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help=(
        "Directory for all run outputs. Canonical artifact: "
        f"{MODEL_SELECTION_RUN_REPORT_BASENAME}."
    ),
)
@click.option(
    "--umap-dim",
    type=click.INT,
    default=8,
    help="UMAP dimensionality for clustering (range: 3-5)",
)
@click.option(
    "--pca-components",
    type=click.INT,
    default=100,
    help="Number of PCA components for dimensionality reduction",
)
@click.option(
    "--cluster-viz-method",
    type=click.Choice(["pca", "umap"], case_sensitive=False),
    default="pca",
    show_default=True,
    help="Reducer for cluster-space visualization (PCA or UMAP on clustering coords).",
)
@click.option(
    "--min-class-size",
    type=click.INT,
    default=20,
    help="Minimum class size for filtering",
)
@click.option(
    "--seed",
    type=click.INT,
    default=13,
    help="Random seed",
)
@click.option(
    "--frac",
    type=click.FLOAT,
    default=0.1,
    help="Fraction of dataset to sample",
)
@click.option(
    "--eval-max-rows",
    type=click.INT,
    default=0,
    show_default=True,
    help=(
        "Stratified cap on mention rows per bootstrap draw (after --frac). "
        "Default 0 = no cap (frac only)."
    ),
)
@click.option(
    "--n-embedding-batches",
    type=click.INT,
    default=None,
    help=(
        "Max parquet read batches per file (see --batch-size for rows per batch); "
        "omit to read all batches"
    ),
)
@click.option(
    "--batch-size",
    type=click.INT,
    default=1000,
    help="Rows per batch when reading mention-level embedding parquet files",
)
@click.option(
    "--prefix",
    type=click.STRING,
    default="res",
    help="Optional prefix for input embedding files to differentiate between models",
)
@click.option(
    "--n-sample",
    type=click.INT,
    default=1,
    help="Number of samples/runs per (model, layer) combination",
)
@click.option(
    "--selected-labels-kb-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Optional path to selected labels KB CSV file. If provided, clustering will only use labels from this KB.",
)
@click.option(
    "--max-scale",
    type=click.INT,
    default=60,
    show_default=True,
    help="Exclusive upper bound for grid evaluation of min_cluster_size (numpy.arange end).",
)
@click.option(
    "--min-scale",
    type=click.INT,
    default=None,
    help=(
        "Inclusive lower bound for min_cluster_size on the grid. "
        "Default: max(1, min_class_size // 2) (legacy: half of --min-class-size)."
    ),
)
@click.option(
    "--clustering-grid-step",
    type=click.INT,
    default=5,
    show_default=True,
    help="Step between consecutive min_cluster_size values on the optimization grid.",
)
@click.option(
    "--fusion-pairs",
    type=click.INT,
    default=5,
    show_default=True,
    help=(
        "After scoring single embeddings (DBCV), evaluate fused pairs: "
        "pick this many distinct pairs with highest sum of singleton DBCV. 0 disables."
    ),
)
@click.option(
    "--fusion-triples",
    type=click.INT,
    default=0,
    show_default=True,
    help=("Same as --fusion-pairs but for three-way fusions (costly). 0 disables."),
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help=(
        "If the checkpoint file exists and matches the run fingerprint, skip completed work. "
        "If the file is missing, start fresh and create it. Use --no-resume to ignore an "
        "existing checkpoint and reinitialize (overwrites on save)."
    ),
)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help=f"Checkpoint JSON path (default: <report-path>/{DEFAULT_CHECKPOINT_NAME})",
)
@click.option(
    "--negative-label",
    type=str,
    default=NEGATIVE_LABEL,
    show_default=True,
    help="Entity label for synthetic negatives (must match embedding parquet).",
)
@click.option(
    "--screener-kind",
    type=click.Choice(["lda", "svm"]),
    default="lda",
    show_default=True,
    help="Estimator saved on Linker when fitting from this pipeline (analysis always logs both).",
)
@click.option(
    "--mode",
    type=click.Choice(["single", "fusion2", "fusion3", "all"]),
    default="all",
    show_default=True,
    help=(
        "single: only single-embedding combinations; fusion2/fusion3: only that fusion order "
        "(requires prior singleton scores in checkpoint); all: singletons then enabled fusions."
    ),
)
def main(
    input_dir: pathlib.Path,
    report_path: pathlib.Path,
    umap_dim: int,
    pca_components: int,
    cluster_viz_method: str,
    min_class_size: int,
    seed: int,
    frac: float,
    eval_max_rows: int,
    n_embedding_batches: int | None,
    batch_size: int,
    n_sample: int,
    prefix: str,
    selected_labels_kb_path: pathlib.Path | None,
    max_scale: int,
    min_scale: int | None,
    clustering_grid_step: int,
    fusion_pairs: int,
    fusion_triples: int,
    resume: bool,
    checkpoint_path: pathlib.Path | None,
    mode: RunMode,
    negative_label: str,
    screener_kind: str,
):
    """
    Process multiple parquet files and compute optimal cluster sizes.

    Files should follow the pattern: <prefix>_<model>_<layer>.parquet

    After scoring each (model, layer) alone (mean DBCV as ``best_score``), optionally
    evaluates fused embeddings: pairs/triples with the highest sum of singleton DBCV
    scores (see ``--fusion-pairs`` / ``--fusion-triples``), then clusters the
    concatenated mention-level vectors via :func:`~pelinker.selection.load_selection_frame`
    and per-bootstrap :func:`~pelinker.selection.evaluate_selection_sample`.

    Checkpointing is on by default (``--resume``): progress is saved under ``--report-path``.
    Use ``--no-resume`` to discard the on-disk checkpoint and start from an empty state.
    """
    console = Console(force_terminal=True, width=120, legacy_windows=False)
    input_dir = input_dir.expanduser()

    selected_labels: set[str] | None = None
    if selected_labels_kb_path is not None:
        selected_labels_kb_path = selected_labels_kb_path.expanduser()
        if not selected_labels_kb_path.exists():
            console.print(
                f"[red]Selected labels KB file not found: {selected_labels_kb_path}[/red]"
            )
            return

        console.print(
            f"[cyan]Loading selected labels KB from {selected_labels_kb_path}[/cyan]"
        )
        try:
            df_selected = pd.read_csv(selected_labels_kb_path)
            if "label" not in df_selected.columns:
                console.print(
                    f"[red]Selected labels KB file must have a 'label' column. Found columns: {list(df_selected.columns)}[/red]"
                )
                return
            selected_labels = set(df_selected["label"].dropna().astype(str))
            console.print(
                f"[green]Loaded {len(selected_labels)} labels from selected labels KB[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error loading selected labels KB: {e}[/red]")
            return

    eval_max_rows_resolved: int | None = (
        None if eval_max_rows <= 0 else int(eval_max_rows)
    )

    report_path = report_path.expanduser()
    report_path.mkdir(parents=True, exist_ok=True)
    detail_path = report_path / CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME
    fine_metadata_path = report_path / CLUSTERING_SEARCH_FINE_METADATA_BASENAME
    fine_screener_eval_path = report_path / FINE_SCREENER_EVAL_BASENAME
    run_report_json_path = model_selection_run_report_path(report_path)
    if not resume:
        for artifact in (detail_path, fine_metadata_path, fine_screener_eval_path):
            try:
                if artifact.exists():
                    artifact.unlink()
            except OSError:
                pass

    fp_payload = fingerprint_config_from_cli(
        input_dir=input_dir,
        umap_dim=umap_dim,
        pca_components=pca_components,
        cluster_viz_method=cluster_viz_method.lower(),
        min_class_size=min_class_size,
        seed=seed,
        frac=frac,
        eval_max_rows=eval_max_rows_resolved,
        n_embedding_batches=n_embedding_batches,
        batch_size=batch_size,
        prefix=prefix,
        n_sample=n_sample,
        selected_labels_kb_path=selected_labels_kb_path,
        max_scale=max_scale,
        min_scale=min_scale,
        clustering_grid_step=clustering_grid_step,
        negative_label=negative_label,
        screener_kind=screener_kind,
    )
    run_fingerprint = compute_run_fingerprint(fp_payload)

    ckpt_path = (
        checkpoint_path.expanduser()
        if checkpoint_path is not None
        else report_path / DEFAULT_CHECKPOINT_NAME
    )

    resumed_from_checkpoint = bool(resume and ckpt_path.exists())
    if resumed_from_checkpoint:
        ckpt = load_checkpoint(ckpt_path)
        if ckpt.run_fingerprint != run_fingerprint:
            console.print(
                "[red]Checkpoint run fingerprint does not match current CLI parameters.[/red]\n"
                f"Checkpoint: {ckpt.run_fingerprint}\n"
                f"Current:    {run_fingerprint}\n"
                "Use the same inputs, or pass --no-resume to reinitialize the checkpoint."
            )
            return
        console.print(
            f"[green]Resuming from checkpoint[/green] [cyan]{ckpt_path}[/cyan]"
        )
    else:
        ckpt = new_checkpoint(run_fingerprint)
        if resume:
            console.print(
                f"[cyan]No checkpoint at[/cyan] [yellow]{ckpt_path}[/yellow][cyan]; "
                f"starting new run (writing checkpoint to[/cyan] [green]{ckpt_path}[/green][cyan]).[/cyan]"
            )
        else:
            console.print(
                f"[cyan]New run (--no-resume); checkpoint reinitialized at[/cyan] "
                f"[green]{ckpt_path}[/green]"
            )

    n_fusion_cleared = reconcile_fusion_checkpoint_params(
        ckpt,
        fusion_pairs=fusion_pairs,
        fusion_triples=fusion_triples,
    )
    if n_fusion_cleared > 0:
        console.print(
            f"[yellow]Fusion settings changed relative to the checkpoint; "
            f"dropped {n_fusion_cleared} cached fusion row(s). Singletons are unchanged.[/yellow]"
        )
    save_checkpoint_atomic(ckpt_path, ckpt)

    completed = set(ckpt.completed_combinations)
    results: list[ClusteringSearchSummaryRow] = _results_from_checkpoint(ckpt)
    (
        best_overall_score,
        best_overall_model,
        best_overall_layer,
        best_per_model,
    ) = _recompute_leaderboard_from_results(results)

    transform_config = TransformConfig(
        pca_components=pca_components,
        umap_components=umap_dim,
        cluster_viz_method=cluster_viz_method.lower(),
        seed=seed,
    )

    parquet_files = sorted(input_dir.glob(f"{prefix}*.parquet"))
    if not parquet_files:
        console.print(
            f"[red]No parquet files found matching pattern '{prefix}*.parquet' in {input_dir}[/red]"
        )
        return

    valid_files: list[tuple[pathlib.Path, str, str]] = []
    for file_path in parquet_files:
        if not file_path.exists():
            continue
        model, layer = parse_model_filename(file_path.name, prefix)
        if model is None or layer is None:
            continue
        # Layer as str matches checkpoint keys from "1:model/layer" and score_by_ml lookups.
        valid_files.append((file_path, model, str(layer)))

    if not valid_files:
        console.print("[red]No valid files to process[/red]")
        return

    path_by_ml = _path_by_model_layer(valid_files)
    metrics_by_file: dict[tuple[str, str], list[pd.DataFrame]] = {}
    best_report: ModelSelectionReport | None = None
    detailed_grid_frames: list[pd.DataFrame] = []
    fine_metadata_frames: list[pd.DataFrame] = []

    run_single = mode in ("single", "all")
    run_fusion2 = mode in ("fusion2", "all")
    run_fusion3 = mode in ("fusion3", "all")

    if mode in ("fusion2", "fusion3"):
        if not _singleton_score_by_model_layer_from_checkpoint(ckpt):
            console.print(
                "[red]Fusion mode requires singleton scores in the checkpoint.[/red] "
                "Run with ``--mode single`` (or ``all``) first, "
                "or ensure summaries for ``1:...`` combinations exist."
            )
            return

    if mode in ("fusion2", "fusion3"):
        expected_singletons = {
            combination_key_from_members([(m, layer)]) for _fp, m, layer in valid_files
        }
        missing = [k for k in sorted(expected_singletons) if k not in completed]
        if missing:
            console.print(
                f"[yellow]Warning:[/yellow] {len(missing)} singleton combination(s) "
                "are not marked complete in the checkpoint; fusion proxy scores may be incomplete."
            )

    # --- single-embedding combinations (arity 1) ---
    if run_single:
        ckpt.stages["single"] = "in_progress"
        save_checkpoint_atomic(ckpt_path, ckpt)

        total_tasks = len(valid_files) * n_sample
        done_tasks = sum(
            n_sample
            for _fp, m, layer in valid_files
            if combination_key_from_members([(m, layer)]) in completed
        )
        initial_total = total_tasks

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Singletons: {len(valid_files)} files × {n_sample} samples...",
                total=initial_total,
                completed=done_tasks,
            )

            for file_path, model, layer in valid_files:
                comb_key = combination_key_from_members([(model, layer)])
                if comb_key in completed:
                    progress.advance(task, advance=n_sample)
                    continue

                file_metrics: list[pd.DataFrame] = []
                file_reports: list[ModelSelectionReport] = []
                all_metrics_dfs: list[pd.DataFrame] = []
                grid_report_samples: list[tuple[int, ModelSelectionReport]] = []
                combo_screener_frames: list[pd.DataFrame] = []

                optimization_config = _clustering_optimization_config_for_run(
                    min_class_size=min_class_size,
                    max_scale=max_scale,
                    min_scale=min_scale,
                    clustering_grid_step=clustering_grid_step,
                    seed=seed,
                    frac=frac,
                    eval_max_rows=eval_max_rows_resolved,
                    n_embedding_batches=n_embedding_batches,
                    batch_size=batch_size,
                    negative_label=negative_label,
                    screener_kind=screener_kind,
                )

                status_parts_base = [
                    f"[cyan]{model}[/cyan]/[yellow]{layer}[/yellow]",
                ]
                try:
                    base_frame = load_selection_frame(
                        file_path=file_path,
                        config=optimization_config,
                        selected_labels=selected_labels,
                        embedding_read_status=lambda m, sp=status_parts_base: (
                            progress.update(
                                task,
                                description=" | ".join([*sp, m]),
                            )
                        ),
                    )
                except Exception as e:
                    base_frame = None
                    console.print(
                        "[yellow]Skipping combo (load failed)[/yellow] "
                        f"{model}/{layer}: {e}"
                    )

                if base_frame is None:
                    progress.advance(task, advance=n_sample)
                    continue

                for sample_idx in range(n_sample):
                    status_parts = [
                        *status_parts_base,
                        f"sample {sample_idx + 1}/{n_sample}",
                    ]
                    if best_overall_score is not None:
                        status_parts.append(
                            f"[green]Best: {best_overall_score:.3f}[/green] "
                            f"([cyan]{best_overall_model}[/cyan]/[yellow]{best_overall_layer}[/yellow])"
                        )
                    if model in best_per_model:
                        status_parts.append(
                            f"[magenta]{model}: {best_per_model[model]:.3f}[/magenta]"
                        )
                    progress.update(task, description=" | ".join(status_parts))

                    try:
                        sample_frame = draw_selection_sample(
                            base_frame,
                            optimization_config,
                            sample_index=sample_idx,
                        )
                        report = evaluate_selection_sample(
                            sample_frame,
                            transform_config,
                            optimization_config=optimization_config,
                            all_metrics_dfs=all_metrics_dfs,
                        )
                    except Exception as e:
                        report = None
                        console.print(
                            "[yellow]Skipping failed sample[/yellow] "
                            f"{model}/{layer} sample {sample_idx + 1}: {e}"
                        )

                    if report is not None:
                        file_metrics.append(report.metrics_df)
                        file_reports.append(report)
                        grid_report_samples.append((sample_idx, report))
                        fine_metadata_frames.append(
                            _fine_clustering_metadata_df(
                                report,
                                model=model,
                                layer=layer,
                                sample_idx=sample_idx,
                            )
                        )
                        if report.screener_oos_datapoints is not None:
                            combo_screener_frames.append(
                                _per_datapoint_scores_df(
                                    report.screener_oos_datapoints,
                                    combo_key=comb_key,
                                    model=model,
                                    layer=layer,
                                    sample_idx=sample_idx,
                                )
                            )
                        if (
                            best_report is None
                            or report.best_score > best_report.best_score
                        ):
                            best_report = report

                    progress.advance(task)
                    gc.collect()

                if file_reports:
                    pooled_mcs, _ = pooled_min_cluster_size_from_metrics_dfs(
                        all_metrics_dfs,
                        optimization_config,
                    )
                    grid_batch = [
                        grid_export_rows_from_report(
                            report,
                            model=model,
                            layer=layer,
                            sample_idx=sample_idx,
                            chosen_min_cluster_size=pooled_mcs,
                        )
                        for sample_idx, report in grid_report_samples
                    ]
                    detailed_grid_frames.extend(grid_batch)

                    metrics_by_file[(model, layer)] = file_metrics
                    summary_row = summarize_clustering_reports_for_search(
                        file_reports,
                        model=model,
                        layer=layer,
                        pooled_min_cluster_size=pooled_mcs,
                    )

                    if len(file_metrics) > 1:
                        plot_metrics_with_error_bars(
                            file_metrics,
                            report_path / f"{model}_{layer}_error_bars.png",
                            chosen_min_cluster_size=float(pooled_mcs),
                        )
                    else:
                        plot_metrics(
                            file_metrics[0], report_path / f"{model}_{layer}.png"
                        )

                    _mark_combination_done(
                        ckpt,
                        ckpt_path,
                        combination_key=comb_key,
                        summary=summary_row,
                        singleton_score_key=comb_key,
                    )
                    n_new = len(file_reports)
                    _merge_new_frames_into_per_sample_grid_csv(
                        detail_path,
                        grid_batch,
                    )
                    _merge_new_frames_into_fine_metadata_jsonl(
                        fine_metadata_path,
                        fine_metadata_frames[-n_new:],
                    )
                    if combo_screener_frames:
                        _merge_new_frames_into_screener_eval_jsonl(
                            fine_screener_eval_path,
                            combo_screener_frames,
                        )
                    completed.add(comb_key)
                    results = _results_from_checkpoint(ckpt)
                    (
                        best_overall_score,
                        best_overall_model,
                        best_overall_layer,
                        best_per_model,
                    ) = _recompute_leaderboard_from_results(results)
                else:
                    _record_failure(
                        ckpt,
                        ckpt_path,
                        combination_key=comb_key,
                        message="All samples failed for this combination",
                    )

        ckpt.stages["single"] = "complete"
        save_checkpoint_atomic(ckpt_path, ckpt)

    # --- fusion combinations ---
    fusion_jobs: list[tuple[int, int]] = []
    if run_fusion2 and fusion_pairs > 0:
        fusion_jobs.append((2, fusion_pairs))
    if run_fusion3 and fusion_triples > 0:
        fusion_jobs.append((3, fusion_triples))

    if fusion_pairs == 0:
        ckpt.stages["fusion2"] = "skipped"
    if fusion_triples == 0:
        ckpt.stages["fusion3"] = "skipped"
    if not fusion_jobs:
        save_checkpoint_atomic(ckpt_path, ckpt)

    score_by_ml = _singleton_score_by_model_layer_from_checkpoint(ckpt)
    singleton_items = singleton_items_by_dbcv_score(valid_files, score_by_ml)

    fusion_batches: list[
        tuple[
            int,
            int,
            list[tuple[list[pathlib.Path], list[str], list[str], float]],
        ]
    ] = []
    for order, top_k in fusion_jobs:
        cand = top_k_fusion_candidates_by_dbcv_proxy(singleton_items, order, top_k)
        if not cand:
            continue
        fusion_batches.append((order, top_k, cand))

    handled_fusion_orders: set[int] = set()
    for order, _top_k, candidates in fusion_batches:
        handled_fusion_orders.add(order)
        model_label = f"fusion{order}"
        stage_name = "fusion2" if order == 2 else "fusion3"
        fusion_task_total = 0
        for paths, models, layers, _sum_proxy in candidates:
            ckey = combination_key_from_members(
                list(zip(models, layers, strict=True)),
            )
            if ckey not in completed:
                fusion_task_total += n_sample

        if fusion_task_total > 0:
            console.print(
                "[cyan]Fused embeddings:[/cyan] evaluating remaining combinations "
                f"× {n_sample} sample(s)..."
            )

        ckpt.stages[stage_name] = "in_progress"
        save_checkpoint_atomic(ckpt_path, ckpt)

        if fusion_task_total > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                refresh_per_second=4,
            ) as fusion_progress:
                ftask = fusion_progress.add_task(
                    "[cyan]Fusion clustering...",
                    total=fusion_task_total,
                )
                for paths, models, layers, sum_proxy in candidates:
                    ordered = sorted(
                        zip(paths, models, layers, strict=True),
                        key=lambda t: (t[1], t[2]),
                    )
                    ordered_paths = [t[0] for t in ordered]
                    o_models = [t[1] for t in ordered]
                    o_layers = [t[2] for t in ordered]
                    layer_label = "+".join(
                        f"{m}/{lyr}" for m, lyr in zip(o_models, o_layers, strict=True)
                    )
                    comb_key = combination_key_from_members(
                        list(zip(o_models, o_layers, strict=True)),
                    )

                    if comb_key in completed:
                        fusion_progress.advance(ftask, advance=n_sample)
                        continue

                    fusion_metrics: list[pd.DataFrame] = []
                    fusion_reports: list[ModelSelectionReport] = []
                    fusion_all_metrics_dfs: list[pd.DataFrame] = []
                    fusion_grid_report_samples: list[
                        tuple[int, ModelSelectionReport]
                    ] = []
                    fusion_combo_screener_frames: list[pd.DataFrame] = []

                    optimization_config = _clustering_optimization_config_for_run(
                        min_class_size=min_class_size,
                        max_scale=max_scale,
                        min_scale=min_scale,
                        clustering_grid_step=clustering_grid_step,
                        seed=seed,
                        frac=frac,
                        eval_max_rows=eval_max_rows_resolved,
                        n_embedding_batches=n_embedding_batches,
                        batch_size=batch_size,
                        negative_label=negative_label,
                        screener_kind=screener_kind,
                    )

                    fusion_status_base = (
                        f"[cyan]{model_label}[/cyan] "
                        f"[yellow]{layer_label}[/yellow] "
                        f"(mean singles≈{sum_proxy / len(paths):.3f})"
                    )
                    try:
                        fusion_base_frame = load_selection_frame(
                            file_paths=ordered_paths,
                            config=optimization_config,
                            selected_labels=selected_labels,
                            embedding_read_status=lambda m, fs=fusion_status_base: (
                                fusion_progress.update(
                                    ftask,
                                    description=f"{fs} | [dim]{m}[/dim]",
                                )
                            ),
                        )
                    except Exception as e:
                        fusion_base_frame = None
                        console.print(
                            "[yellow]Skipping fusion combo (load failed)[/yellow] "
                            f"{layer_label}: {e}"
                        )

                    if fusion_base_frame is None:
                        fusion_progress.advance(ftask, advance=n_sample)
                        continue

                    for sample_idx in range(n_sample):
                        fusion_status = (
                            f"{fusion_status_base} sample {sample_idx + 1}/{n_sample}"
                        )
                        fusion_progress.update(ftask, description=fusion_status)
                        try:
                            sample_frame = draw_selection_sample(
                                fusion_base_frame,
                                optimization_config,
                                sample_index=sample_idx,
                            )
                            report = evaluate_selection_sample(
                                sample_frame,
                                transform_config,
                                optimization_config=optimization_config,
                                all_metrics_dfs=fusion_all_metrics_dfs,
                            )
                        except Exception as e:
                            report = None
                            console.print(
                                "[yellow]Skipping failed fusion sample[/yellow] "
                                f"{layer_label} sample {sample_idx + 1}: {e}"
                            )
                        if report is not None:
                            fusion_metrics.append(report.metrics_df)
                            fusion_reports.append(report)
                            fusion_grid_report_samples.append((sample_idx, report))
                            fine_metadata_frames.append(
                                _fine_clustering_metadata_df(
                                    report,
                                    model=model_label,
                                    layer=layer_label,
                                    sample_idx=sample_idx,
                                )
                            )
                            if report.screener_oos_datapoints is not None:
                                fusion_combo_screener_frames.append(
                                    _per_datapoint_scores_df(
                                        report.screener_oos_datapoints,
                                        combo_key=comb_key,
                                        model=model_label,
                                        layer=layer_label,
                                        sample_idx=sample_idx,
                                    )
                                )
                            if (
                                best_report is None
                                or report.best_score > best_report.best_score
                            ):
                                best_report = report
                        fusion_progress.advance(ftask)
                        gc.collect()

                    if fusion_reports:
                        pooled_mcs, _ = pooled_min_cluster_size_from_metrics_dfs(
                            fusion_all_metrics_dfs,
                            optimization_config,
                        )
                        fusion_grid_batch = [
                            grid_export_rows_from_report(
                                report,
                                model=model_label,
                                layer=layer_label,
                                sample_idx=sample_idx,
                                chosen_min_cluster_size=pooled_mcs,
                            )
                            for sample_idx, report in fusion_grid_report_samples
                        ]
                        detailed_grid_frames.extend(fusion_grid_batch)

                        fusion_summary = summarize_clustering_reports_for_search(
                            fusion_reports,
                            model=model_label,
                            layer=layer_label,
                            pooled_min_cluster_size=pooled_mcs,
                        )
                        metrics_by_file[(model_label, layer_label)] = fusion_metrics

                        safe = layer_label.replace("/", "_").replace("+", "__")
                        out_metric = report_path / f"{model_label}_{safe}.png"
                        if len(fusion_metrics) > 1:
                            plot_metrics_with_error_bars(
                                fusion_metrics,
                                out_metric,
                                chosen_min_cluster_size=float(pooled_mcs),
                            )
                        else:
                            plot_metrics(fusion_metrics[0], out_metric)

                        _mark_combination_done(
                            ckpt,
                            ckpt_path,
                            combination_key=comb_key,
                            summary=fusion_summary,
                            singleton_score_key=None,
                        )
                        n_new = len(fusion_reports)
                        _merge_new_frames_into_per_sample_grid_csv(
                            detail_path,
                            fusion_grid_batch,
                        )
                        _merge_new_frames_into_fine_metadata_jsonl(
                            fine_metadata_path,
                            fine_metadata_frames[-n_new:],
                        )
                        if fusion_combo_screener_frames:
                            _merge_new_frames_into_screener_eval_jsonl(
                                fine_screener_eval_path,
                                fusion_combo_screener_frames,
                            )
                        completed.add(comb_key)
                        results = _results_from_checkpoint(ckpt)
                    else:
                        _record_failure(
                            ckpt,
                            ckpt_path,
                            combination_key=comb_key,
                            message="All fusion samples failed for this combination",
                        )

        ckpt.stages[stage_name] = "complete"
        save_checkpoint_atomic(ckpt_path, ckpt)

    for order, _top_k in fusion_jobs:
        if order in handled_fusion_orders:
            continue
        stage_name = "fusion2" if order == 2 else "fusion3"
        ckpt.stages[stage_name] = "complete"
    if fusion_jobs:
        save_checkpoint_atomic(ckpt_path, ckpt)

    results = _results_from_checkpoint(ckpt)
    if not results:
        console.print("[red]No results to save[/red]")
        return

    df_results = pd.DataFrame([r.to_flat_dict() for r in results])
    df_results.insert(
        0,
        "combo_key",
        [_combo_key_for_results_row(row) for _, row in df_results.iterrows()],
    )
    df_results = df_results.sort_values(["model", "layer"])

    df_grid_detail = _read_optional_csv(detail_path)
    if df_grid_detail is not None and not df_grid_detail.empty:
        df_grid_detail = _dedupe_per_sample_grid(df_grid_detail)
        tmp_grid = detail_path.with_suffix(detail_path.suffix + ".tmp")
        df_grid_detail.to_csv(tmp_grid, index=False)
        tmp_grid.replace(detail_path)

    summary_figures = render_model_selection_summary_figures(
        report_path,
        checkpoint_path=ckpt_path,
        grid_csv_path=detail_path,
        fine_screener_eval_path=fine_screener_eval_path,
    )
    for fig_path in summary_figures.written_paths:
        console.print(
            f"[green]✓[/green] Summary figure saved to: [cyan]{fig_path}[/cyan]"
        )
    for note in summary_figures.skipped_messages:
        console.print(f"[yellow]{note}[/yellow]")

    console.print("\n[bold green]Results Summary[/bold green]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Layer", style="yellow")
    table.add_column("Best Size", justify="right", style="green")
    table.add_column("Clusters", justify="right", style="bright_blue")
    table.add_column("Properties", justify="right", style="magenta")
    table.add_column("Best Score", justify="right", style="blue")
    table.add_column("Scr AUC", justify="right", style="white")
    table.add_column("Comb AUC", justify="right", style="white")

    for _, row in df_results.iterrows():
        if n_sample > 1:
            best_size_str = f"{int(row['best_size'])} ± {row['best_size_std']:.1f}"
            clusters_str = (
                f"{row['n_clusters_emergent']:.1f} ± "
                f"{row['n_clusters_emergent_std']:.1f}"
            )
            properties_str = (
                f"{int(row['number_properties'])} ± {row['number_properties_std']:.1f}"
            )
            best_score_str = f"{row['best_score']:.3f} ± {row['best_score_std']:.3f}"
            sa = row.get("screener_auc_mean")
            ca = row.get("combined_auc_mean")
            scr_auc_str = (
                f"{float(sa):.3f} ± {float(row.get('screener_auc_std') or 0):.3f}"
                if sa is not None and not (isinstance(sa, float) and math.isnan(sa))
                else "—"
            )
            comb_auc_str = (
                f"{float(ca):.3f} ± {float(row.get('combined_auc_std') or 0):.3f}"
                if ca is not None and not (isinstance(ca, float) and math.isnan(ca))
                else "—"
            )
        else:
            best_size_str = str(int(row["best_size"]))
            clusters_str = str(int(round(row["n_clusters_emergent"])))
            properties_str = str(int(row["number_properties"]))
            best_score_str = f"{row['best_score']:.3f}"
            sa = row.get("screener_auc_mean")
            ca = row.get("combined_auc_mean")
            scr_auc_str = (
                f"{float(sa):.3f}"
                if sa is not None and not (isinstance(sa, float) and math.isnan(sa))
                else "—"
            )
            comb_auc_str = (
                f"{float(ca):.3f}"
                if ca is not None and not (isinstance(ca, float) and math.isnan(ca))
                else "—"
            )

        table.add_row(
            str(row["model"]),
            str(row["layer"]),
            best_size_str,
            clusters_str,
            properties_str,
            best_score_str,
            scr_auc_str,
            comb_auc_str,
        )

    console.print(table)
    if df_grid_detail is not None and not df_grid_detail.empty:
        console.print(
            f"[green]✓[/green] Per-sample grid (all min_cluster_size values) saved to: "
            f"[cyan]{detail_path}[/cyan]"
        )
    fm: pd.DataFrame | None = None
    if fine_metadata_path.exists():
        fm = _read_optional_jsonl_gzip(fine_metadata_path)
        if fm is not None:
            console.print(
                "[green]✓[/green] Fine clustering metadata (gzip JSON Lines) saved to: "
                f"[cyan]{fine_metadata_path}[/cyan]"
            )
    if fine_screener_eval_path.exists():
        se = _read_optional_jsonl_gzip(fine_screener_eval_path)
        if se is not None:
            console.print(
                "[green]✓[/green] Fine screener eval datapoints saved to: "
                f"[cyan]{fine_screener_eval_path}[/cyan]"
            )

    top_idx = df_results["best_score"].idxmax()
    top_row = df_results.loc[top_idx]
    top_summary = clustering_search_summary_row_from_flat_dict(
        {str(k): top_row[k] for k in top_row.index}
    )
    console.print(
        f"\n[bold green]Best mean DBCV (best_score): {float(top_row['best_score']):.3f}[/bold green] "
        f"([cyan]{top_row['model']}[/cyan]/[yellow]{top_row['layer']}[/yellow])"
    )

    if best_report is None:
        viz_config = _clustering_optimization_config_for_run(
            min_class_size=min_class_size,
            max_scale=max_scale,
            min_scale=min_scale,
            clustering_grid_step=clustering_grid_step,
            seed=seed,
            frac=frac,
            eval_max_rows=eval_max_rows_resolved,
            n_embedding_batches=n_embedding_batches,
            batch_size=batch_size,
            negative_label=negative_label,
            screener_kind=screener_kind,
        )
        console.print(
            "[cyan]Materializing best clustering report for UMAP (not held in memory)...[/cyan]"
        )
        best_report = _materialize_best_report(
            top_summary,
            valid_files=valid_files,
            path_by_ml=path_by_ml,
            transform_config=transform_config,
            optimization_config=viz_config,
            selected_labels=selected_labels,
        )

    if best_report is not None:
        cluster_viz_path = report_path / "cluster_viz_best.html"
        console.print(
            "[green]✓[/green] Generating cluster-space visualization for best model..."
        )
        cluster_viz_df = pd.DataFrame(
            best_report.cluster_viz,
            columns=[
                f"cviz_{j:02d}" for j in range(int(best_report.cluster_viz.shape[1]))
            ],
            index=best_report.assignments.index,
        )
        assign_cols = (
            ["entity", "cluster"]
            if "entity" in best_report.assignments.columns
            else ["cluster"]
        )
        plot_df = pd.concat(
            [
                best_report.assignments[assign_cols].rename(
                    columns={"cluster": "class"}
                ),
                cluster_viz_df,
            ],
            axis=1,
        )
        plot_cluster_viz(
            plot_df,
            output_path=str(cluster_viz_path),
            viz_method=best_report.cluster_viz_method,
        )
        console.print(
            "[green]✓[/green] Cluster visualization saved to: "
            f"[cyan]{cluster_viz_path}[/cyan]"
        )

    checkpoint_payload = {
        "path": str(ckpt_path),
        "stages": dict(ckpt.stages),
        "completed_combinations_count": len(ckpt.completed_combinations),
        "failure_count": len(ckpt.failures),
        "resumed": resumed_from_checkpoint,
    }
    best_overall_payload: dict[str, object] | None = None
    if best_overall_score is not None:
        best_overall_payload = {
            "model": best_overall_model,
            "layer": best_overall_layer,
            "best_score": float(best_overall_score),
        }
        br_sel = df_results.loc[
            (df_results["model"].astype(str) == str(best_overall_model))
            & (df_results["layer"].astype(str) == str(best_overall_layer))
        ]
        bk = (
            br_sel.iloc[0]
            if len(br_sel) > 0
            else df_results.loc[df_results["best_score"].idxmax()]
        )
        sbk_mean = bk.get("screener_auc_mean")
        cb_mean = bk.get("combined_auc_mean")
        ob_mean = bk.get("oov_auc_mean")
        if sbk_mean is not None and not (
            isinstance(sbk_mean, float) and math.isnan(sbk_mean)
        ):
            best_overall_payload["screener_auc"] = float(sbk_mean)
        if cb_mean is not None and not (
            isinstance(cb_mean, float) and math.isnan(cb_mean)
        ):
            best_overall_payload["combined_auc"] = float(cb_mean)
        if ob_mean is not None and not (
            isinstance(ob_mean, float) and math.isnan(ob_mean)
        ):
            best_overall_payload["oov_auc"] = float(ob_mean)
        s_kind = bk.get("screener_best_kind")
        if isinstance(s_kind, str) and s_kind:
            best_overall_payload["screener_best_kind"] = s_kind
        o_kind = bk.get("oov_winner_kind")
        if isinstance(o_kind, str) and o_kind:
            best_overall_payload["oov_winner_kind"] = o_kind

    run_report = ModelSelectionRunReport(
        schema=MODEL_SELECTION_RUN_REPORT_SCHEMA,
        generated_at=utc_now_iso(),
        run_fingerprint=run_fingerprint,
        run_config=fp_payload,
        checkpoint=checkpoint_payload,
        combinations=[
            {
                "combination_key": combination_key_from_members(
                    _parse_fusion_members(r.layer)
                    if r.model.startswith("fusion")
                    else [(r.model, r.layer)]
                ),
                "model": r.model,
                "layer": r.layer,
                "summary": _json_ready_flat_row(r),
            }
            for r in results
        ],
        failures=[
            {
                "combination_key": f.combination_key,
                "error": f.error,
                "at": f.at,
            }
            for f in ckpt.failures
        ],
        best_overall=best_overall_payload,
        best_per_model={k: float(v) for k, v in sorted(best_per_model.items())},
    )
    write_model_selection_run_report_json(run_report_json_path, run_report)
    console.print(
        f"\n[green]✓[/green] Standardized run report saved to: [cyan]{run_report_json_path}[/cyan]"
    )


if __name__ == "__main__":
    main()
