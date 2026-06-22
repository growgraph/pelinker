"""Summary JSON and figure regeneration for model-selection reports."""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pelinker.config import ClusteringOptimizationConfig
from pelinker.grid_export import (
    GRID_COL_CHOSEN_MIN_CLUSTER_SIZE,
    apply_chosen_min_cluster_size_to_grid,
    per_combo_metrics_from_grid,
    write_grid_chosen_hyperparameters,
)
from pelinker.model_selection.artifacts import (
    dedupe_per_sample_grid,
    read_optional_csv,
    read_optional_jsonl_gzip,
    results_from_checkpoint,
)
from pelinker.model_selection.fine_metadata import (
    write_pca_quality_pairgrids_from_fine_metadata,
)
from pelinker.model_selection.fusion import combo_key_for_results_row
from pelinker.model_selection_checkpoint import (
    DEFAULT_CHECKPOINT_NAME,
    load_checkpoint,
    utc_now_iso,
)
from pelinker.plotting import (
    plot_dbcv_vs_ari_from_grid,
    plot_heatmap,
    plot_metrics,
    plot_metrics_with_error_bars,
    plot_roc_comparison,
    plot_screener_oov_bar,
    solve_pooled_grid_by_combo_from_grid,
)
from pelinker.reporting import (
    CLUSTERING_SEARCH_FINE_METADATA_BASENAME,
    CLUSTERING_SEARCH_GRID_CHOSEN_JSON_BASENAME,
    CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME,
    FINE_SCREENER_EVAL_BASENAME,
    MODEL_SELECTION_SUMMARY_JSON_SCHEMA,
    model_selection_summary_json_path,
    write_model_selection_summary_json,
)


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


def write_model_selection_summary_from_results(
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

    df_grid_detail = read_optional_csv(detail_path)
    if df_grid_detail is not None and not df_grid_detail.empty:
        df_grid_detail = dedupe_per_sample_grid(df_grid_detail)

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

    results = results_from_checkpoint(ckpt)
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
        [combo_key_for_results_row(row) for _, row in df_results.iterrows()],
    )
    df_results = df_results.sort_values(["model", "layer"])
    try:
        summary_json_path = write_model_selection_summary_from_results(
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
        roc_df = read_optional_jsonl_gzip(screener_path)
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

    fm = read_optional_jsonl_gzip(fm_path)
    if fm is not None and not fm.empty:
        pca_written, pca_skipped = write_pca_quality_pairgrids_from_fine_metadata(
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
