"""Fine clustering metadata extraction and PCA quality pairgrid export."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from pelinker.plotting import plot_pca_quality_pairgrid
from pelinker.reporting import ModelSelectionReport


def validated_oov_label_series(
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


def with_validated_class_label(
    df: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    out = df.copy()
    oov = validated_oov_label_series(out, source_name=source_name)
    out["class_label"] = np.where(oov == 1, "negative", "positive")
    return out


def safe_combo_plot_stem(model: str, layer: str) -> str:
    safe_layer = str(layer).replace("/", "_").replace("+", "__")
    return f"{model}_{safe_layer}"


def pca_pairgrid_output_path(
    report_path: pathlib.Path,
    *,
    model: str,
    layer: str,
    sample_idx: int,
) -> pathlib.Path:
    stem = safe_combo_plot_stem(model, layer)
    return report_path / f"{stem}_sample{int(sample_idx)}_pca_quality_pairgrid.png"


def fine_metadata_one_sample_per_combo(fm: pd.DataFrame) -> pd.DataFrame:
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


def write_pca_quality_pairgrids_from_fine_metadata(
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
        fm = fine_metadata_one_sample_per_combo(fm)

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
            combo_plot = with_validated_class_label(combo_df, source_name=combo_source)
        except ValueError as exc:
            skipped.append(f"PCA quality pair grid ({label}): {exc}")
            continue

        out_path = pca_pairgrid_output_path(
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


def clustering_metadata_df(
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
