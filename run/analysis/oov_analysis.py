#!/usr/bin/env python
"""Pre-classifier anomaly-space analysis for entity linking.

Loads the fit-B clustering report (``pca_residuals`` / ``pca_mahalanobis`` from
the training stage) and the OOV mention dump produced by
``link_files --dump-mention-anomaly`` and produces a suite of
publication-ready PDF figures that:

  1.  Show marginal distributions of PCA residual and Mahalanobis distance
      for each data origin (fit / KB-validated OOV / unconfirmed OOV).
  2.  Show the 2-D joint anomaly space with KDE contours and the linear
      decision boundary.
  3.  Provide ROC and Precision-Recall curves for each univariate and
      bivariate threshold strategy to support principled operating-point
      selection.
  4.  Sweep the decision-boundary offset and display the full
      precision / recall / F1 surface.
  5.  Produce a composite "paper figure" combining the four key panels.
  6.  Analyse how the pre-classifier anomaly filter aligns with the
      trained negative screener.

Usage
-----
    uv run python run/analysis/oov_analysis.py \\
        --fit-report   ../../../pelinker/reports/fit-b/linker_fit.clustering_report.json.gz \\
        --oov-csv      ../../../pelinker/reports/current/oov.csv \\
        --out-dir      ~/data/pelinker/figs/paper
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from pelinker.io.json_files import load_json_path

# ─── Aesthetic constants ──────────────────────────────────────────────────────

PALETTE: dict[str, str] = {
    "from_fit": "#1A5276",  # deep navy   – training anchor
    "kb_match": "#1E8449",  # forest green – OOV confirmed by KB lemma
    "oov": "#A93226",  # deep crimson – unconfirmed OOV
}

ORIGIN_ORDER = ["from_fit", "kb_match", "oov"]

ORIGIN_LABELS: dict[str, str] = {
    "from_fit": "Training (fit-B)",
    "kb_match": "OOV – KB validated",
    "oov": "OOV – unconfirmed",
}

COL_R = "pca_residual"  # PCA reconstruction residual
COL_M = "pca_mahalanobis"  # Mahalanobis distance from training manifold
COL_Z = "anomaly_score_max_z"  # max component Z-score


# ─── Matplotlib style ─────────────────────────────────────────────────────────


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E8E8E8",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ─── Data loading & assembly ──────────────────────────────────────────────────


def _load_fit_report(path: Path) -> pd.DataFrame:
    report: dict = load_json_path(path)
    df = pd.DataFrame(
        {
            COL_R: report["pca_residuals"],
            COL_M: report["pca_mahalanobis"],
        }
    )
    df["origin"] = "from_fit"
    df["mention"] = "NA"
    df["score"] = 1.0
    df["screener_decision"] = np.nan
    df["screener_is_negative"] = False
    df[COL_Z] = np.nan
    return df


def _load_oov(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".pkl":
        df = pd.read_pickle(path)
    elif suffix == ".gz":
        try:
            df = pd.read_parquet(path)
        except Exception:
            try:
                obj = load_json_path(path)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                df = pd.read_csv(path, compression="gzip")
            else:
                df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    else:
        df = pd.read_csv(path)

    # Annotate origin: KB-validated OOV vs pure OOV
    kb_col = next(
        (
            c
            for c in ("lemma_kb_matches_predicted_entity", "is_kb_match")
            if c in df.columns
        ),
        None,
    )
    df["origin"] = "oov"
    if kb_col is not None:
        df.loc[df[kb_col].fillna(False).astype(bool), "origin"] = "kb_match"
    return df


def _assemble(fit_df: pd.DataFrame, oov_df: pd.DataFrame) -> pd.DataFrame:
    shared = [COL_R, COL_M, "origin", "mention", "score"]
    optional = [
        COL_Z,
        "screener_decision",
        "screener_is_negative",
        "entity_id_predicted",
        "lemma_kb_matches_predicted_entity",
    ]

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        cols = shared + [c for c in optional if c in df.columns]
        out = df[cols].copy()
        for c in shared + optional:
            if c not in out.columns:
                out[c] = np.nan
        return out[shared + optional]

    return pd.concat([_align(fit_df), _align(oov_df)], ignore_index=True)


# ─── Decision boundary ────────────────────────────────────────────────────────


def _boundary(xlo: float, ylo: float, xhi: float, yhi: float) -> tuple[float, float]:
    """Return (slope k, intercept b) for line d_M = k·r + b."""
    k = (yhi - ylo) / (xhi - xlo)
    b = ylo - k * xlo
    return k, b


def _signed_dist(df: pd.DataFrame, k: float, b: float) -> pd.Series:
    """Positive = below the line (accepted), negative = above (rejected)."""
    return (k * df[COL_R] + b) - df[COL_M]


# ─── Plotting helpers ─────────────────────────────────────────────────────────


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  ✓  {path.name}")
    plt.close(fig)


def _origin_legend_handles() -> list:
    return [
        mpatches.Patch(facecolor=PALETTE[o], label=ORIGIN_LABELS[o], alpha=0.85)
        for o in ORIGIN_ORDER
    ]


def _quantile_xlim(series: pd.Series, margin: float = 0.05) -> tuple[float, float]:
    lo, hi = series.quantile(0.005), series.quantile(0.995)
    span = hi - lo
    return lo - margin * span, hi + margin * span


# ─── Figure 1 : Marginal distributions ───────────────────────────────────────


def _plot_marginal_ax(
    ax: plt.Axes,
    dfa: pd.DataFrame,
    col: str,
    xlabel: str,
    *,
    show_legend: bool = True,
    candidate_thrs: list[tuple[float, str]] | None = None,
) -> None:
    """Draw overlaid KDE + rug + optional threshold lines on *ax*."""
    for origin in ORIGIN_ORDER:
        sub = dfa.loc[dfa["origin"] == origin, col].dropna()
        if sub.empty:
            continue
        c = PALETTE[origin]
        lbl = ORIGIN_LABELS[origin]
        sns.kdeplot(
            sub,
            ax=ax,
            color=c,
            linewidth=2.0,
            fill=(origin == "from_fit"),
            alpha=0.12,
            label=lbl,
        )
        # Rug ticks (sparse for large from_fit)
        rng = np.random.default_rng(42)
        rug_vals = (
            sub.values
            if len(sub) <= 500
            else rng.choice(sub.values, 500, replace=False)
        )
        ax.plot(
            rug_vals,
            np.full(len(rug_vals), -0.003),
            "|",
            color=c,
            alpha=0.35,
            markersize=4,
            transform=ax.get_xaxis_transform(),
        )

    if candidate_thrs:
        for thr, label in candidate_thrs:
            ax.axvline(
                thr,
                color="#7D3C98",
                linestyle=":",
                linewidth=1.4,
                alpha=0.9,
                label=label,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if show_legend:
        ax.legend(frameon=False, fontsize=8)


def plot_marginals(
    dfa: pd.DataFrame,
    out_dir: Path,
    *,
    thr_residual: float | None = None,
    thr_mahal: float | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.suptitle(
        "Marginal anomaly-metric distributions by data origin",
        fontweight="bold",
        y=1.02,
    )

    _plot_marginal_ax(
        axes[0],
        dfa,
        COL_R,
        "PCA reconstruction residual  $r$",
        candidate_thrs=[(thr_residual, f"thr = {thr_residual:.2f}")]
        if thr_residual
        else None,
    )
    _plot_marginal_ax(
        axes[1],
        dfa,
        COL_M,
        "Mahalanobis distance  $d_M$",
        show_legend=False,
        candidate_thrs=[(thr_mahal, f"thr = {thr_mahal:.2f}")] if thr_mahal else None,
    )

    # Shared legend
    axes[1].legend(
        handles=_origin_legend_handles(),
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    plt.tight_layout()
    _save(fig, out_dir / "fig1_marginal_distributions.pdf")


# ─── Figure 2 : 2-D joint anomaly space ───────────────────────────────────────


def _kde_contours(
    ax: plt.Axes,
    df: pd.DataFrame,
    origin: str,
    *,
    levels: int = 5,
    thresh: float = 0.07,
    filled: bool = False,
) -> None:
    sub = df.loc[df["origin"] == origin, [COL_R, COL_M]].dropna()
    if len(sub) < 10:
        return
    linestyles = {"from_fit": "solid", "kb_match": "dashed", "oov": "dotted"}
    try:
        sns.kdeplot(
            data=sub,
            x=COL_R,
            y=COL_M,
            ax=ax,
            color=PALETTE[origin],
            levels=levels,
            thresh=thresh,
            fill=filled,
            alpha=0.10 if filled else 0.0,
            linewidths=1.6 if not filled else 1.0,
            linestyles=linestyles.get(origin, "solid"),
        )
    except Exception:
        pass


def plot_joint_2d(
    dfa: pd.DataFrame,
    out_dir: Path,
    k: float,
    b: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    # --- KDE contours: from_fit as filled reference ---
    _kde_contours(ax, dfa, "from_fit", filled=True)
    _kde_contours(ax, dfa, "kb_match")
    _kde_contours(ax, dfa, "oov")

    # --- Scatter for OOV points (manageable size) ---
    for origin in ["kb_match", "oov"]:
        sub = dfa.loc[dfa["origin"] == origin].dropna(subset=[COL_R, COL_M])
        if sub.empty:
            continue
        ax.scatter(
            sub[COL_R],
            sub[COL_M],
            c=PALETTE[origin],
            s=15,
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )

    # --- Decision boundary ---
    x_lo, x_hi = _quantile_xlim(dfa[COL_R].dropna())
    xr = np.linspace(x_lo, x_hi, 300)
    yr = k * xr + b
    ax.plot(
        xr,
        yr,
        color="#7D3C98",
        linewidth=2.0,
        linestyle="--",
        label=f"$d_M = {k:.1f}\\,r {b:+.1f}$",
        zorder=6,
    )
    y_ceil = dfa[COL_M].quantile(0.998) * 1.05
    ax.fill_between(xr, yr, y_ceil, alpha=0.07, color="#7D3C98", zorder=2)

    # Annotation text for zones
    ax.text(
        0.97,
        0.97,
        "Rejection zone",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#7D3C98",
        alpha=0.75,
        fontstyle="italic",
    )
    ax.text(
        0.03,
        0.05,
        "Accepted zone",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#1A5276",
        alpha=0.75,
        fontstyle="italic",
    )

    # --- Legend ---
    fit_line = mlines.Line2D(
        [],
        [],
        color=PALETTE["from_fit"],
        linewidth=1.5,
        linestyle="solid",
        label=ORIGIN_LABELS["from_fit"],
    )
    kb_line = mlines.Line2D(
        [],
        [],
        color=PALETTE["kb_match"],
        linewidth=1.5,
        linestyle="dashed",
        label=ORIGIN_LABELS["kb_match"],
    )
    oov_dot = mlines.Line2D(
        [],
        [],
        color=PALETTE["oov"],
        marker="o",
        markersize=5,
        linestyle="None",
        label=ORIGIN_LABELS["oov"],
    )
    bdy_line = mlines.Line2D(
        [],
        [],
        color="#7D3C98",
        linewidth=2.0,
        linestyle="--",
        label=f"Boundary: $d_M = {k:.1f}r {b:+.1f}$",
    )
    ax.legend(
        handles=[fit_line, kb_line, oov_dot, bdy_line],
        frameon=False,
        loc="upper left",
        fontsize=8,
    )

    ax.set_xlabel("PCA residual  $r$")
    ax.set_ylabel("Mahalanobis distance  $d_M$")
    ax.set_title("Joint anomaly space: training vs. OOV mentions", fontweight="bold")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save(fig, out_dir / "fig2_joint_anomaly_space.pdf")


# ─── Figure 3 : ROC and PR curves ─────────────────────────────────────────────


def _binary_labels(dfa: pd.DataFrame) -> np.ndarray:
    """1 = confirmed entity (from_fit or kb_match), 0 = unconfirmed OOV."""
    return dfa["origin"].isin(["from_fit", "kb_match"]).astype(int).values


def _best_f1_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float, float, float]:
    """Return (threshold, precision, recall, f1) at max-F1 operating point."""
    precision, recall, thrs = precision_recall_curve(y_true, y_score)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    idx = np.argmax(f1[:-1])  # last element has no threshold
    return float(thrs[idx]), float(precision[idx]), float(recall[idx]), float(f1[idx])


def plot_threshold_curves(
    dfa: pd.DataFrame,
    out_dir: Path,
    k: float,
    b: float,
) -> dict[str, float]:
    """Return recommended thresholds keyed by metric name."""
    y_true = _binary_labels(dfa)
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        print("[warn] single-class data – ROC/PR skipped", file=sys.stderr)
        return {}

    # Score convention: higher → more likely to be a confirmed entity (positive).
    # For anomaly metrics (lower = better), we negate.
    metrics: list[tuple[str, np.ndarray, str]] = [
        ("PCA residual $r$", -dfa[COL_R].fillna(dfa[COL_R].max()).values, "#2980B9"),
        ("Mahalanobis $d_M$", -dfa[COL_M].fillna(dfa[COL_M].max()).values, "#27AE60"),
        (
            "Boundary distance $\\delta$",
            _signed_dist(dfa, k, b).fillna(-1e6).values,
            "#7D3C98",
        ),
    ]
    if dfa[COL_Z].notna().any():
        metrics.append(
            ("Max-$z$ score", -dfa[COL_Z].fillna(dfa[COL_Z].max()).values, "#D35400"),
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle(
        "Threshold selection: ROC and Precision–Recall curves",
        fontweight="bold",
        y=1.02,
    )

    rec_thrs: dict[str, float] = {}

    for label, y_score, color in metrics:
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(
            fpr, tpr, color=color, lw=2.0, label=f"{label}  (AUC = {roc_auc:.3f})"
        )

        # PR
        prec, rec, thrs_pr = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        axes[1].plot(rec, prec, color=color, lw=2.0, label=f"{label}  (AP = {ap:.3f})")

        # Optimal F1 operating point on ROC
        thr_best, p_best, r_best, f1_best = _best_f1_threshold(y_true, y_score)
        rec_thrs[label] = thr_best
        fpr_op = ((y_score < thr_best) & (y_true == 0)).sum() / max(
            (y_true == 0).sum(), 1
        )
        tpr_op = ((y_score >= thr_best) & (y_true == 1)).sum() / max(
            (y_true == 1).sum(), 1
        )
        axes[0].scatter([fpr_op], [tpr_op], color=color, s=55, zorder=5, marker="D")
        axes[1].scatter(
            [r_best],
            [p_best],
            color=color,
            s=55,
            zorder=5,
            marker="D",
            label=f"_F1={f1_best:.2f}",
        )

    # Chance line
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve  (◆ = max-F1 point)")
    axes[0].legend(frameon=False, fontsize=8)

    # Baseline for PR: fraction of positives
    baseline = y_true.mean()
    axes[1].axhline(
        baseline,
        color="gray",
        linestyle="--",
        lw=0.8,
        alpha=0.5,
        label=f"Chance ({baseline:.2f})",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall  (◆ = max-F1 point)")
    axes[1].legend(frameon=False, fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir / "fig3_roc_pr_curves.pdf")
    return rec_thrs


# ─── Figure 4 : Boundary-offset sweep ────────────────────────────────────────


def plot_operating_points(
    dfa: pd.DataFrame,
    out_dir: Path,
    k: float,
    b: float,
) -> float:
    """Sweep the linear boundary offset b; return b* at maximum F1."""
    y_true = _binary_labels(dfa)
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return b

    b_vals = np.linspace(b - 10.0, b + 6.0, 120)
    precs, recs, f1s = [], [], []
    for b_t in b_vals:
        sd = _signed_dist(dfa, k, b_t).values
        y_pred = (sd >= 0).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precs.append(p)
        recs.append(r)
        f1s.append(f)

    best_idx = int(np.argmax(f1s))
    b_star = float(b_vals[best_idx])

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(b_vals, precs, color="#2980B9", lw=2.0, label="Precision")
    ax.plot(b_vals, recs, color="#27AE60", lw=2.0, label="Recall")
    ax.plot(b_vals, f1s, color="#A93226", lw=2.2, label="F1")

    ax.axvline(
        b_star,
        color="#7D3C98",
        linestyle="--",
        lw=1.6,
        label=f"$b^*={b_star:.2f}$  (F1={f1s[best_idx]:.3f})",
    )
    ax.axvline(
        b,
        color="#888888",
        linestyle=":",
        lw=1.2,
        alpha=0.8,
        label=f"Current  $b={b:.2f}$",
    )

    # Shade the difference between current and optimal
    ax.axvspan(min(b, b_star), max(b, b_star), alpha=0.06, color="#7D3C98")

    ax.set_xlabel("Boundary intercept  $b$  (in  $d_M = k \\cdot r + b$)")
    ax.set_ylabel("Score")
    ax.set_title(
        "Precision / Recall / F1 vs.  boundary intercept  $b$",
        fontweight="bold",
    )
    ax.set_ylim(0, 1.06)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(frameon=False, fontsize=8, ncol=2)

    plt.tight_layout()
    _save(fig, out_dir / "fig4_operating_points.pdf")

    print(
        f"\n  Optimal boundary:  k = {k:.3f},  b* = {b_star:.3f}"
        f"\n  Precision = {precs[best_idx]:.3f}   "
        f"Recall = {recs[best_idx]:.3f}   "
        f"F1 = {f1s[best_idx]:.3f}"
    )
    return b_star


# ─── Figure 5 : Score vs. anomaly metrics ─────────────────────────────────────


def plot_score_anomaly(dfa: pd.DataFrame, out_dir: Path) -> None:
    sub = dfa.loc[dfa["origin"].isin(["oov", "kb_match"])].dropna(
        subset=[COL_R, COL_M, "score"]
    )
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(
        "Linker membership score vs. anomaly metrics  (OOV mentions)",
        fontweight="bold",
        y=1.02,
    )

    for ax, col, xlabel in [
        (axes[0], COL_R, "PCA residual  $r$"),
        (axes[1], COL_M, "Mahalanobis distance  $d_M$"),
    ]:
        for origin in ["kb_match", "oov"]:
            s = sub.loc[sub["origin"] == origin]
            if s.empty:
                continue
            ax.scatter(
                s["score"],
                s[col],
                c=PALETTE[origin],
                s=14,
                alpha=0.5,
                linewidths=0,
                label=ORIGIN_LABELS[origin],
                zorder=3,
            )

        # Hexbin density estimate for dense regions
        try:
            ax.hexbin(
                sub["score"],
                sub[col],
                gridsize=30,
                cmap="Greys",
                alpha=0.15,
                zorder=1,
                mincnt=3,
            )
        except Exception:
            pass

        ax.set_xlabel("Cluster membership score")
        ax.set_ylabel(xlabel)
        ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir / "fig5_score_vs_anomaly.pdf")


# ─── Figure 6 : Decision map (accepted / rejected) ────────────────────────────


def plot_decision_map(
    dfa: pd.DataFrame,
    out_dir: Path,
    k: float,
    b: float,
) -> None:
    oov = (
        dfa.loc[dfa["origin"].isin(["oov", "kb_match"])]
        .dropna(subset=[COL_R, COL_M])
        .copy()
    )
    if oov.empty:
        return

    oov["accepted"] = _signed_dist(oov, k, b) >= 0

    n_acc = oov["accepted"].sum()
    n_rej = (~oov["accepted"]).sum()

    fig, ax = plt.subplots(figsize=(6.8, 5.5))

    # from_fit reference (filled KDE)
    fit = dfa.loc[dfa["origin"] == "from_fit", [COL_R, COL_M]].dropna()
    if len(fit) >= 10:
        try:
            sns.kdeplot(
                data=fit,
                x=COL_R,
                y=COL_M,
                ax=ax,
                color=PALETTE["from_fit"],
                levels=5,
                thresh=0.07,
                fill=True,
                alpha=0.10,
                linewidths=1.0,
            )
        except Exception:
            pass

    # OOV: scatter, colour by decision
    _dec_cfg = [
        (True, "o", "#27AE60", "#1E5631", "Accepted OOV"),
        (False, "X", "#E74C3C", "#7B241C", "Rejected OOV"),
    ]
    for accepted, marker, fc, ec, label in _dec_cfg:
        s = oov.loc[oov["accepted"] == accepted]
        if s.empty:
            continue
        ax.scatter(
            s[COL_R],
            s[COL_M],
            c=fc,
            marker=marker,
            s=18,
            alpha=0.65,
            edgecolors=ec,
            linewidths=0.4,
            label=label,
            zorder=4,
        )

    # KB-validated ring overlay
    kb = oov.loc[oov["origin"] == "kb_match"]
    if not kb.empty:
        ax.scatter(
            kb[COL_R],
            kb[COL_M],
            facecolors="none",
            edgecolors="#F39C12",
            s=60,
            linewidths=1.5,
            zorder=5,
            label=f"KB validated  (n={len(kb)})",
        )

    # Decision boundary + shaded rejection region
    x_lo, x_hi = _quantile_xlim(dfa[COL_R].dropna())
    xr = np.linspace(x_lo, x_hi, 300)
    yr = k * xr + b
    y_ceil = dfa[COL_M].quantile(0.998) * 1.05
    ax.fill_between(xr, yr, y_ceil, alpha=0.07, color="#7D3C98", zorder=2)
    ax.plot(
        xr,
        yr,
        color="#7D3C98",
        linewidth=2.2,
        linestyle="--",
        label=f"Boundary  $d_M = {k:.1f}r {b:+.1f}$",
        zorder=6,
    )

    # Count annotations
    ax.text(
        0.97,
        0.96,
        f"Rejected: {n_rej}  ({100 * n_rej / (n_acc + n_rej):.0f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#7B241C",
    )
    ax.text(
        0.03,
        0.05,
        f"Accepted: {n_acc}  ({100 * n_acc / (n_acc + n_rej):.0f}%)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#1E5631",
    )

    fit_proxy = mlines.Line2D(
        [],
        [],
        color=PALETTE["from_fit"],
        linewidth=1.5,
        linestyle="solid",
        label=ORIGIN_LABELS["from_fit"],
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [fit_proxy] + handles,
        [ORIGIN_LABELS["from_fit"]] + labels,
        frameon=False,
        loc="upper left",
        fontsize=8,
    )

    ax.set_xlabel("PCA residual  $r$")
    ax.set_ylabel("Mahalanobis distance  $d_M$")
    ax.set_title("OOV decision map in anomaly space", fontweight="bold")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save(fig, out_dir / "fig6_decision_map.pdf")


# ─── Figure 7 : Screener-anomaly alignment ────────────────────────────────────


def plot_screener_alignment(
    dfa: pd.DataFrame,
    out_dir: Path,
    k: float,
    b: float,
) -> None:
    """Compare screener margin with anomaly metrics for OOV mentions."""
    sub = dfa.loc[
        dfa["origin"].isin(["oov", "kb_match"]) & dfa["screener_decision"].notna()
    ].copy()
    if sub.empty:
        return

    sub["boundary_dist"] = _signed_dist(sub, k, b)
    sub["screener_pos"] = sub["screener_decision"] >= 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(
        "Screener margin vs. anomaly metrics  (OOV mentions)",
        fontweight="bold",
        y=1.02,
    )

    for ax, col, xlabel in [
        (axes[0], COL_R, "PCA residual  $r$"),
        (axes[1], "boundary_dist", "Boundary signed distance  $\\delta$"),
    ]:
        for origin in ["kb_match", "oov"]:
            s = sub.loc[sub["origin"] == origin].dropna(
                subset=[col, "screener_decision"]
            )
            if s.empty:
                continue
            ax.scatter(
                s[col],
                s["screener_decision"],
                c=PALETTE[origin],
                s=14,
                alpha=0.5,
                linewidths=0,
                label=ORIGIN_LABELS[origin],
                zorder=3,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(
            0
            if col == "boundary_dist"
            else dfa.loc[dfa["origin"] == "from_fit", COL_R].quantile(0.95),
            color="gray",
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Screener margin  (pos. → accepted)")
        ax.legend(frameon=False, fontsize=8)

    # Compute Spearman correlation for boundary distance
    valid = sub.dropna(subset=["boundary_dist", "screener_decision"])
    if len(valid) > 5:
        rho, pval = spearmanr(valid["boundary_dist"], valid["screener_decision"])
        axes[1].set_title(
            f"Boundary distance vs. screener  ($\\rho={rho:.2f}$, $p={pval:.2e}$)",
            fontsize=9,
        )

    plt.tight_layout()
    _save(fig, out_dir / "fig7_screener_alignment.pdf")


# ─── Figure 8 : Composite paper figure (2×2) ─────────────────────────────────


def plot_composite(
    dfa: pd.DataFrame,
    out_dir: Path,
    k: float,
    b: float,
    b_star: float,
) -> None:
    """A single 2×2 figure suitable for direct inclusion in a paper."""
    fig = plt.figure(figsize=(11, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    ax_joint = fig.add_subplot(gs[0, 0])
    ax_marg = fig.add_subplot(gs[0, 1])
    ax_roc = fig.add_subplot(gs[1, 0])
    ax_sweep = fig.add_subplot(gs[1, 1])

    # ── Panel A: 2-D joint density ─────────────────────────────────────
    for origin in ORIGIN_ORDER:
        _kde_contours(ax_joint, dfa, origin, filled=(origin == "from_fit"))

    for origin in ["kb_match", "oov"]:
        s = dfa.loc[dfa["origin"] == origin].dropna(subset=[COL_R, COL_M])
        if not s.empty:
            ax_joint.scatter(
                s[COL_R],
                s[COL_M],
                c=PALETTE[origin],
                s=10,
                alpha=0.45,
                linewidths=0,
                zorder=3,
            )

    x_lo, x_hi = _quantile_xlim(dfa[COL_R].dropna())
    xr = np.linspace(x_lo, x_hi, 200)
    ax_joint.plot(
        xr,
        k * xr + b,
        color="#7D3C98",
        lw=1.8,
        linestyle="--",
        label=f"$d_M={k:.0f}r{b:+.0f}$",
        zorder=5,
    )
    ax_joint.fill_between(
        xr, k * xr + b, dfa[COL_M].quantile(0.998) * 1.05, alpha=0.07, color="#7D3C98"
    )
    ax_joint.set_xlabel("PCA residual  $r$", fontsize=9)
    ax_joint.set_ylabel("Mahalanobis  $d_M$", fontsize=9)
    ax_joint.set_title("(A)  Joint anomaly space", fontsize=10, fontweight="bold")
    ax_joint.legend(frameon=False, fontsize=7, loc="upper left")
    ax_joint.set_xlim(x_lo, x_hi)
    ax_joint.set_ylim(bottom=0)

    # ── Panel B: marginal KDEs ─────────────────────────────────────────
    for origin in ORIGIN_ORDER:
        s = dfa.loc[dfa["origin"] == origin, COL_R].dropna()
        if s.empty:
            continue
        sns.kdeplot(
            s,
            ax=ax_marg,
            color=PALETTE[origin],
            linewidth=1.8,
            fill=(origin == "from_fit"),
            alpha=0.12,
            label=ORIGIN_LABELS[origin],
        )
    ax_marg.set_xlabel("PCA residual  $r$", fontsize=9)
    ax_marg.set_ylabel("Density", fontsize=9)
    ax_marg.set_title("(B)  Marginal distributions", fontsize=10, fontweight="bold")
    ax_marg.legend(frameon=False, fontsize=7)

    # ── Panel C: ROC curves ────────────────────────────────────────────
    y_true = _binary_labels(dfa)
    if y_true.sum() > 0 and (1 - y_true).sum() > 0:
        score_configs = [
            ("$r$", -dfa[COL_R].fillna(dfa[COL_R].max()).values, "#2980B9"),
            ("$d_M$", -dfa[COL_M].fillna(dfa[COL_M].max()).values, "#27AE60"),
            ("$\\delta$", _signed_dist(dfa, k, b).fillna(-1e6).values, "#7D3C98"),
        ]
        for lbl, ys, col in score_configs:
            fpr, tpr, _ = roc_curve(y_true, ys)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(
                fpr, tpr, color=col, lw=1.8, label=f"{lbl}  (AUC={roc_auc:.2f})"
            )
        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
    ax_roc.set_xlabel("FPR", fontsize=9)
    ax_roc.set_ylabel("TPR", fontsize=9)
    ax_roc.set_title("(C)  ROC curves", fontsize=10, fontweight="bold")
    ax_roc.legend(frameon=False, fontsize=7)

    # ── Panel D: boundary-offset sweep ────────────────────────────────
    if y_true.sum() > 0 and (1 - y_true).sum() > 0:
        b_vals = np.linspace(b - 10.0, b + 6.0, 120)
        f1s, precs, recs = [], [], []
        for b_t in b_vals:
            sd = _signed_dist(dfa, k, b_t).values
            yp = (sd >= 0).astype(int)
            tp = ((yp == 1) & (y_true == 1)).sum()
            fp = ((yp == 1) & (y_true == 0)).sum()
            fn = ((yp == 0) & (y_true == 1)).sum()
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            precs.append(p)
            recs.append(r)
            f1s.append(f)

        ax_sweep.plot(b_vals, precs, color="#2980B9", lw=1.8, label="Precision")
        ax_sweep.plot(b_vals, recs, color="#27AE60", lw=1.8, label="Recall")
        ax_sweep.plot(b_vals, f1s, color="#A93226", lw=2.0, label="F1")
        ax_sweep.axvline(
            b_star, color="#7D3C98", linestyle="--", lw=1.4, label=f"$b^*={b_star:.1f}$"
        )
        ax_sweep.axvline(
            b, color="#888", linestyle=":", lw=1.0, alpha=0.7, label=f"init $b={b:.1f}$"
        )
        ax_sweep.set_ylim(0, 1.06)
        ax_sweep.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    ax_sweep.set_xlabel("Boundary intercept  $b$", fontsize=9)
    ax_sweep.set_ylabel("Score", fontsize=9)
    ax_sweep.set_title("(D)  F1 vs. boundary intercept", fontsize=10, fontweight="bold")
    ax_sweep.legend(frameon=False, fontsize=7, ncol=2)

    fig.suptitle(
        "Pre-classifier anomaly filter: separability and threshold selection",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    _save(fig, out_dir / "fig0_composite_paper.pdf")


# ─── Summary statistics ───────────────────────────────────────────────────────


def _print_summary(dfa: pd.DataFrame, k: float, b: float) -> None:
    print("\n" + "=" * 62)
    print("  Dataset summary")
    print("=" * 62)
    for origin in ORIGIN_ORDER:
        n = (dfa["origin"] == origin).sum()
        sub = dfa.loc[dfa["origin"] == origin]
        r_med = sub[COL_R].median()
        m_med = sub[COL_M].median()
        print(f"  {ORIGIN_LABELS[origin]:<28} n={n:>5}  r̃={r_med:.3f}  d̃_M={m_med:.3f}")

    print("\n" + "─" * 62)
    print("  Boundary statistics  (k={k:.3f}, b={b:.3f})".format(k=k, b=b))
    for origin in ORIGIN_ORDER:
        sub = dfa.loc[dfa["origin"] == origin].dropna(subset=[COL_R, COL_M])
        if sub.empty:
            continue
        below = (_signed_dist(sub, k, b) >= 0).mean()
        print(f"  {ORIGIN_LABELS[origin]:<28} accepted: {below * 100:.1f}%")
    print("=" * 62 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--fit-report",
        type=Path,
        # default=Path(
        #     "../../../pelinker/reports/fit-b/linker_fit.clustering_report.json.gz"
        # ),
        help="Fit-B clustering report (json.gz).",
    )
    p.add_argument(
        "--oov-csv",
        type=Path,
        # default=Path("../../../pelinker/reports/current/oov.csv"),
        help="OOV mention dump from  link_files --dump-mention-anomaly.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/alexander/data/pelinker/figs/paper"),
        help="Output directory for PDF figures.",
    )
    # Linear decision boundary, defined by two points in (r, d_M) space
    p.add_argument(
        "--x-lo", type=float, default=0.30, help="r-coordinate of the low anchor point."
    )
    p.add_argument(
        "--y-lo",
        type=float,
        default=6.0,
        help="d_M-coordinate of the low anchor point.",
    )
    p.add_argument(
        "--x-hi",
        type=float,
        default=0.42,
        help="r-coordinate of the high anchor point.",
    )
    p.add_argument(
        "--y-hi",
        type=float,
        default=10.0,
        help="d_M-coordinate of the high anchor point.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_style()

    print("Loading data …")
    fit_df = _load_fit_report(args.fit_report)
    oov_df = _load_oov(args.oov_csv)
    dfa = _assemble(fit_df, oov_df)

    for origin in ORIGIN_ORDER:
        print(
            f"  {ORIGIN_LABELS[origin]:<30} {(dfa['origin'] == origin).sum():>5} rows"
        )

    k, b = _boundary(args.x_lo, args.y_lo, args.x_hi, args.y_hi)
    print(f"\nDecision boundary:  d_M = {k:.3f}·r + {b:.3f}")
    _print_summary(dfa, k, b)

    out = args.out_dir
    print("Generating figures …")

    # Individual figures
    plot_marginals(dfa, out)
    plot_joint_2d(dfa, out, k, b)
    plot_threshold_curves(dfa, out, k, b)
    b_star = plot_operating_points(dfa, out, k, b)
    plot_score_anomaly(dfa, out)
    plot_decision_map(dfa, out, k, b)
    plot_screener_alignment(dfa, out, k, b)

    # Composite paper figure (uses b_star from the sweep)
    plot_composite(dfa, out, k, b, b_star)

    print("\nAll figures written to:", out)


if __name__ == "__main__":
    main()
