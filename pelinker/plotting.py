import colorsys
import pathlib

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

# Force a non-interactive backend because this project only saves plots to files.
# This avoids Tk/threading crashes in long-running or parallelized workflows.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Polygon, Rectangle, RegularPolygon, Wedge
from plotly import express as px, graph_objects as go

# Columns appended in ``run/analysis/model_selection.py`` grid export (per-sample summaries).
GRID_COL_CHOSEN_MIN_CLUSTER_SIZE = "chosen_min_cluster_size"
GRID_COL_SAMPLE_BEST_DBCV = "sample_best_dbcv"
GRID_COL_SAMPLE_ARI = "sample_ari"

# χ²(2) critical value at p≈0.95 for Gaussian 95% contour (no scipy).
_CHI2_PPF_95_DF2 = 5.991464550106692

# Arity → marker style: 1 = triangle, 2 = square (bimodal fill), 3 = circle (tricolor wedges).
ARITY_MARKER_SCATTER: dict[str, str] = {
    "singleton": "^",
    "fusion2": "s",
    "fusion3": "o",
}

_AXIS_MAX = 1.1
_MARKER_HALF_WIDTH = 0.028
_MIN_ELLIPSE_SEMI_AXIS = 0.018
# Uniform footprint: square side = 2 * _MARKER_HALF_WIDTH; circle uses same radius; triangle side matches.
_MARKER_FACE_ALPHA = 0.52
_MARKER_OUTLINE_ALPHA = 0.72
_ELLIPSE_FILL_ALPHA = 0.14
_ELLIPSE_EDGE_ALPHA = 0.88
_ELLIPSE_INFLATE = 1.12


def _arity_from_model(model: str) -> str:
    if model == "fusion2":
        return "fusion2"
    if model == "fusion3":
        return "fusion3"
    return "singleton"


def _base_models_in_row(model: str, layer: str) -> list[str]:
    """Constituent encoder model names (singleton: one; fusion: from ``a/L1+b/L2``)."""
    if model not in ("fusion2", "fusion3"):
        return [str(model)]
    names: list[str] = []
    for part in str(layer).split("+"):
        p = part.strip()
        if not p:
            continue
        m, _, _ = p.partition("/")
        if m:
            names.append(m)
    return names if names else [str(model)]


def _layer_spec_code(model: str, layer: str) -> str:
    """
    Short label for the layer configuration only (no encoder name): singleton ``layer`` as-is;
    fusion ``a/L1+b/L2`` → ``L1+L2``.
    """
    if model not in ("fusion2", "fusion3"):
        return str(layer)
    specs: list[str] = []
    for part in str(layer).split("+"):
        p = part.strip()
        if not p:
            continue
        _m, sep, lyr = p.partition("/")
        specs.append(lyr if sep else p)
    return ",".join(specs) if specs else str(layer)


def _model_color_map(unique_models: list[str]) -> dict[str, str]:
    models = sorted(set(unique_models))
    palette = sns.color_palette("husl", n_colors=max(len(models), 1))
    return {m: matplotlib.colors.to_hex(palette[i]) for i, m in enumerate(models)}


def _rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    return matplotlib.colors.to_rgba(hex_color, alpha=alpha)


def _merge_wedge_spans(
    spans: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Merge adjacent angular spans that share the same face color."""
    if not spans:
        return []
    merged: list[tuple[float, float, str]] = []
    t1, t2, c = spans[0]
    for t1n, t2n, cn in spans[1:]:
        if cn == c and t1n == t2:
            t2 = t2n
        else:
            merged.append((t1, t2, c))
            t1, t2, c = t1n, t2n, cn
    merged.append((t1, t2, c))
    return merged


def _covariance_ellipse_95(
    cov: np.ndarray,
    *,
    min_semi_axis: float = _MIN_ELLIPSE_SEMI_AXIS,
) -> tuple[float, float, float] | None:
    """
    Return (width, height, angle_deg) for ~95% Gaussian ellipse.
    Floors semi-axes so nearly-degenerate covariances stay visible.
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.shape != (2, 2):
        return None
    vals, vecs = np.linalg.eigh(c)
    if np.any(vals < -1e-12):
        return None
    vals = np.maximum(vals, 0.0)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    semi_major = max(float(np.sqrt(_CHI2_PPF_95_DF2 * vals[0])), min_semi_axis)
    semi_minor = max(float(np.sqrt(_CHI2_PPF_95_DF2 * vals[1])), min_semi_axis)
    width = 2.0 * semi_major
    height = 2.0 * semi_minor
    theta = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    if not np.isfinite(width) or not np.isfinite(height):
        return None
    return width, height, theta


def _draw_arity_marker(
    ax: plt.Axes,
    mx: float,
    my: float,
    *,
    arity: str,
    models: list[str],
    color_by_model: dict[str, str],
    zorder: int = 5,
) -> None:
    """Draw △ / □ / ○ in data coords; transparent fills; no internal seams for same-color adjacency."""
    h = _MARKER_HALF_WIDTH
    h_sq = 0.8 * h
    fa = _MARKER_FACE_ALPHA
    oa = _MARKER_OUTLINE_ALPHA
    edge_rgba = (0.0, 0.0, 0.0, oa)

    if arity == "singleton":
        col = color_by_model.get(models[0], "#333333")
        side = 2.0 * h
        radius = float(side / np.sqrt(3.0))
        tri = RegularPolygon(
            (mx, my),
            numVertices=3,
            radius=radius,
            orientation=np.pi / 2.0,
            facecolor=_rgba(col, fa),
            edgecolor=edge_rgba,
            linewidth=0.85,
            zorder=zorder,
        )
        ax.add_patch(tri)
        return

    if arity == "fusion2" and len(models) >= 2:
        c0 = color_by_model.get(models[0], "#888888")
        c1 = color_by_model.get(models[1], "#444444")
        if c0 == c1:
            sq = Polygon(
                [
                    (mx - h_sq, my - h_sq),
                    (mx - h_sq, my + h_sq),
                    (mx + h_sq, my + h_sq),
                    (mx + h_sq, my - h_sq),
                ],
                closed=True,
                facecolor=_rgba(c0, fa),
                edgecolor=edge_rgba,
                linewidth=0.85,
                zorder=zorder,
            )
            ax.add_patch(sq)
            return
        left = Polygon(
            [
                (mx - h_sq, my - h_sq),
                (mx - h_sq, my + h_sq),
                (mx, my + h_sq),
                (mx, my - h_sq),
            ],
            closed=True,
            facecolor=_rgba(c0, fa),
            linewidth=0,
            zorder=zorder,
        )
        right = Polygon(
            [
                (mx, my - h_sq),
                (mx, my + h_sq),
                (mx + h_sq, my + h_sq),
                (mx + h_sq, my - h_sq),
            ],
            closed=True,
            facecolor=_rgba(c1, fa),
            linewidth=0,
            zorder=zorder,
        )
        ax.add_patch(left)
        ax.add_patch(right)
        ax.add_patch(
            Rectangle(
                (mx - h_sq, my - h_sq),
                2 * h_sq,
                2 * h_sq,
                fill=False,
                linewidth=0.9,
                edgecolor=edge_rgba,
                zorder=zorder + 1,
            )
        )
        return

    if arity == "fusion3" and len(models) >= 3:
        r = h
        base_spans = [(90.0, 210.0), (210.0, 330.0), (330.0, 450.0)]
        colored = [
            (t1, t2, color_by_model.get(models[i], "#666666"))
            for i, (t1, t2) in enumerate(base_spans)
        ]
        for t1, t2, col in _merge_wedge_spans(colored):
            w = Wedge(
                (mx, my),
                r,
                t1,
                t2,
                facecolor=_rgba(col, fa),
                linewidth=0,
                zorder=zorder,
            )
            ax.add_patch(w)
        ax.add_patch(
            Ellipse(
                (mx, my),
                width=2 * r,
                height=2 * r,
                facecolor="none",
                edgecolor=edge_rgba,
                linewidth=0.9,
                zorder=zorder + 1,
            )
        )
        return

    # Fallback: solid patch with arity-appropriate shape
    mkey = models[0] if models else ""
    col = color_by_model.get(mkey, "#333333")
    if arity == "fusion2":
        fb: Polygon | Ellipse = Polygon(
            [
                (mx - h, my - h),
                (mx - h, my + h),
                (mx + h, my + h),
                (mx + h, my - h),
            ],
            closed=True,
            facecolor=_rgba(col, fa),
            edgecolor=edge_rgba,
            linewidth=0.9,
            zorder=zorder,
        )
    elif arity == "fusion3":
        fb = Ellipse(
            (mx, my),
            width=2 * h,
            height=2 * h,
            facecolor=_rgba(col, fa),
            edgecolor=edge_rgba,
            linewidth=0.9,
            zorder=zorder,
        )
    else:
        side = 2.0 * h
        radius = float(side / np.sqrt(3.0))
        fb = RegularPolygon(
            (mx, my),
            numVertices=3,
            radius=radius,
            orientation=np.pi / 2.0,
            facecolor=_rgba(col, fa),
            edgecolor=edge_rgba,
            linewidth=0.85,
            zorder=zorder,
        )
    ax.add_patch(fb)


def plot_dbcv_vs_ari_from_grid(
    df_grid: pd.DataFrame,
    output_path: pathlib.Path,
) -> bool:
    """
    Scatter of mean DBCV vs mean ARI per (model, layer); shape = arity (△/□/○),
    fill colors = base encoder model(s); text = layer spec only (e.g. fusion ``2+3``).
    95% covariance ellipses when ``n_sample`` ≥ 2.

    Expects ``sample_best_dbcv``, ``sample_ari`` on the grid export.
    Both axes are fixed to ``[0, _AXIS_MAX]``.

    Returns:
        True if a figure was written, False if required data were absent.
    """
    needed = {
        "model",
        "layer",
        "sample_idx",
        GRID_COL_SAMPLE_BEST_DBCV,
        GRID_COL_SAMPLE_ARI,
    }
    if not needed.issubset(df_grid.columns):
        return False

    df = df_grid.loc[
        :,
        [
            "model",
            "layer",
            "sample_idx",
            GRID_COL_SAMPLE_BEST_DBCV,
            GRID_COL_SAMPLE_ARI,
        ],
    ].drop_duplicates(subset=["model", "layer", "sample_idx"], keep="first")
    df = df[df[GRID_COL_SAMPLE_ARI].notna()].copy()
    if df.empty:
        return False

    all_models: list[str] = []
    for m, lyr in df[["model", "layer"]].drop_duplicates().itertuples(index=False):
        all_models.extend(_base_models_in_row(str(m), str(lyr)))
    color_by_model = _model_color_map(all_models)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    arities_present: set[str] = set()

    for (model, layer), g in df.groupby(["model", "layer"], sort=False):
        xy = np.column_stack(
            [
                g[GRID_COL_SAMPLE_BEST_DBCV].to_numpy(dtype=np.float64),
                g[GRID_COL_SAMPLE_ARI].to_numpy(dtype=np.float64),
            ]
        )
        mean = xy.mean(axis=0)
        n = xy.shape[0]
        arity = _arity_from_model(str(model))
        arities_present.add(arity)
        models_row = _base_models_in_row(str(model), str(layer))

        # Halo drawn first: light filled ellipse + clear dashed rim, slightly inflated so
        # it remains visible through transparent markers.
        if n >= 2:
            cov = np.cov(xy, rowvar=False, ddof=1)
            ell = _covariance_ellipse_95(cov)
            if ell is not None:
                w, h, ang = ell
                wi, hi = w * _ELLIPSE_INFLATE, h * _ELLIPSE_INFLATE
                patch = Ellipse(
                    xy=(float(mean[0]), float(mean[1])),
                    width=wi,
                    height=hi,
                    angle=ang,
                    facecolor=(0.45, 0.48, 0.52, _ELLIPSE_FILL_ALPHA),
                    edgecolor=(0.12, 0.14, 0.18, _ELLIPSE_EDGE_ALPHA),
                    linewidth=1.45,
                    linestyle=(0, (4.5, 3.0)),
                    zorder=2,
                )
                ax.add_patch(patch)

        _draw_arity_marker(
            ax,
            float(mean[0]),
            float(mean[1]),
            arity=arity,
            models=models_row,
            color_by_model=color_by_model,
            zorder=5,
        )
        layer_code = _layer_spec_code(str(model), str(layer))
        if len(layer_code) > 22:
            layer_code = layer_code[:19] + "…"
        ax.annotate(
            layer_code,
            (mean[0], mean[1]),
            textcoords="offset points",
            # xytext=(8, 8),
            xytext=(-3 - 1.5 * (len(layer_code) - 1), -3),
            fontsize=8,
            alpha=0.88,
            zorder=6,
        )

    ax.set_xlim(0.0, _AXIS_MAX)
    ax.set_ylim(0.0, _AXIS_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("DBCV (per-sample best, mean over samples)")
    ax.set_ylabel("Adjusted Rand Index (per-sample, mean over samples)")
    ax.set_title("DBCV vs ARI; dashed ellipse ≈95% (n_sample >= 2)")

    order_a = ["singleton", "fusion2", "fusion3"]
    arity_labels = {
        "singleton": "singleton",
        "fusion2": "pair fusion",
        "fusion3": "triple fusion",
    }
    edge_legend = (0.0, 0.0, 0.0, _MARKER_OUTLINE_ALPHA)
    legend_shapes = [
        Line2D(
            [0],
            [0],
            marker=ARITY_MARKER_SCATTER[a],
            color="none",
            label=arity_labels[a],
            markerfacecolor=_rgba("#bbbbbb", _MARKER_FACE_ALPHA),
            markeredgecolor=edge_legend,
            markersize=10,
        )
        for a in order_a
        if a in arities_present
    ]
    legend_colors = [
        Patch(
            facecolor=_rgba(color_by_model[m], _MARKER_FACE_ALPHA),
            edgecolor=edge_legend,
            linewidth=0.5,
            label=m,
        )
        for m in sorted(color_by_model.keys())
    ]
    if legend_shapes or legend_colors:
        leg1 = ax.legend(
            handles=legend_shapes,
            title="Arity",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=legend_colors,
            title="Base model",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
            frameon=True,
        )

    ax.grid(True, alpha=0.28, linestyle="--", zorder=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_metrics_with_error_bars(
    metrics_list: list[pd.DataFrame],
    output_path: pathlib.Path,
    *,
    chosen_min_cluster_size: float | None = None,
):
    """
    Plot metrics across multiple runs with error bars using seaborn lineplot.

    Args:
        metrics_list: List of DataFrames, each with columns: min_cluster_size, icm, n_clusters, dbcv, ari
        output_path: Path to save the figure
        chosen_min_cluster_size: Optional vertical marker for the selected grid value (e.g. from smoother / argmax).
    """
    # Combine all metrics DataFrames, adding a run_id column
    combined_metrics = []
    for run_id, df in enumerate(metrics_list):
        df_copy = df.copy()
        df_copy["run_id"] = run_id
        combined_metrics.append(df_copy)

    df_combined = pd.concat(combined_metrics, ignore_index=True)

    # Filter out trivial points where n_clusters <= 1
    df_combined = df_combined[df_combined["n_clusters"] > 1].copy()

    if len(df_combined) == 0:
        print(
            f"Warning: No valid data points after filtering (n_clusters > 1) for {output_path}"
        )
        return

    has_ari = "ari" in df_combined.columns and bool(df_combined["ari"].notna().any())
    ncols = 3 if has_ari else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 2:
        ax_dbcv, ax_k = axes[0], axes[1]
        ax_ari = None
    else:
        ax_dbcv, ax_k, ax_ari = axes[0], axes[1], axes[2]

    # Color palette for different plots
    colors = ["#2E86AB", "#A23B72", "#F18F01"]  # Blue, Purple, Orange

    def _maybe_vline(ax) -> None:
        if chosen_min_cluster_size is None:
            return
        ax.axvline(
            chosen_min_cluster_size,
            color="0.35",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            zorder=0,
        )

    # Plot DBCV score with error bars
    sns.lineplot(
        data=df_combined,
        x="min_cluster_size",
        y="dbcv",
        ax=ax_dbcv,
        errorbar="sd",  # Standard deviation error bars
        marker="o",
        color=colors[0],
        linewidth=2,
        markersize=8,
        err_kws={"alpha": 0.3, "linewidth": 1.5},
    )
    _maybe_vline(ax_dbcv)
    ax_dbcv.set_xlabel("min_cluster_size", fontsize=12, fontweight="bold")
    ax_dbcv.set_ylabel("DBCV Score", fontsize=12, fontweight="bold", color=colors[0])
    ax_dbcv.set_title("DBCV Score vs. min_cluster_size", fontsize=13, fontweight="bold")
    ax_dbcv.grid(True, alpha=0.3, linestyle="--")
    ax_dbcv.tick_params(axis="y", labelcolor=colors[0])
    ax_dbcv.spines["top"].set_visible(False)
    ax_dbcv.spines["right"].set_visible(False)

    # Plot n_clusters with error bars (log scale)
    sns.lineplot(
        data=df_combined,
        x="min_cluster_size",
        y="n_clusters",
        ax=ax_k,
        errorbar="sd",
        marker="^",
        color=colors[1],
        linewidth=2,
        markersize=8,
        err_kws={"alpha": 0.3, "linewidth": 1.5},
    )
    _maybe_vline(ax_k)
    ax_k.set_xlabel("min_cluster_size", fontsize=12, fontweight="bold")
    ax_k.set_ylabel("n clusters", fontsize=12, fontweight="bold", color=colors[1])
    ax_k.set_title(
        "Number of Clusters vs. min_cluster_size", fontsize=13, fontweight="bold"
    )
    ax_k.grid(True, alpha=0.3, linestyle="--")
    ax_k.tick_params(axis="y", labelcolor=colors[1])
    ax_k.spines["top"].set_visible(False)
    ax_k.spines["right"].set_visible(False)

    if ax_ari is not None:
        sns.lineplot(
            data=df_combined,
            x="min_cluster_size",
            y="ari",
            ax=ax_ari,
            errorbar="sd",
            marker="D",
            color=colors[2],
            linewidth=2,
            markersize=7,
            err_kws={"alpha": 0.3, "linewidth": 1.5},
        )
        _maybe_vline(ax_ari)
        ax_ari.set_xlabel("min_cluster_size", fontsize=12, fontweight="bold")
        ax_ari.set_ylabel("ARI", fontsize=12, fontweight="bold", color=colors[2])
        ax_ari.set_title("ARI vs. min_cluster_size", fontsize=13, fontweight="bold")
        ax_ari.grid(True, alpha=0.3, linestyle="--")
        ax_ari.tick_params(axis="y", labelcolor=colors[2])
        ax_ari.spines["top"].set_visible(False)
        ax_ari.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(
    df_results: pd.DataFrame,
    output_path: pathlib.Path,
    metric: str = "best_score",
    metric_label: str | None = None,
):
    """
    Create a heatmap with model (rows) and layer (columns).
    Color represents the specified metric, text shows best_size and metric name.

    Args:
        df_results: DataFrame with columns: model, layer, best_size, and the metric column
        output_path: Path to save the heatmap figure
        metric: Column name for the metric to display as color (default: "best_score")
        metric_label: Label for the metric (default: uses metric column name)
    """
    if metric_label is None:
        metric_label = metric.replace("_", " ").title()

    # Create pivot tables
    score_pivot = df_results.pivot(index="model", columns="layer", values=metric)
    size_pivot = df_results.pivot(index="model", columns="layer", values="best_size")

    # Create figure
    fig, ax = plt.subplots(
        figsize=(
            max(8, len(score_pivot.columns) * 0.8),
            max(6, len(score_pivot.index) * 0.6),
        )
    )

    # Create heatmap with metric as color
    # Use RdBu_r (Red-Blue reversed) for clear visual distinction: red=high, blue=low
    sns.heatmap(
        score_pivot,
        annot=False,  # We'll add custom annotations
        fmt=".3f",
        cmap="RdBu_r",
        center=None,  # Center colormap at the median for better contrast
        cbar_kws={"label": metric_label, "shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        square=False,
    )

    # Add best_size and metric name as text annotations
    # Calculate mean score for text color threshold
    valid_scores = score_pivot.values[~pd.isna(score_pivot.values)]
    mean_score = valid_scores.mean() if len(valid_scores) > 0 else 0

    for i in range(len(score_pivot.index)):
        for j in range(len(score_pivot.columns)):
            score_val = score_pivot.iloc[i, j]
            size_val = size_pivot.iloc[i, j]

            if not pd.isna(score_val) and not pd.isna(size_val):
                # Use white text for darker cells (lower scores), black for lighter cells
                text_color = "white" if score_val < mean_score else "black"
                # Format metric value based on its magnitude
                if abs(score_val) < 0.01:
                    metric_str = f"{score_val:.2e}"
                elif abs(score_val) < 1:
                    metric_str = f"{score_val:.3f}"
                else:
                    metric_str = f"{score_val:.2f}"
                # Add text annotation with best_size and metric value
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{int(size_val)}\n{metric_str}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=8,
                    linespacing=1.2,
                )

    ax.set_title(f"Clustering Results: {metric_label} (color) and Best Size (text)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _sorted_class_labels_natural(class_series: pd.Series) -> list[str]:
    """Order legend / color map: numeric cluster ids sort numerically, not lexically."""
    labels = [str(x) for x in class_series.unique()]

    def sort_key(label: str) -> tuple[int, float] | tuple[int, str]:
        try:
            return (0, float(label))
        except ValueError:
            return (1, label)

    return sorted(labels, key=sort_key)


def _distinct_category_hex_colors(n: int) -> list[str]:
    """
    One distinct color per category for large palettes (e.g. ~50 HDBSCAN clusters).

    Uses golden-ratio hue steps with staggered saturation and value so neighbors
    stay distinguishable when many traces overlap in 3D.
    """
    if n <= 0:
        return []
    golden = 0.618033988749895
    colors_hex: list[str] = []
    for i in range(n):
        hue = (i * golden) % 1.0
        sat = 0.58 + 0.32 * ((i % 5) / 4.0)
        val = 0.68 + 0.26 * (((i // 5) % 4) / 3.0)
        r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val)
        colors_hex.append(
            f"#{int(r_f * 255 + 0.5):02x}{int(g_f * 255 + 0.5):02x}{int(b_f * 255 + 0.5):02x}"
        )
    return colors_hex


def plot_umap_viz(
    df: pd.DataFrame,
    output_path: str | pathlib.Path = "umap.html",
) -> None:
    if "entity" not in df.columns:
        raise ValueError("plot_umap_viz requires an 'entity' column")
    if "class" not in df.columns:
        raise ValueError("plot_umap_viz requires a 'class' column")
    label_col = "entity"
    df = df.copy()
    df["show_label"] = df[label_col]
    show_rate = max(len(df) // 20, 1)
    df.loc[df.index % show_rate != 0, "show_label"] = ""

    df["class"] = df["class"].astype(str)
    class_order = _sorted_class_labels_natural(df["class"])
    n_classes = len(class_order)
    color_discrete_map = dict(
        zip(class_order, _distinct_category_hex_colors(n_classes), strict=True)
    )

    axis_title_font = dict(size=15)
    axis_tick_font = dict(size=13)

    fig = px.scatter_3d(
        df,
        x="uviz_00",
        y="uviz_01",
        z="uviz_02",
        color="class",
        color_discrete_map=color_discrete_map,
        category_orders={"class": class_order},
        hover_name=label_col,
        labels={"uviz_00": "Dim 1", "uviz_01": "Dim 2", "uviz_02": "Dim 3"},
        template="plotly_white",
    )
    # fullData.name is the cluster label for this colored trace (one trace per class).
    fig.update_traces(
        marker=dict(size=6, opacity=0.82, line=dict(width=0)),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Cluster: <b>%{fullData.name}</b><br>"
            "Dim 1: %{x:.4f}<br>Dim 2: %{y:.4f}<br>Dim 3: %{z:.4f}"
            "<extra></extra>"
        ),
        selector=dict(mode="markers"),
    )

    df_labels = df[df["show_label"] != ""]
    text_trace = go.Scatter3d(
        x=df_labels["uviz_00"],
        y=df_labels["uviz_01"],
        z=df_labels["uviz_02"],
        mode="text",
        text=df_labels["show_label"],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
        textfont=dict(size=14, color="black"),
    )
    fig.add_trace(text_trace)

    legend_font = 13 if n_classes > 36 else 14
    fig.update_layout(
        font=dict(size=14),
        title=dict(
            text=f"3D embedding visualization ({len(df):,} points, {n_classes} clusters)",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        hoverlabel=dict(font=dict(size=15)),
        hovermode="closest",
        scene=dict(
            xaxis=dict(
                title=dict(text="Dim 1", font=axis_title_font),
                tickfont=axis_tick_font,
            ),
            yaxis=dict(
                title=dict(text="Dim 2", font=axis_title_font),
                tickfont=axis_tick_font,
            ),
            zaxis=dict(
                title=dict(text="Dim 3", font=axis_title_font),
                tickfont=axis_tick_font,
            ),
            bgcolor="rgb(250,250,252)",
        ),
        legend=dict(
            title=dict(text="Cluster", font=dict(size=15)),
            traceorder="normal",
            itemsizing="constant",
            font=dict(size=legend_font),
            yanchor="top",
            y=0.99,
            x=1.02,
            xanchor="left",
        ),
        margin=dict(l=0, r=120, b=0, t=56),
    )

    fig.write_html(str(output_path))


def plot_metrics(df: pd.DataFrame, fname):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("min_cluster_size")
    ax1.set_ylabel("DBCV Score", color=color1)
    ax1.plot(
        df["min_cluster_size"],
        df["dbcv"],
        marker="o",
        color=color1,
        label="DBCV",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Add a second y-axis for icm
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    # ax2.set_yscale("log")
    ax2.set_ylabel("ICM", color=color2)
    ax2.plot(
        df["min_cluster_size"],
        df["icm"],
        marker="s",
        linestyle="--",
        color=color2,
        label="ICM",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add a second y-axis for icm
    ax3 = ax1.twinx()
    color2 = "tab:green"
    # ax3.set_yscale("log")
    ax3.set_ylabel("n_clusters", color=color2)
    ax3.plot(
        df["min_cluster_size"],
        df["n_clusters"],
        marker="s",
        linestyle="--",
        color=color2,
        label="ICM",
    )

    ax3.tick_params(axis="y", labelcolor=color2)

    # Titles and layout
    plt.title("Clustering metrics vs. min_cluster_size (HDBSCAN)")
    fig.tight_layout()

    try:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
    finally:
        plt.close(fig)
