import colorsys
import logging
import pathlib
import re

import matplotlib
import numpy as np
import pandas as pd

from pySankey.sankey import sankey
from pelinker.search.grid_solver import (
    grid_solver_overrides_active,
    resolve_chosen_min_cluster_size_by_combo_from_grid,
    should_resolve_chosen_min_cluster_size,
    solve_pooled_grid_from_metrics_list,
)
from pelinker.clustering.grid import SmoothedGridOptimumResult
from pelinker.core.config import ClusteringOptimizationConfig, GridObjectiveSpec
from pelinker.search.grid_export import (
    apply_chosen_min_cluster_size_to_grid,
    has_grid_points_for_dbcv_ari_scatter,
    select_grid_points_at_chosen_min_cluster_size,
)
from pelinker.reports.schema import LinkerFitDiagnostics, ModelSelectionReport
from pelinker.core.scaling import ScaleCurve
import seaborn as sns

# Force a non-interactive backend because this project only saves plots to files.
# This avoids Tk/threading crashes in long-running or parallelized workflows.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Polygon, Rectangle, RegularPolygon, Wedge
from plotly import express as px, graph_objects as go

logger = logging.getLogger(__name__)

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


def _save_figure_multi_format(
    fig: plt.Figure,
    output_path: pathlib.Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> tuple[pathlib.Path, ...]:
    output_path = pathlib.Path(output_path)
    known_suffixes = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}
    base = (
        output_path.with_suffix("")
        if output_path.suffix.lower() in known_suffixes
        else output_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for fmt in formats:
        out = base.parent / f"{base.name}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(out)
    return tuple(written)


def _arity_from_model(model: str) -> str:
    if model == "fusion2":
        return "fusion2"
    if model == "fusion3":
        return "fusion3"
    return "singleton"


def _parse_fusion_layer_parts(layer: str) -> list[tuple[str, str]]:
    """Split fusion ``a/L1+b/L2`` into ``(model, layer_spec)`` pairs."""
    parts: list[tuple[str, str]] = []
    for part in str(layer).split("+"):
        p = part.strip()
        if not p:
            continue
        model, sep, lyr = p.partition("/")
        parts.append((model, lyr if sep else p))
    return parts


def _style_metrics_axis(ax: plt.Axes, *, ylabel: str, title: str, color: str) -> None:
    ax.set_xlabel("min_cluster_size", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold", color=color)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelcolor=color)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _base_models_in_row(model: str, layer: str) -> list[str]:
    """Constituent encoder model names (singleton: one; fusion: from ``a/L1+b/L2``)."""
    if model not in ("fusion2", "fusion3"):
        return [str(model)]
    names = [m for m, _ in _parse_fusion_layer_parts(layer) if m]
    return names if names else [str(model)]


def _layer_spec_code(model: str, layer: str) -> str:
    """
    Short label for the layer configuration only (no encoder name): singleton ``layer`` as-is;
    fusion ``a/L1+b/L2`` → ``L1+L2``.
    """
    if model not in ("fusion2", "fusion3"):
        return str(layer)
    specs = [lyr for _m, lyr in _parse_fusion_layer_parts(layer)]
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
    *,
    optimization_config: ClusteringOptimizationConfig | None = None,
    grid_cluster_count_reward: float | None = None,
    grid_n_entities: int | None = None,
    grid_objective: GridObjectiveSpec | None = None,
    grid_one_se_k: float | None = None,
) -> bool:
    """
    Scatter of mean DBCV vs mean ARI per (model, layer); shape = arity (△/□/○),
    fill colors = base encoder model(s); text = layer spec only (e.g. fusion ``2+3``).
    95% covariance ellipses when ``n_sample`` ≥ 2.

    Uses ``(dbcv, ari)`` at ``chosen_min_cluster_size`` for each bootstrap ``sample_idx``
    (same hyperparameter as the vertical line on per-combination error-bar plots).
    Both axes are fixed to ``[0, _AXIS_MAX]``.

    When any solver argument is set (``optimization_config`` or grid override kwargs),
    ``chosen_min_cluster_size`` is re-computed per (model, layer) from the grid metrics
    instead of using values stored in ``df_grid``.

    Returns:
        True if a figure was written, False if required data were absent.
    """
    if not has_grid_points_for_dbcv_ari_scatter(df_grid):
        return False

    df_plot = df_grid
    if grid_solver_overrides_active(
        optimization_config=optimization_config,
        grid_cluster_count_reward=grid_cluster_count_reward,
        grid_n_entities=grid_n_entities,
        grid_objective=grid_objective,
        grid_one_se_k=grid_one_se_k,
    ):
        chosen_by_combo = resolve_chosen_min_cluster_size_by_combo_from_grid(
            df_grid,
            optimization_config,
            grid_cluster_count_reward=grid_cluster_count_reward,
            grid_n_entities=grid_n_entities,
            grid_objective=grid_objective,
            grid_one_se_k=grid_one_se_k,
        )
        if chosen_by_combo:
            df_plot = apply_chosen_min_cluster_size_to_grid(df_grid, chosen_by_combo)

    try:
        df = select_grid_points_at_chosen_min_cluster_size(df_plot)
    except ValueError:
        return False
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
                g["dbcv"].to_numpy(dtype=np.float64),
                g["ari"].to_numpy(dtype=np.float64),
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
    ax.set_xlabel("DBCV at chosen min_cluster_size (mean over samples)")
    ax.set_ylabel("ARI at chosen min_cluster_size (mean over samples)")
    ax.set_title(
        "DBCV vs ARI at consensus hyperparameter; dashed ellipse ≈95% (n_sample >= 2)"
    )

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

    _draw_pareto_front(ax, df)

    ax.grid(True, alpha=0.28, linestyle="--", zorder=0)
    plt.tight_layout()
    _save_figure_multi_format(fig, output_path)
    plt.close(fig)
    return True


def pareto_front_mask(dbcv: np.ndarray, ari: np.ndarray) -> np.ndarray:
    """Boolean mask of points not dominated on ``(dbcv, ari)``, both maximized.

    A point is dominated when another is at least as good on both metrics and strictly
    better on one. The front is the honest answer to "maximize both": every scalar score,
    including the geometric mean, just picks one point off it.
    """
    d = np.asarray(dbcv, dtype=np.float64)
    a = np.asarray(ari, dtype=np.float64)
    n = d.size
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not (np.isfinite(d[i]) and np.isfinite(a[i])):
            continue
        dominated = (
            (d >= d[i])
            & (a >= a[i])
            & ((d > d[i]) | (a > a[i]))
            & np.isfinite(d)
            & np.isfinite(a)
        )
        mask[i] = not bool(np.any(dominated))
    return mask


def _draw_pareto_front(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Outline the non-dominated (model, layer) candidates on the DBCV-vs-ARI scatter."""
    centroids = (
        df.groupby(["model", "layer"], sort=False)[["dbcv", "ari"]].mean().reset_index()
    )
    if len(centroids) < 2:
        return
    d = centroids["dbcv"].to_numpy(dtype=np.float64)
    a = centroids["ari"].to_numpy(dtype=np.float64)
    front = pareto_front_mask(d, a)
    if not np.any(front):
        return
    order = np.argsort(d[front])
    ax.plot(
        d[front][order],
        a[front][order],
        linestyle="--",
        linewidth=1.4,
        color="0.35",
        alpha=0.8,
        marker="none",
        zorder=1,
        label="Pareto front",
    )


def _draw_grid_objective_panel(
    ax: plt.Axes,
    grid_solve: SmoothedGridOptimumResult | None,
    *,
    color: str,
) -> str:
    """Draw the combined-objective curve with its one-SE decision band. Returns the title.

    Shows *why* a ``min_cluster_size`` was chosen: the pooled objective with its paired
    standard errors, the ``mu[argmax] - k*se`` threshold, which grid points are
    statistically tied with the best, and where the rule landed.
    """
    if grid_solve is None or len(grid_solve.x) == 0:
        return "Grid objective (unavailable)"

    x = np.array(grid_solve.x, dtype=np.float64)
    y_obj = np.array(grid_solve.y_objective, dtype=np.float64)
    y_smooth = np.array(grid_solve.y_smooth, dtype=np.float64)
    se = np.array(grid_solve.y_se, dtype=np.float64)

    if se.size == y_smooth.size and np.any(se > 0):
        ax.fill_between(
            x,
            y_smooth - se,
            y_smooth + se,
            color=color,
            alpha=0.18,
            linewidth=0,
            label="±1 paired SE",
            zorder=0,
        )

    ax.plot(
        x,
        y_obj,
        marker="s",
        color=color,
        linewidth=1.2,
        markersize=5,
        alpha=0.55,
        label="objective (raw)",
        zorder=2,
    )
    if np.any(np.isfinite(y_smooth)):
        ax.plot(
            x,
            y_smooth,
            color=color,
            linewidth=2.2,
            label="objective (smoothed)",
            zorder=3,
        )

    eligible = np.array(grid_solve.y_eligible, dtype=bool)
    if eligible.size == x.size and np.any(eligible):
        ax.plot(
            x[eligible],
            y_smooth[eligible],
            linestyle="none",
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.4,
            label="within 1 SE of best",
            zorder=4,
        )

    argmax_x = grid_solve.argmax_min_cluster_size
    if argmax_x is not None:
        i_star = int(np.argmin(np.abs(x - float(argmax_x))))
        if np.isfinite(y_smooth[i_star]):
            ax.axhline(
                y_smooth[i_star],
                color="0.45",
                linestyle=":",
                linewidth=1.2,
                label="best mean",
                zorder=1,
            )
            if grid_solve.one_se_k > 0 and se.size == x.size:
                # Each point is tested against its own paired SE, so the decision boundary
                # is a curve, not a level.
                ax.plot(
                    x,
                    y_smooth[i_star] - grid_solve.one_se_k * se,
                    color="0.45",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"best − {grid_solve.one_se_k:g}·SE",
                    zorder=1,
                )
        ax.plot(
            [float(argmax_x)],
            [y_smooth[i_star]],
            marker="*",
            markersize=14,
            color="#333333",
            linestyle="none",
            label="argmax",
            zorder=5,
        )

    has_cluster_term = any(
        abs(v) > 1e-12 for v in grid_solve.y_cluster_term if np.isfinite(v)
    )
    bits = ["pooled"]
    if has_cluster_term:
        bits.append("cluster reward")
    if grid_solve.n_samples:
        bits.append(f"n={grid_solve.n_samples}")
    suffix = ", ".join(bits)
    if grid_solve.selection == "degenerate_fallback":
        return f"Grid objective — DEGENERATE, fell back to DBCV ({suffix})"
    return f"Grid objective ({suffix}); pick: {grid_solve.selection}"


def plot_metrics_with_error_bars(
    metrics_list: list[pd.DataFrame],
    output_path: pathlib.Path,
    *,
    chosen_min_cluster_size: float | None = None,
    grid_solve: SmoothedGridOptimumResult | None = None,
    optimization_config: ClusteringOptimizationConfig | None = None,
    grid_cluster_count_reward: float | None = None,
    grid_n_entities: int | None = None,
    grid_objective: GridObjectiveSpec | None = None,
    grid_one_se_k: float | None = None,
):
    """
    Plot metrics across multiple runs with error bars using seaborn lineplot.

    Args:
        metrics_list: List of DataFrames, each with columns: min_cluster_size, icm, n_clusters, dbcv, ari
        output_path: Path to save the figure
        chosen_min_cluster_size: Optional vertical marker for the selected grid value (e.g. from smoother / argmax).
        grid_solve: Precomputed pooled grid diagnostics (avoids a second solve; drives objective panel).
        optimization_config: When set (or when any grid override kwarg is set and ``chosen_min_cluster_size``
            is omitted), re-run the pooled grid solver for the vertical marker.
        grid_cluster_count_reward: Override :attr:`~pelinker.core.config.ClusteringOptimizationConfig.grid_cluster_count_reward`.
        grid_n_entities: Override :attr:`~pelinker.core.config.ClusteringOptimizationConfig.grid_n_entities`.
        grid_objective: Override :attr:`~pelinker.core.config.ClusteringOptimizationConfig.grid_objective`.
        grid_one_se_k: Override :attr:`~pelinker.core.config.ClusteringOptimizationConfig.grid_one_se_k`.
    """
    # A supplied chosen_min_cluster_size must not suppress the objective *curve* — only
    # override the value. Solving for diagnostics whenever they are missing is what keeps
    # the objective panel from silently rendering blank on the live-run path.
    override_wins = should_resolve_chosen_min_cluster_size(
        chosen_min_cluster_size=chosen_min_cluster_size,
        optimization_config=optimization_config,
        grid_cluster_count_reward=grid_cluster_count_reward,
        grid_n_entities=grid_n_entities,
        grid_objective=grid_objective,
        grid_one_se_k=grid_one_se_k,
    )
    if grid_solve is None and metrics_list:
        try:
            grid_solve = solve_pooled_grid_from_metrics_list(
                metrics_list,
                optimization_config,
                grid_cluster_count_reward=grid_cluster_count_reward,
                grid_n_entities=grid_n_entities,
                grid_objective=grid_objective,
                grid_one_se_k=grid_one_se_k,
            )
        except ValueError as exc:
            # Diagnostics are best-effort; never fail a plot over them.
            logger.debug(
                "Grid objective diagnostics unavailable for %s: %s", output_path, exc
            )

    if grid_solve is not None and (chosen_min_cluster_size is None or override_wins):
        chosen_min_cluster_size = float(grid_solve.chosen_min_cluster_size)

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
    colors = ["#2E86AB", "#A23B72", "#C44E52", "#F18F01"]  # Blue, Purple, Red, Orange

    if has_ari:
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        ax_dbcv, ax_ari, ax_k, ax_obj = axes[0], axes[1], axes[2], axes[3]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_dbcv, ax_k, ax_obj = axes[0], axes[1], axes[2]
        ax_ari = None

    def _maybe_vline(ax: plt.Axes) -> None:
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

    sns.lineplot(
        data=df_combined,
        x="min_cluster_size",
        y="dbcv",
        ax=ax_dbcv,
        errorbar="sd",
        marker="o",
        color=colors[0],
        linewidth=2,
        markersize=8,
        err_kws={"alpha": 0.3, "linewidth": 1.5},
    )
    _maybe_vline(ax_dbcv)
    _style_metrics_axis(
        ax_dbcv,
        ylabel="DBCV Score",
        title="DBCV vs. min_cluster_size",
        color=colors[0],
    )

    if ax_ari is not None:
        sns.lineplot(
            data=df_combined,
            x="min_cluster_size",
            y="ari",
            ax=ax_ari,
            errorbar="sd",
            marker="D",
            color=colors[1],
            linewidth=2,
            markersize=7,
            err_kws={"alpha": 0.3, "linewidth": 1.5},
        )
        _maybe_vline(ax_ari)
        _style_metrics_axis(
            ax_ari,
            ylabel="ARI",
            title="ARI vs. min_cluster_size",
            color=colors[1],
        )

    sns.lineplot(
        data=df_combined,
        x="min_cluster_size",
        y="n_clusters",
        ax=ax_k,
        errorbar="sd",
        marker="^",
        color=colors[2],
        linewidth=2,
        markersize=8,
        err_kws={"alpha": 0.3, "linewidth": 1.5},
    )
    _maybe_vline(ax_k)
    _style_metrics_axis(
        ax_k,
        ylabel="n clusters",
        title="Number of Clusters vs. min_cluster_size",
        color=colors[2],
    )

    obj_color = colors[3]
    obj_title = _draw_grid_objective_panel(ax_obj, grid_solve, color=obj_color)
    _maybe_vline(ax_obj)
    _style_metrics_axis(
        ax_obj, ylabel="Grid objective", title=obj_title, color=obj_color
    )
    if grid_solve is not None and len(grid_solve.x) > 0:
        ax_obj.legend(fontsize=8, loc="best", framealpha=0.85)

    plt.tight_layout()
    _save_figure_multi_format(fig, output_path)
    plt.close(fig)


def plot_heatmap(
    df_results: pd.DataFrame,
    output_path: pathlib.Path,
    metric: str = "best_score",
    metric_label: str | None = None,
    *,
    secondary_metric: str | None = "best_size",
):
    """
    Create a heatmap with model (rows) and layer (columns).
    Color represents the specified metric; text overlays ``secondary_metric`` and metric value.

    Args:
        df_results: DataFrame with columns: model, layer, …
        output_path: Path to save the heatmap figure
        metric: Column name for the metric to display as color (default: "best_score")
        metric_label: Label for the metric (default: uses metric column name)
        secondary_metric: Column for text annotation besides the metric cell (default:
            ``"best_size"``); use ``None`` to annotate metric value only.
    """
    if metric_label is None:
        metric_label = metric.replace("_", " ").title()

    # Create pivot tables
    score_pivot = df_results.pivot(index="model", columns="layer", values=metric)
    size_pivot: pd.DataFrame | None = None
    if secondary_metric is not None and secondary_metric in df_results.columns:
        size_pivot = df_results.pivot(
            index="model", columns="layer", values=secondary_metric
        )

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
            if pd.isna(score_val):
                continue
            secondary_ok = True
            size_val: float | None = None
            if size_pivot is not None:
                sv = size_pivot.iloc[i, j]
                if pd.isna(sv):
                    secondary_ok = False
                else:
                    size_val = float(sv)

            if not secondary_ok and size_pivot is not None:
                continue

            text_color = "white" if score_val < mean_score else "black"
            if abs(score_val) < 0.01:
                metric_str = f"{score_val:.2e}"
            elif abs(score_val) < 1:
                metric_str = f"{score_val:.3f}"
            else:
                metric_str = f"{score_val:.2f}"

            if size_val is None:
                anno = metric_str
            else:
                size_display = (
                    str(int(round(size_val)))
                    if secondary_metric == "best_size"
                    else f"{size_val:.2f}"
                )
                anno = f"{size_display}\n{metric_str}"

            ax.text(
                j + 0.5,
                i + 0.5,
                anno,
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=8,
                linespacing=1.2,
            )

    sub_label = secondary_metric.replace("_", " ").title() if secondary_metric else None
    if sub_label:
        ax.set_title(f"Results: {metric_label} (color) and {sub_label} + value (text)")
    else:
        ax.set_title(f"Results: {metric_label} (color)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")

    plt.tight_layout()
    _save_figure_multi_format(fig, output_path)
    plt.close()


def plot_screener_oov_bar(
    summary_df: pd.DataFrame,
    output_path: pathlib.Path,
) -> bool:
    """
    Grouped bar chart per (model, layer): mean AUC for screener / OOV / combined.

    Requires ``screener_auc_mean``, ``oov_auc_mean``, ``combined_auc_mean``.
    Single-embedding rows only (excludes fusion* model labels).
    """
    required = ("screener_auc_mean", "oov_auc_mean", "combined_auc_mean")
    if not all(c in summary_df.columns for c in required):
        return False
    df = (
        summary_df[~summary_df["model"].astype(str).str.startswith("fusion")]
        .dropna(subset=list(required))
        .copy()
    )
    if df.empty:
        return False
    df["_label"] = (
        df["model"].astype(str) + " / " + df["layer"].astype(str).str.slice(0, 24)
    )
    df = df.sort_values("combined_auc_mean", ascending=False)
    df = df.reset_index(drop=True)
    labels = df["_label"].tolist()
    n = len(labels)
    x = np.arange(n)
    w = 0.25

    ys = df["screener_auc_mean"].to_numpy()
    yo = df["oov_auc_mean"].to_numpy()
    yc = df["combined_auc_mean"].to_numpy()

    errs = []
    for col_std in ("screener_auc_std", "oov_auc_std", "combined_auc_std"):
        if col_std in df.columns:
            errs.append(df[col_std].fillna(0.0).to_numpy())
        else:
            errs.append(np.zeros(n))

    colors = ["#4C72B0", "#55A868", "#C44E52"]

    all_vals = np.concatenate([ys, yo, yc])
    all_errs = np.concatenate([errs[0], errs[1], errs[2]])
    data_min = max(0.0, (all_vals - all_errs).min())
    data_max = min(1.0, (all_vals + all_errs).max())
    margin = max(0.02, (data_max - data_min) * 0.15)
    y_lo = max(0.0, data_min - margin)
    y_hi = min(1.0, data_max + margin)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 6))
    ax.bar(
        x - w,
        ys,
        width=w,
        yerr=errs[0],
        capsize=3,
        label="Screener AUC",
        color=colors[0],
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 1.0, "ecolor": "dimgray", "capthick": 1.0},
    )
    ax.bar(
        x,
        yo,
        width=w,
        yerr=errs[1],
        capsize=3,
        label="OOV AUC",
        color=colors[1],
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 1.0, "ecolor": "dimgray", "capthick": 1.0},
    )
    ax.bar(
        x + w,
        yc,
        width=w,
        yerr=errs[2],
        capsize=3,
        label="Combined AUC",
        color=colors[2],
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 1.0, "ecolor": "dimgray", "capthick": 1.0},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, color="gray")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylabel("AUC")
    ax.set_title("Mean AUC: screener / OOV / combined (± std)")
    plt.tight_layout()
    _save_figure_multi_format(fig, output_path)
    plt.close()
    return True


def plot_roc_comparison(
    scores_df: pd.DataFrame,
    output_path: pathlib.Path,
    *,
    combo_keys: list[str],
) -> bool:
    """
    ROC curves for ``screener_best_score``, ``oov_score``, and ``combined_score``
    pooled over samples per ``combo_key``.
    """
    if scores_df.empty or not combo_keys:
        return False
    from sklearn.metrics import auc as sk_auc, roc_curve

    avail = scores_df.loc[scores_df["combo_key"].isin(combo_keys)]
    usable = avail["combo_key"].unique().tolist()
    if not usable:
        return False

    cols = {"y_true", "screener_best_score", "oov_score", "combined_score"}
    if not cols.issubset(scores_df.columns):
        return False

    n_p = len(usable)
    ncols = min(3, n_p)
    nrows = int(np.ceil(n_p / ncols)) if n_p else 1
    # ncells = max(1, nrows * ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ix, ck in enumerate(usable):
        if ix >= axes_flat.shape[0]:
            break
        ax = axes_flat[ix]
        sub = scores_df.loc[scores_df["combo_key"] == ck].copy()
        y = np.asarray(sub["y_true"], dtype=np.int64)
        if np.unique(y).size < 2:
            ax.set_visible(False)
            continue
        for name, serie, ls in (
            ("Screener", sub["screener_best_score"], "-"),
            ("OOV", sub["oov_score"], "--"),
            ("Combined", sub["combined_score"], "-."),
        ):
            s = np.asarray(serie, dtype=np.float64)
            try:
                fpr, tpr, _ = roc_curve(y, s)
                a = float(sk_auc(fpr, tpr))
            except ValueError:
                continue
            ax.plot(fpr, tpr, ls=ls, lw=2, label=f"{name} (AUC={a:.3f})")

        ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.35)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title(str(ck)[:60], fontsize=9)

    for k in range(n_p, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout()
    _save_figure_multi_format(fig, output_path)
    plt.close()
    return True


def _sorted_class_labels_natural(class_series: pd.Series) -> list[str]:
    """Order legend / color map: numeric cluster ids sort numerically, not lexically."""
    labels = [str(x) for x in class_series.unique()]

    def sort_key(label: str) -> tuple[int, float] | tuple[int, str]:
        try:
            return (0, float(label))
        except ValueError:
            return (1, label)

    return sorted(labels, key=sort_key)


_CLUSTER_VIZ_LEGEND_MAX_CHARS = 28
_CLUSTER_VIZ_HOVER_WRAP = 48


def _truncate_legend_label(
    label: str, *, max_chars: int = _CLUSTER_VIZ_LEGEND_MAX_CHARS
) -> str:
    """Shorten long KB-out display names for the Plotly legend."""
    text = str(label).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)].rstrip("-_ ") + "…"


def _wrap_hover_html(text: str, *, width: int = _CLUSTER_VIZ_HOVER_WRAP) -> str:
    """Insert ``<br>`` breaks so long hover strings stay readable."""
    raw = str(text).replace("\n", " ").strip()
    if not raw or width < 8:
        return raw
    parts: list[str] = []
    remaining = raw
    while len(remaining) > width:
        cut = remaining.rfind(" ", 0, width + 1)
        if cut < width // 3:
            cut = width
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return "<br>".join(parts)


def _format_hover_float(value: object) -> str:
    """Format a numeric hover value to at most 3 decimal places."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


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


def plot_cluster_viz(
    df: pd.DataFrame,
    output_path: str | pathlib.Path = "cluster_viz.html",
    *,
    viz_method: str = "pca",
) -> None:
    if "entity" not in df.columns:
        raise ValueError("plot_cluster_viz requires an 'entity' column")
    if "class" not in df.columns:
        raise ValueError("plot_cluster_viz requires a 'class' column")
    if "cviz_00" not in df.columns or "cviz_01" not in df.columns:
        raise ValueError("plot_cluster_viz requires cviz_00 and cviz_01 columns")
    use_3d = "cviz_02" in df.columns
    label_col = "entity"
    df = df.copy()
    df["show_label"] = df[label_col]
    show_rate = max(len(df) // 20, 1)
    df.loc[df.index % show_rate != 0, "show_label"] = ""

    df["class_full"] = df["class"].astype(str)
    # Stable short legend keys; full name stays in hover via class_full.
    unique_full = _sorted_class_labels_natural(df["class_full"])
    full_to_legend: dict[str, str] = {}
    used_legend: set[str] = set()
    for full in unique_full:
        base = _truncate_legend_label(full)
        legend = base
        suffix = 2
        while legend in used_legend:
            trim = max(1, _CLUSTER_VIZ_LEGEND_MAX_CHARS - len(f"…{suffix}") - 1)
            legend = f"{base[:trim].rstrip('-_ ')}…{suffix}"
            suffix += 1
        used_legend.add(legend)
        full_to_legend[full] = legend
    df["class"] = df["class_full"].map(full_to_legend)
    class_order = [full_to_legend[f] for f in unique_full]
    n_classes = len(class_order)
    color_discrete_map = dict(
        zip(class_order, _distinct_category_hex_colors(n_classes), strict=True)
    )

    axis_title_font = dict(size=15)
    axis_tick_font = dict(size=13)

    df["cluster_hover"] = df["class_full"].map(_wrap_hover_html)
    if "context" in df.columns:
        df["context"] = df["context"].map(
            lambda v: _wrap_hover_html("" if pd.isna(v) else str(v))
        )
    if "cluster_score" in df.columns:
        df["cluster_score"] = df["cluster_score"].map(_format_hover_float)

    # Hover order: cluster, pmid, mention, context, coords, cluster score.
    hover_before_coords: list[tuple[str, str]] = [("cluster_hover", "Cluster")]
    for col, label in (
        ("pmid", "PMID"),
        ("mention", "Mention"),
        ("context", "Context"),
    ):
        if col in df.columns:
            hover_before_coords.append((col, label))
    hover_after_coords: list[tuple[str, str]] = []
    if "cluster_score" in df.columns:
        hover_after_coords.append(("cluster_score", "Cluster score"))
    hover_specs = hover_before_coords + hover_after_coords

    scatter_kwargs: dict[str, object] = {
        "x": "cviz_00",
        "y": "cviz_01",
        "color": "class",
        "color_discrete_map": color_discrete_map,
        "category_orders": {"class": class_order},
        "labels": {"cviz_00": "Dim 1", "cviz_01": "Dim 2"},
        "template": "plotly_white",
        "custom_data": [c for c, _ in hover_specs],
    }
    if use_3d:
        scatter_kwargs["z"] = "cviz_02"
        scatter_kwargs["labels"] = {
            "cviz_00": "Dim 1",
            "cviz_01": "Dim 2",
            "cviz_02": "Dim 3",
        }

    if use_3d:
        fig = px.scatter_3d(df, **scatter_kwargs)
    else:
        fig = px.scatter(df, **scatter_kwargs)

    dim_z_line = ",%{z:.3f}" if use_3d else ""
    hover_lines = ""
    for i, (_, label) in enumerate(hover_before_coords):
        if i == 0:
            # Cluster name as the bold header (no "Cluster:" prefix).
            hover_lines += f"<b>%{{customdata[{i}]}}</b><br>"
        else:
            hover_lines += f"{label}: %{{customdata[{i}]}}<br>"
    hover_lines += f"Coord: (%{{x:.3f}}, %{{y:.3f}}{dim_z_line})<br>"
    for j, (_, label) in enumerate(hover_after_coords):
        i = len(hover_before_coords) + j
        hover_lines += f"{label}: %{{customdata[{i}]}}<br>"
    hover_lines += "<extra></extra>"

    fig.update_traces(
        marker=dict(size=6, opacity=0.82, line=dict(width=0)),
        hovertemplate=hover_lines,
        selector=dict(mode="markers"),
    )

    df_labels = df[df["show_label"] != ""]
    if use_3d:
        text_trace = go.Scatter3d(
            x=df_labels["cviz_00"],
            y=df_labels["cviz_01"],
            z=df_labels["cviz_02"],
            mode="text",
            text=df_labels["show_label"],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
            textfont=dict(size=14, color="black"),
        )
    else:
        text_trace = go.Scatter(
            x=df_labels["cviz_00"],
            y=df_labels["cviz_01"],
            mode="text",
            text=df_labels["show_label"],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
            textfont=dict(size=14, color="black"),
        )
    fig.add_trace(text_trace)

    legend_font = 12 if n_classes > 36 else 13
    method_label = viz_method.upper() if viz_method == "pca" else "UMAP"
    # Keep a stable viewport; do not grow height with legend length (that uncenters the plot).
    # Plotly 6 scrolls an overflowing legend when the figure height is fixed.
    fig_height = 720 if use_3d else 680
    fig_width = 1100
    layout: dict[str, object] = {
        "font": dict(size=14),
        "width": fig_width,
        "height": fig_height,
        "title": dict(
            text=(
                f"{method_label} · {len(df):,} pts · {n_classes} clusters"
                "<br><sup>Legend: double-click to isolate · click to show/hide"
                " · drag to pan · scroll to zoom</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        "hoverlabel": dict(font=dict(size=14), align="left"),
        "hovermode": "closest",
        # 2D: pan with drag (Plotly default is zoom, which feels broken for exploration).
        "dragmode": "pan",
        "legend": dict(
            # Keep the title on one line: multi-line HTML titles clip legend items in Plotly.
            title=dict(
                text="Cluster (dbl-click isolate)",
                font=dict(size=13),
            ),
            traceorder="normal",
            itemsizing="constant",
            font=dict(size=legend_font),
            yanchor="top",
            y=0.99,
            x=1.01,
            xanchor="left",
            bgcolor="rgba(255,255,255,0.9)",
            borderwidth=0,
        ),
        # Symmetric-ish margins; leave room on the right for the legend without skewing the plot.
        "margin": dict(l=60, r=220, b=50, t=70, pad=4),
    }
    if use_3d:
        layout["dragmode"] = "orbit"
        layout["scene"] = dict(
            aspectmode="data",
            dragmode="orbit",
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
            # Keep the 3D viewport centered in the available area (legend sits outside).
            domain=dict(x=[0.0, 0.78], y=[0.0, 1.0]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        )
    else:
        # Equal aspect so PCA/UMAP space is not stretched; automargin keeps labels in frame.
        layout["xaxis"] = dict(
            title=dict(text="Dim 1", font=axis_title_font),
            tickfont=axis_tick_font,
            automargin=True,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
        )
        layout["yaxis"] = dict(
            title=dict(text="Dim 2", font=axis_title_font),
            tickfont=axis_tick_font,
            automargin=True,
            constrain="domain",
        )
    fig.update_layout(**layout)

    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)


def plot_metrics(
    df: pd.DataFrame,
    output_path: pathlib.Path,
    *,
    chosen_min_cluster_size: float | None = None,
    grid_solve: SmoothedGridOptimumResult | None = None,
    optimization_config: ClusteringOptimizationConfig | None = None,
) -> None:
    """Single-sample version of :func:`plot_metrics_with_error_bars`.

    With one sample the paired standard errors are all zero, so the objective panel shows
    the argmax of the smoothed curve with no decision band — which is exactly what the
    solver does in that case.
    """
    if df.empty:
        return

    df_plot = df.copy()
    df_plot = df_plot[df_plot["n_clusters"] > 1].copy()
    if df_plot.empty:
        print(
            f"Warning: No valid data points after filtering (n_clusters > 1) for {output_path}"
        )
        return

    if grid_solve is None:
        try:
            grid_solve = solve_pooled_grid_from_metrics_list([df], optimization_config)
        except ValueError as exc:
            logger.debug(
                "Grid objective diagnostics unavailable for %s: %s", output_path, exc
            )
    if grid_solve is not None and chosen_min_cluster_size is None:
        chosen_min_cluster_size = float(grid_solve.chosen_min_cluster_size)

    has_ari = "ari" in df_plot.columns and bool(df_plot["ari"].notna().any())
    colors = ["#2E86AB", "#A23B72", "#C44E52", "#F18F01"]  # Blue, Purple, Red, Orange

    if has_ari:
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        ax_dbcv, ax_ari, ax_k, ax_obj = axes[0], axes[1], axes[2], axes[3]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_dbcv, ax_k, ax_obj = axes[0], axes[1], axes[2]

    def _maybe_vline(ax: plt.Axes) -> None:
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

    ax_dbcv.plot(
        df_plot["min_cluster_size"],
        df_plot["dbcv"],
        marker="o",
        color=colors[0],
        linewidth=2,
    )
    _maybe_vline(ax_dbcv)
    _style_metrics_axis(
        ax_dbcv,
        ylabel="DBCV Score",
        title="DBCV vs. min_cluster_size",
        color=colors[0],
    )

    if has_ari:
        ax_ari.plot(
            df_plot["min_cluster_size"],
            df_plot["ari"],
            marker="D",
            color=colors[1],
            linewidth=2,
        )
        _maybe_vline(ax_ari)
        _style_metrics_axis(
            ax_ari,
            ylabel="ARI",
            title="ARI vs. min_cluster_size",
            color=colors[1],
        )

    ax_k.plot(
        df_plot["min_cluster_size"],
        df_plot["n_clusters"],
        marker="^",
        color=colors[2],
        linewidth=2,
    )
    _maybe_vline(ax_k)
    _style_metrics_axis(
        ax_k,
        ylabel="n clusters",
        title="Number of Clusters vs. min_cluster_size",
        color=colors[2],
    )

    obj_title = _draw_grid_objective_panel(ax_obj, grid_solve, color=colors[3])
    _maybe_vline(ax_obj)
    _style_metrics_axis(
        ax_obj, ylabel="Grid objective", title=obj_title, color=colors[3]
    )
    if grid_solve is not None and len(grid_solve.x) > 0:
        ax_obj.legend(fontsize=8, loc="best", framealpha=0.85)

    plt.tight_layout()
    try:
        _save_figure_multi_format(fig, output_path)
    finally:
        plt.close(fig)


def diagnostics_to_pairgrid_dataframe(diag: LinkerFitDiagnostics) -> pd.DataFrame:
    """
    Build a DataFrame for :func:`plot_pca_quality_pairgrid` from linker fit diagnostics.

    ``class_label`` is ``negative`` / ``positive`` from ``oov_label`` (1 / 0).
    """
    ol = np.asarray(diag.oov_label, dtype=np.int64).ravel()
    return pd.DataFrame(
        {
            "pca_residual": np.asarray(diag.pca_residual, dtype=np.float64),
            "pca_mahalanobis": np.asarray(diag.pca_mahalanobis, dtype=np.float64),
            "pca_spectral_entropy": np.asarray(
                diag.pca_spectral_entropy, dtype=np.float64
            ),
            "class_label": np.where(ol == 1, "negative", "positive"),
        }
    )


_PCA_CLASS_MARKERS: dict[str, str] = {
    "positive": "o",
    "negative": "^",
}


def _balanced_subsample_by_class(
    source: pd.DataFrame,
    n: int,
    class_col: str,
) -> pd.DataFrame:
    """Cap each class at n // n_classes so minority classes stay visible in plots."""
    classes = sorted(source[class_col].unique())
    if not classes:
        return source.iloc[0:0].copy()
    if len(source) <= n:
        return source.copy()
    n_per = max(1, n // len(classes))
    parts: list[pd.DataFrame] = []
    for cls in classes:
        sub = source.loc[source[class_col] == cls]
        k = min(len(sub), n_per)
        if k > 0:
            parts.append(sub.sample(n=k, random_state=0))
    if not parts:
        return source.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _pca_pairgrid_title(
    *,
    subtitle: str | None,
    plot_df: pd.DataFrame,
    class_col: str,
    scatter_df: pd.DataFrame,
    kde_df: pd.DataFrame,
) -> str:
    raw_counts = plot_df[class_col].value_counts()
    count_bits = " ".join(
        f"{cls}={int(raw_counts[cls]):,}" for cls in sorted(raw_counts.index)
    )
    head = subtitle if subtitle else "PCA quality"
    return (
        f"{head} | {count_bits} | "
        f"n={len(plot_df):,} plot scatter={len(scatter_df):,} kde={len(kde_df):,}"
    )


def plot_pca_quality_pairgrid(
    df: pd.DataFrame,
    output_path: pathlib.Path,
    *,
    class_col: str = "class_label",
    subtitle: str | None = None,
    max_scatter_points: int = 4_000,
    max_kde_points: int = 20_000,
) -> bool:
    """
    PairGrid of PCA quality features (pca_residual, pca_spectral_entropy, pca_mahalanobis)
    with hue for class (negative / positive).

    Upper triangle: balanced class subsample (equal cap per class) with distinct markers.
    Lower triangle & diagonal: KDE per hue with ``common_norm=False`` so each class is
    normalized on its own scale (not pooled against the majority class).

    Args:
        df: DataFrame with feature columns and class_col.
        output_path: PNG path (PDF sibling also written).
        class_col: Column used for hue.
        subtitle: Combo label for the figure suptitle (e.g. model/layer/sample).
        max_scatter_points: Cap for upper-triangle scatter layer (split evenly by class).
        max_kde_points: Cap for lower-triangle KDE and diagonal KDE layers.
    """
    feature_cols = ["pca_residual", "pca_spectral_entropy", "pca_mahalanobis"]
    needed = set(feature_cols + [class_col])
    if not needed.issubset(df.columns):
        return False

    plot_df = df.loc[:, feature_cols + [class_col]].dropna().copy()
    if plot_df.empty:
        return False

    scatter_df = _balanced_subsample_by_class(plot_df, max_scatter_points, class_col)
    kde_df = _balanced_subsample_by_class(plot_df, max_kde_points, class_col)

    classes = sorted(plot_df[class_col].unique())
    palette = sns.color_palette("Set2", n_colors=len(classes))
    color_map = dict(zip(classes, palette, strict=True))
    marker_map = {
        cls: _PCA_CLASS_MARKERS.get(str(cls), ("o", "^")[i % 2])
        for i, cls in enumerate(classes)
    }
    class_sizes = plot_df[class_col].value_counts()
    # Draw majority first, minority on top in scatter panels.
    classes_by_size = sorted(classes, key=lambda c: int(class_sizes.get(c, 0)))

    g = sns.PairGrid(
        kde_df, vars=feature_cols, hue=class_col, palette=color_map, diag_sharey=False
    )
    g.map_lower(
        sns.kdeplot,
        fill=True,
        alpha=0.4,
        thresh=0.05,
        common_norm=False,
        legend=False,
    )
    g.map_diag(sns.kdeplot, lw=2, common_norm=False, fill=False, legend=False)

    n_vars = len(feature_cols)
    for row in range(n_vars):
        for col in range(n_vars):
            if col <= row:
                continue
            ax = g.axes[row, col]
            x_col = feature_cols[col]
            y_col = feature_cols[row]
            for z, cls in enumerate(classes_by_size):
                sub = scatter_df.loc[scatter_df[class_col] == cls]
                if sub.empty:
                    continue
                ax.scatter(
                    sub[x_col].to_numpy(),
                    sub[y_col].to_numpy(),
                    color=color_map[cls],
                    marker=marker_map[cls],
                    alpha=0.55,
                    s=16,
                    linewidths=0.35,
                    edgecolors="white",
                    rasterized=True,
                    zorder=z + 1,
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[cls],
            color="w",
            markerfacecolor=color_map[cls],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=7,
            label=str(cls),
            linestyle="",
        )
        for cls in classes
    ]
    g.axes[0, 0].legend(
        handles=legend_handles, title="class", loc="best", framealpha=0.9
    )

    g.figure.suptitle(
        _pca_pairgrid_title(
            subtitle=subtitle,
            plot_df=plot_df,
            class_col=class_col,
            scatter_df=scatter_df,
            kde_df=kde_df,
        ),
        y=1.02,
        fontsize=10,
    )
    g.figure.tight_layout()
    _save_figure_multi_format(g.figure, output_path)
    plt.close(g.figure)
    return True


_SANKEY_DEFAULT_INCHES_PER_LABEL = 0.4
_SANKEY_DEFAULT_MIN_FIG_HEIGHT = 6.0
_SANKEY_DEFAULT_MIN_BAND_HEIGHT = 0.15


def _scale_sankey_weights_for_min_band(
    work: pd.DataFrame,
    weights: np.ndarray,
    *,
    left_column: str,
    min_band_height: float,
) -> np.ndarray:
    """
    Uniformly scale flow weights so the thinnest node band meets a minimum height.

    pySankey band thickness is proportional to mass; small totals cram labels even
    when the figure is tall. Scaling preserves relative flow widths.
    """
    if min_band_height <= 0:
        return weights
    left_totals = work.groupby(left_column, sort=False)["count"].sum()
    right_totals = work.groupby("cluster", sort=False)["count"].sum()
    if left_totals.empty or right_totals.empty:
        return weights
    min_band = float(min(left_totals.min(), right_totals.min()))
    if min_band <= 0:
        return weights
    scale = max(1.0, min_band_height / min_band)
    if scale <= 1.0:
        return weights
    return weights * scale


def plot_cluster_entity_sankey(
    flow_df: pd.DataFrame,
    *,
    save_dir: pathlib.Path,
    basename: str = "fit_cluster_entity_sankey",
    max_clusters: int | None = None,
    max_entities: int | None = None,
    min_within_cluster_fraction: float = 0.0,
    cluster_labels: dict[int, str] | None = None,
    cluster_label_include_id: bool = False,
    inches_per_label: float = _SANKEY_DEFAULT_INCHES_PER_LABEL,
    min_fig_height: float = _SANKEY_DEFAULT_MIN_FIG_HEIGHT,
    min_band_height: float = _SANKEY_DEFAULT_MIN_BAND_HEIGHT,
) -> list[pathlib.Path]:
    """
    Bipartite KB-in entity→cluster Sankey from a long flow table (cluster, entity, count).

    Expects unbundled entity mass (no pie/bar ``Other (...)`` rows). Caps by dropping
    the long tail. pySankey sizes bands by weight only; ``inches_per_label`` sets
    figure height; ``min_band_height`` uniformly scales weights for readability.
    """
    if flow_df.empty:
        return []
    from pelinker.clustering.composition import limit_entity_flow_for_plots

    work = limit_entity_flow_for_plots(
        flow_df,
        max_clusters=max_clusters,
        max_entities=max_entities,
        min_within_cluster_fraction=min_within_cluster_fraction,
    )
    if work.empty:
        return []
    work = work.copy()
    work["entity"] = work["entity"].astype(str)
    if cluster_labels:
        work["cluster"] = (
            work["cluster"]
            .astype(int)
            .map(
                lambda cid: _format_cluster_sankey_label(
                    int(cid),
                    cluster_labels,
                    include_id=cluster_label_include_id,
                )
            )
        )
    else:
        work["cluster"] = work["cluster"].astype(str)
    left = work["entity"].to_numpy()
    right = work["cluster"].to_numpy()
    weights = work["count"].astype(float).to_numpy()
    weights = _scale_sankey_weights_for_min_band(
        work,
        weights,
        left_column="entity",
        min_band_height=min_band_height,
    )

    n_labels = max(work["entity"].nunique(), work["cluster"].nunique())
    fig_height = max(min_fig_height, n_labels * inches_per_label)

    # pySankey always calls plt.figure() internally; figure_name is a path string, not a Figure.
    sankey(
        left,
        right,
        leftWeight=weights,
        aspect=12,
        fontsize=10,
        closePlot=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(14, fig_height)
    written: list[pathlib.Path] = []
    for path in _save_figure_multi_format(fig, save_dir / basename):
        written.append(path)
    plt.close(fig)
    return written


def _format_cluster_sankey_label(
    cluster_id: int,
    cluster_labels: dict[int, str],
    *,
    include_id: bool,
) -> str:
    label = cluster_labels.get(cluster_id, str(cluster_id))
    if include_id:
        return f"{label} [{cluster_id}]"
    return label


_WORD_SPAN_RE = re.compile(r"\S+")


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD_SPAN_RE.finditer(text)]


def mention_context_window(
    text: str,
    a_abs: int,
    b_abs: int,
    *,
    words_before: int = 5,
    words_after: int = 5,
) -> str:
    """Return a short phrase around ``[a_abs, b_abs)`` with the mention marked «…»."""
    spans = _word_spans(text)
    if not spans:
        return ""

    start_word: int | None = None
    end_word: int | None = None
    for i, (wa, wb) in enumerate(spans):
        if wb <= a_abs:
            continue
        if wa >= b_abs:
            break
        if start_word is None:
            start_word = i
        end_word = i

    if start_word is None:
        for i, (wa, _wb) in enumerate(spans):
            if wa >= a_abs:
                start_word = i
                end_word = i
                break
        if start_word is None:
            return ""

    assert end_word is not None
    win_start = max(0, start_word - words_before)
    win_end = min(len(spans) - 1, end_word + words_after)
    parts: list[str] = []

    for i in range(win_start, win_end + 1):
        wa, wb = spans[i]
        word = text[wa:wb]

        if i == start_word:
            word = f"<b>{word}"

        if i == end_word:
            word = f"{word}</b>"

        parts.append(word)
    return " ".join(parts)


def load_pmid_texts(
    table_path: str | pathlib.Path,
    pmids: set[str],
    *,
    chunk_size: int = 10_000,
) -> dict[str, str]:
    """Stream a PMID/text table and return rows for the requested ``pmids`` only."""
    from pelinker.data.tables import load_pmid_texts_from_table

    return load_pmid_texts_from_table(table_path, pmids, chunk_size=chunk_size)


def enrich_fit_cluster_viz_plot_df_with_context(
    plot_df: pd.DataFrame,
    pmid_text_table_path: str | pathlib.Path,
    *,
    words_before: int = 5,
    words_after: int = 5,
) -> pd.DataFrame:
    """Add a ``context`` column: five words before/after each mention (from ``a_abs``/``b_abs``)."""
    required = {"pmid", "a_abs", "b_abs"}
    if not required.issubset(plot_df.columns):
        return plot_df

    pmid_texts = load_pmid_texts(
        pmid_text_table_path,
        {str(p) for p in plot_df["pmid"].astype(str).unique()},
    )
    contexts: list[str] = []
    for pmid, a_abs, b_abs in zip(
        plot_df["pmid"].astype(str),
        plot_df["a_abs"],
        plot_df["b_abs"],
        strict=True,
    ):
        text = pmid_texts.get(pmid)
        if text is None or pd.isna(a_abs) or pd.isna(b_abs):
            contexts.append("")
            continue
        contexts.append(
            mention_context_window(
                text,
                int(a_abs),
                int(b_abs),
                words_before=words_before,
                words_after=words_after,
            )
        )

    out = plot_df.copy()
    out["context"] = contexts
    return out


CLUSTER_VIZ_MEMBERSHIP_COLUMNS = (
    "clustering_in_sample",
    "screener_pass",
    "manifold_oov_pass",
)


def filter_assignments_for_cluster_viz(
    assign: pd.DataFrame,
    *,
    exclude_noise: bool = True,
    hdbscan_fit_scope: bool = True,
) -> pd.DataFrame:
    """Restrict cluster viz rows to HDBSCAN-fit + screener/OOV-pass mentions when flagged."""
    from pelinker.clustering.composition import filter_emergent_assignments

    out = assign.copy()
    if exclude_noise:
        out = filter_emergent_assignments(out)
    if not hdbscan_fit_scope:
        return out
    present = [c for c in CLUSTER_VIZ_MEMBERSHIP_COLUMNS if c in out.columns]
    if not present:
        return out
    mask = np.ones(len(out), dtype=bool)
    for col in present:
        mask &= out[col].astype(bool).to_numpy()
    return out.loc[mask]


def build_fit_cluster_viz_plot_df(
    report: ModelSelectionReport,
    *,
    exclude_noise: bool = True,
    hdbscan_fit_scope: bool = True,
    cluster_labels: dict[int, str] | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """Build a :func:`plot_cluster_viz` frame from a :class:`~pelinker.reports.schema.ModelSelectionReport`."""
    cluster_viz = report.cluster_viz
    if cluster_viz is None or cluster_viz.size == 0 or cluster_viz.shape[1] < 1:
        return None, report.cluster_viz_method
    assign = filter_assignments_for_cluster_viz(
        report.assignments,
        exclude_noise=exclude_noise,
        hdbscan_fit_scope=hdbscan_fit_scope,
    )
    if len(assign) == 0:
        return None, report.cluster_viz_method
    n_dims = int(cluster_viz.shape[1])
    viz_cols = [f"cviz_{j:02d}" for j in range(n_dims)]
    viz_df = pd.DataFrame(
        cluster_viz, columns=viz_cols, index=report.assignments.index
    ).loc[assign.index]
    assign = assign.copy()
    if "cluster" in assign.columns:
        assign["cluster_id"] = assign["cluster"].astype(int)
    rename: dict[str, str] = {}
    if "cluster" in assign.columns:
        rename["cluster"] = "class"
    plot_assign = assign.rename(columns=rename)
    if cluster_labels and "cluster_id" in plot_assign.columns:
        plot_assign["class"] = plot_assign["cluster_id"].map(
            lambda cid: cluster_labels.get(int(cid), str(cid))
        )
    cols = [c for c in ("entity", "class") if c in plot_assign.columns]
    extra = [
        c
        for c in (
            "cluster_id",
            "pmid",
            "mention",
            "a_abs",
            "b_abs",
            "screener_score",
            "cluster_score",
        )
        if c in plot_assign.columns
    ]
    return pd.concat(
        [plot_assign[cols + extra], viz_df], axis=1
    ), report.cluster_viz_method


def plot_scale_curve(
    curve: ScaleCurve,
    output_path: pathlib.Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> tuple[pathlib.Path, ...]:
    """Log-log plot of chosen ``min_cluster_size`` vs realized N, with the fitted law.

    Rungs whose chosen value sat on a grid bound are drawn hollow and red: on those
    points the grid, not the sample size, decided the answer, and the slope through
    them is not evidence.
    """
    ns = np.array([r.n_rows_realized for r in curve.rungs], dtype=np.float64)
    mcs = np.array([r.chosen_min_cluster_size for r in curve.rungs], dtype=np.float64)
    pinned = np.array([r.is_pinned for r in curve.rungs], dtype=bool)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Fitted law across the measured span, extended by 2x to show the extrapolation.
    x_line = np.logspace(
        np.log10(ns.min()), np.log10(ns.max() * 2.0), num=100, dtype=np.float64
    )
    y_line = np.exp(curve.log_intercept + curve.log_slope * np.log(x_line))
    ax.plot(
        x_line,
        y_line,
        color="#3b6ea5",
        linewidth=1.8,
        label=(
            f"fit: log(MCS) = {curve.log_intercept:.3f} + "
            f"{curve.log_slope:.3f}·log(N)   (R²={curve.r_squared:.3f})"
        ),
    )
    ax.axvspan(
        ns.max(),
        x_line.max(),
        color="#999999",
        alpha=0.12,
        label="extrapolation (beyond measured N)",
    )

    if np.any(~pinned):
        ax.scatter(
            ns[~pinned],
            mcs[~pinned],
            s=70,
            color="#3b6ea5",
            edgecolor="black",
            zorder=3,
            label="measured rung",
        )
    if np.any(pinned):
        ax.scatter(
            ns[pinned],
            mcs[pinned],
            s=90,
            facecolor="none",
            edgecolor="#c0392b",
            linewidth=2.0,
            zorder=4,
            label="pinned on grid bound (not evidence)",
        )

    lo = curve.rungs[0].grid_min_scale
    hi = curve.rungs[0].grid_max_scale
    if lo is not None:
        ax.axhline(lo, color="#c0392b", linestyle=":", linewidth=1.0, alpha=0.7)
    if hi is not None:
        ax.axhline(hi, color="#c0392b", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mention rows clustered (realized N)")
    ax.set_ylabel("chosen min_cluster_size")
    ax.set_title("min_cluster_size scaling with sample size")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    written = _save_figure_multi_format(fig, output_path, formats=formats)
    plt.close(fig)
    return written
