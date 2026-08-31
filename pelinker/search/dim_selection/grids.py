"""PCA / UMAP dimension search grids and winner selection."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

DEFAULT_PCA_GRID: tuple[int, ...] = (40, 80, 120, 180)
DEFAULT_UMAP_GRID: tuple[int, ...] = (4, 6, 8, 12)

REFINE_PCA_DELTA = 40
REFINE_PCA_STEP = 20
REFINE_UMAP_DELTA = 2
REFINE_UMAP_STEP = 1
MIN_PCA_COMPONENTS = 2
MIN_UMAP_COMPONENTS = 2


def parse_int_grid(spec: str | Sequence[int], *, name: str) -> tuple[int, ...]:
    """Parse a comma-separated int grid or pass through a sequence of ints."""
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{name} grid must be non-empty")
        values = [int(p) for p in parts]
    else:
        values = [int(v) for v in spec]
    if not values:
        raise ValueError(f"{name} grid must be non-empty")
    if any(v < 1 for v in values):
        raise ValueError(f"{name} grid values must be >= 1; got {values}")
    # Preserve order, drop duplicates.
    seen: set[int] = set()
    out: list[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return tuple(out)


def cell_key(pca_components: int, umap_dim: int) -> str:
    return f"pca{int(pca_components)}/umap{int(umap_dim)}"


def parse_cell_key(key: str) -> tuple[int, int]:
    if "/umap" not in key or not key.startswith("pca"):
        raise ValueError(f"invalid dim-selection cell key: {key!r}")
    pca_part, umap_part = key.split("/umap", 1)
    return int(pca_part.removeprefix("pca")), int(umap_part)


def coarse_cells(
    pca_grid: Sequence[int],
    umap_grid: Sequence[int],
) -> list[tuple[int, int]]:
    """Cartesian product of coarse PCA × UMAP grids (stable order)."""
    return [(int(p), int(u)) for p in pca_grid for u in umap_grid]


def _range_around(
    center: int,
    *,
    delta: int,
    step: int,
    lo: int,
) -> list[int]:
    start = max(lo, int(center) - int(delta))
    end = int(center) + int(delta)
    vals = list(range(start, end + 1, int(step)))
    if center not in vals:
        vals.append(int(center))
    return sorted(set(vals))


def refine_cells(
    best_pca: int,
    best_umap: int,
    *,
    already: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """
    Local neighborhood around the coarse winner.

    PCA: best±40 step 20 (clipped to >= 2).
    UMAP: best±2 step 1 (clipped to >= 2).
    Skips cells already evaluated when ``already`` is provided.
    """
    done = already or set()
    pcas = _range_around(
        best_pca,
        delta=REFINE_PCA_DELTA,
        step=REFINE_PCA_STEP,
        lo=MIN_PCA_COMPONENTS,
    )
    umaps = _range_around(
        best_umap,
        delta=REFINE_UMAP_DELTA,
        step=REFINE_UMAP_STEP,
        lo=MIN_UMAP_COMPONENTS,
    )
    out: list[tuple[int, int]] = []
    for p in pcas:
        for u in umaps:
            # UMAP dim must not exceed PCA dim for a sensible pipeline.
            if u > p:
                continue
            cell = (p, u)
            if cell not in done:
                out.append(cell)
    return out


def cluster_viz_components_for_umap(umap_dim: int) -> int:
    return min(3, int(umap_dim))


def pick_winner_row(df_results: pd.DataFrame) -> dict[str, object]:
    """
    Choose the best (pca, umap) row by outer DBCV+ARI score.

    ``outer_score`` = min–max pooled mean DBCV + mean ARI across candidate cells
    (same formula as inner ``dbcv_ari_geomean``). Ties: lower outer std, then
    smaller ``pca_components``, then smaller ``umap_dim``.
    """
    from pelinker.clustering.ranking import pick_best_row

    if df_results.empty:
        raise ValueError("df_results must be a non-empty DataFrame")
    required = {
        "best_score",
        "best_score_std",
        "pca_components",
        "umap_dim",
    }
    missing = required - set(df_results.columns)
    if missing:
        raise ValueError(f"df_results missing columns: {sorted(missing)}")

    return pick_best_row(
        df_results,
        tie_break_cols=("pca_components", "umap_dim"),
    )
