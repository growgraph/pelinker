"""
Join and concatenate embeddings from multiple parquet sources (same KB export schema).

Mention-level fusion joins on (pmid, entity, mention) and concatenates embed vectors
in source order. Entity-level fusion averages mentions within each file per entity,
takes the intersection of entities across sources, and concatenates those means.

``Linker.predict`` fuses mention tensors across sources in metadata order; when
``model_type`` differs between sources, span alignment is not guaranteed—use the same
backbone for all sources if order-sensitive fusion is required.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pelinker.io import read_batches

logger = logging.getLogger(__name__)


def _parquet_capped_batch_count(
    path: pathlib.Path,
    batch_size: int,
    n_embedding_batches: int | None,
) -> int | None:
    """Return expected batch count for progress, or None if metadata is unavailable."""
    try:
        file = pq.ParquetFile(path.as_posix())
        rows = int(file.metadata.num_rows)
    except Exception:
        return None
    if rows <= 0:
        return None
    full = (rows + batch_size - 1) // batch_size
    if n_embedding_batches is not None:
        return min(full, n_embedding_batches)
    return full


def _should_emit_read_progress(batch_index: int, total_batches: int | None) -> bool:
    if total_batches is None or total_batches <= 1:
        return True
    step = max(1, total_batches // 50)
    return batch_index == 1 or batch_index % step == 0 or batch_index >= total_batches


def _maybe_emit_read_line(
    path: pathlib.Path,
    batch_idx: int,
    total_batches: int | None,
    status_fn: Callable[[str], None] | None,
) -> None:
    if status_fn is None:
        return
    if not _should_emit_read_progress(batch_idx, total_batches):
        return
    if total_batches is not None:
        status_fn(f"batch {batch_idx}/{total_batches}")
    else:
        status_fn(f"batch {batch_idx}")


def _for_each_embedding_parquet_batch(
    path: pathlib.Path,
    batch_size: int,
    n_embedding_batches: int | None,
    *,
    read_status: Callable[[str], None] | None,
    show_read_progress: bool,
    on_batch: Callable[[int, pd.DataFrame], None],
) -> None:
    """Iterate parquet row batches; optional outer status or standalone Rich progress."""
    total_batches = _parquet_capped_batch_count(path, batch_size, n_embedding_batches)

    def _run(status_fn: Callable[[str], None] | None) -> None:
        for i, batch in enumerate(read_batches(path.as_posix(), batch_size=batch_size)):
            on_batch(i, batch)
            _maybe_emit_read_line(path, i + 1, total_batches, status_fn)
            if n_embedding_batches is not None and i >= n_embedding_batches - 1:
                break

    if read_status is not None:
        _run(read_status)
    elif show_read_progress:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=Console(force_terminal=True),
            transient=True,
            refresh_per_second=4,
        ) as progress:
            tid = progress.add_task(
                f"Reading {path.name}",
                total=total_batches,
            )

            def _bar_status(msg: str) -> None:
                if total_batches is not None:
                    try:
                        part = msg.split("batch ", 1)[1]
                        num = int(part.split("/", 1)[0])
                    except (IndexError, ValueError):
                        num = None
                    if num is not None:
                        progress.update(tid, description=msg, completed=num)
                        return
                progress.update(tid, description=msg)

            _run(_bar_status)
    else:
        _run(None)


JOIN_KEYS: tuple[str, str, str] = ("pmid", "entity", "mention")

REQUIRED_MENTION_COLUMNS: frozenset[str] = frozenset(
    {"pmid", "entity", "mention", "embed"}
)

MENTION_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "a",
    "b",
    "a_abs",
    "b_abs",
    "itext",
    "ichunk",
)


def _normalize_entity_column(df: pd.DataFrame) -> pd.DataFrame:
    if "entity" not in df.columns:
        raise ValueError("DataFrame must contain an 'entity' column")
    return df


def _to_embed_array(v: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr.reshape(-1) if arr.ndim > 1 and arr.shape[0] == 1 else arr.squeeze()


def provenance_columns_in(df: pd.DataFrame) -> list[str]:
    """Return provenance column names present in ``df``."""
    return [c for c in MENTION_PROVENANCE_COLUMNS if c in df.columns]


def _first_non_null(series: pd.Series) -> object:
    for v in series:
        if pd.notna(v):
            return v
    return pd.NA


def dedupe_mean_embed_by_keys(
    df: pd.DataFrame, *, keys: Sequence[str] = JOIN_KEYS
) -> pd.DataFrame:
    """
    Collapse duplicate rows sharing the same join keys by averaging embed vectors.

    Optional :data:`MENTION_PROVENANCE_COLUMNS` are kept via first non-null per group.
    """
    missing = [c for c in keys if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    def _mean_group(embed_series: pd.Series) -> np.ndarray:
        vecs = np.stack([_to_embed_array(x) for x in embed_series.values])
        return np.mean(vecs, axis=0)

    agg: dict[str, tuple[str, object]] = {"embed": ("embed", _mean_group)}
    for col in provenance_columns_in(df):
        agg[col] = (col, _first_non_null)

    grouped = df.groupby(list(keys), sort=False).agg(**agg).reset_index()
    return grouped


def mention_level_concat_frames(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    Inner-join mention rows across sources and set ``embed`` to concatenated vectors.

    Duplicate keys within a single source are averaged before the join.
    Concatenation order matches ``dfs`` index order (must match metadata.sources).
    """
    if len(dfs) == 0:
        raise ValueError("mention_level_concat_frames requires at least one frame")
    prepared: list[pd.DataFrame] = []
    for i, df in enumerate(dfs):
        df = _normalize_entity_column(df)
        cols = set(df.columns)
        if not REQUIRED_MENTION_COLUMNS.issubset(cols):
            missing = REQUIRED_MENTION_COLUMNS - cols
            raise ValueError(f"Frame {i} missing columns: {sorted(missing)}")
        keep = list(JOIN_KEYS) + ["embed"]
        if i == 0:
            keep.extend(provenance_columns_in(df))
        sub = df[keep].copy()
        sub = dedupe_mean_embed_by_keys(sub)
        sub = sub.rename(columns={"embed": f"_e{i}"})
        prepared.append(sub)

    out = prepared[0]
    for i in range(1, len(prepared)):
        out = out.merge(prepared[i], on=list(JOIN_KEYS), how="inner")
        if len(out) == 0:
            break

    emb_cols = [f"_e{i}" for i in range(len(dfs))]
    if len(out) == 0:
        return pd.DataFrame(columns=list(JOIN_KEYS) + ["embed"])

    if len(dfs) == 1:
        return out.rename(columns={"_e0": "embed"})

    def _concat_row(row: pd.Series) -> np.ndarray:
        parts = [np.asarray(row[c], dtype=np.float32) for c in emb_cols]
        return np.concatenate(parts, axis=0)

    out = out.copy()
    out["embed"] = [_concat_row(out.iloc[i]) for i in range(len(out))]
    out = out.drop(columns=emb_cols)
    return out


def read_parquet_to_dataframe(path: pathlib.Path) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()


def read_embedding_parquet_batches_concat(
    path: pathlib.Path,
    *,
    batch_size: int,
    n_embedding_batches: int | None = None,
    read_status: Callable[[str], None] | None = None,
    show_read_progress: bool = False,
) -> pd.DataFrame | None:
    """
    Stream mention-level embedding parquet via ``read_batches`` and concatenate batches.

    Same batching contract as ``ClusteringOptimizationConfig`` (``batch_size`` rows per
    batch; optional cap on batch count).

    If ``read_status`` is set, it receives short status lines (for combining with an
    outer Rich progress bar). If it is omitted and ``show_read_progress`` is True, a
    compact transient progress display is used for this read only.
    """
    if not path.exists():
        return None
    agg: list[pd.DataFrame] = []
    try:

        def on_batch(_i: int, batch: pd.DataFrame) -> None:
            agg.append(batch)

        _for_each_embedding_parquet_batch(
            path,
            batch_size,
            n_embedding_batches,
            read_status=read_status,
            show_read_progress=show_read_progress,
            on_batch=on_batch,
        )
    except Exception:
        return None
    if not agg:
        return None
    return pd.concat(agg, ignore_index=True)


def concat_mention_level_embedding_sources(
    paths: Sequence[pathlib.Path],
    *,
    batch_size: int,
    n_embedding_batches: int | None = None,
    read_status: Callable[[str], None] | None = None,
    show_read_progress: bool = False,
) -> pd.DataFrame | None:
    """
    Load one or more mention-level parquet sources like
    :func:`~pelinker.selection.load_selection_frame`:
    read each path in batches, optionally inner-join across sources, return one frame
    (no ``frac`` sampling).
    """
    if len(paths) == 0:
        return None
    parts: list[pd.DataFrame] = []
    n_paths = len(paths)
    for pi, p in enumerate(paths):
        prefix = f"[{pi + 1}/{n_paths}] " if n_paths > 1 else ""

        def path_read(msg: str, _pfx: str = prefix) -> None:
            if read_status is not None:
                read_status(_pfx + msg)

        part = read_embedding_parquet_batches_concat(
            p,
            batch_size=batch_size,
            n_embedding_batches=n_embedding_batches,
            read_status=path_read if read_status is not None else None,
            show_read_progress=show_read_progress and read_status is None,
        )
        if part is None or len(part) == 0:
            return None
        parts.append(part)
    if len(parts) == 1:
        frame = parts[0]
    else:
        try:
            frame = mention_level_concat_frames(parts)
        except Exception:
            return None
    if len(frame) == 0:
        return None
    return frame


def _accumulate_property_means_from_frame(
    sums: dict[str, np.ndarray],  # legacy name; keys are entity labels
    counts: dict[str, int],
    df: pd.DataFrame,
    kb_labels: set[str] | None,
    *,
    embed_column: str = "embed",
    entity_column: str = "entity",
) -> None:
    df = _normalize_entity_column(df)
    if entity_column not in df.columns or embed_column not in df.columns:
        return
    if kb_labels is not None:
        work = df.loc[df[entity_column].isin(kb_labels)]
    else:
        work = df
    if len(work) == 0:
        return
    for entity_label, group in work.groupby(entity_column, sort=False):
        key = str(entity_label)
        stacked = np.stack(
            [
                np.asarray(x, dtype=np.float64).reshape(-1)
                for x in group[embed_column].values
            ]
        )
        batch_sum = stacked.sum(axis=0)
        batch_count = int(stacked.shape[0])
        if key in sums:
            sums[key] += batch_sum
            counts[key] += batch_count
        else:
            sums[key] = batch_sum
            counts[key] = batch_count


def property_mean_vectors_from_parquet_batches(
    path: pathlib.Path,
    kb_labels: set[str] | None,
    *,
    batch_size: int,
    n_embedding_batches: int | None = None,
    read_status: Callable[[str], None] | None = None,
    show_read_progress: bool = False,
) -> dict[str, np.ndarray]:
    """Mean embedding per entity by streaming parquet batches (no full-file read)."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    def on_batch(_i: int, batch: pd.DataFrame) -> None:
        _accumulate_property_means_from_frame(sums, counts, batch, kb_labels)

    _for_each_embedding_parquet_batch(
        path,
        batch_size,
        n_embedding_batches,
        read_status=read_status,
        show_read_progress=show_read_progress,
        on_batch=on_batch,
    )
    return {k: sums[k] / counts[k] for k in sums}


def mean_embedding_per_property(
    df: pd.DataFrame,
    kb_labels: set[str] | None,
    *,
    embed_column: str = "embed",
    entity_column: str = "entity",
) -> dict[str, np.ndarray]:
    """
    Average all mention embeddings per entity label (same logic as Linker single-file load).
    """
    df = _normalize_entity_column(df)
    if kb_labels is not None:
        work = df[df[entity_column].isin(kb_labels)].copy()
    else:
        work = df.copy()
    if len(work) == 0:
        return {}

    work["_emb_arr"] = work[embed_column].apply(
        lambda x: np.asarray(x, dtype=np.float64)
    )
    out: dict[str, np.ndarray] = {}
    for entity_label, group in work.groupby(entity_column):
        stacked = np.stack(group["_emb_arr"].tolist())
        out[str(entity_label)] = np.mean(stacked, axis=0)
    return out


def property_mean_vectors_per_parquet(
    path: pathlib.Path,
    kb_labels: set[str] | None,
    *,
    batch_size: int = 1000,
    n_embedding_batches: int | None = None,
) -> dict[str, np.ndarray]:
    return property_mean_vectors_from_parquet_batches(
        path,
        kb_labels,
        batch_size=batch_size,
        n_embedding_batches=n_embedding_batches,
    )


def fused_property_vectors_from_paths(
    paths: Sequence[pathlib.Path],
    kb_labels: set[str] | None,
    *,
    batch_size: int = 1000,
    n_embedding_batches: int | None = None,
    read_status: Callable[[str], None] | None = None,
    show_read_progress: bool = False,
) -> dict[str, np.ndarray]:
    """
    Per-source per-entity means, then concatenate vectors for entities in the intersection.

    Source order matches ``paths`` order (must align with ``EmbeddingModelMetadata.sources``).
    Parquet inputs are read in batches (same mechanism as clustering analysis on loaded frames).
    """
    if len(paths) == 0:
        raise ValueError("paths must be non-empty")
    n_paths = len(paths)
    per_source: list[dict[str, np.ndarray]] = []
    for pi, p in enumerate(paths):
        prefix = f"[{pi + 1}/{n_paths}] " if n_paths > 1 else ""

        def path_read(msg: str, _pfx: str = prefix) -> None:
            if read_status is not None:
                read_status(_pfx + msg)

        per_source.append(
            property_mean_vectors_from_parquet_batches(
                p,
                kb_labels,
                batch_size=batch_size,
                n_embedding_batches=n_embedding_batches,
                read_status=path_read if read_status is not None else None,
                show_read_progress=show_read_progress and read_status is None,
            )
        )
    common = set(per_source[0].keys())
    for d in per_source[1:]:
        common &= set(d.keys())
    if not common:
        return {}

    fused: dict[str, np.ndarray] = {}
    for entity in sorted(common):
        parts = [per_source[i][entity] for i in range(len(paths))]
        fused[entity] = np.concatenate(parts, axis=0)
    return fused


def property_fused_dataframe_for_linker_order(
    fused_vectors: Mapping[str, np.ndarray],
    labels_map: Mapping[str, str],
) -> pd.DataFrame:
    """
    Rows sorted like Linker fused load: sorted entity labels that resolve to an
    ``entity_id`` in ``labels_map``. Columns: ``entity_id``, ``entity``, ``embed``.
    """
    rows: list[dict[str, object]] = []
    for entity_label in sorted(fused_vectors.keys()):
        entity_id = None
        for eid, label in labels_map.items():
            if label == entity_label:
                entity_id = eid
                break
        if entity_id is None:
            logger.warning(
                "Entity label '%s' not found in labels_map, skipping", entity_label
            )
            continue
        vec = fused_vectors[entity_label]
        rows.append(
            {
                "entity_id": entity_id,
                "entity": entity_label,
                "embed": vec.astype(np.float32, copy=False),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["entity_id", "entity", "embed"])
    return pd.DataFrame(rows)
