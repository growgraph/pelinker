"""
Join and concatenate embeddings from multiple parquet sources (same KB export schema).

Mention-level fusion joins on (pmid, property, mention) and concatenates embed vectors
in source order. Property-level fusion averages mentions within each file per property,
takes the intersection of properties across sources, and concatenates those means.

``Linker.predict`` fuses mention tensors across sources in metadata order; when
``model_type`` differs between sources, span alignment is not guaranteed—use the same
backbone for all sources if order-sensitive fusion is required.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

JOIN_KEYS: tuple[str, str, str] = ("pmid", "property", "mention")

REQUIRED_MENTION_COLUMNS: frozenset[str] = frozenset(
    {"pmid", "property", "mention", "embed"}
)


def _to_embed_array(v: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr.reshape(-1) if arr.ndim > 1 and arr.shape[0] == 1 else arr.squeeze()


def dedupe_mean_embed_by_keys(
    df: pd.DataFrame, *, keys: Sequence[str] = JOIN_KEYS
) -> pd.DataFrame:
    """
    Collapse duplicate rows sharing the same join keys by averaging embed vectors.
    """
    missing = [c for c in keys if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    def _mean_group(embed_series: pd.Series) -> np.ndarray:
        vecs = np.stack([_to_embed_array(x) for x in embed_series.values])
        return np.mean(vecs, axis=0)

    grouped = (
        df.groupby(list(keys), sort=False)
        .agg(embed=("embed", _mean_group))
        .reset_index()
    )
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
        cols = set(df.columns)
        if not REQUIRED_MENTION_COLUMNS.issubset(cols):
            missing = REQUIRED_MENTION_COLUMNS - cols
            raise ValueError(f"Frame {i} missing columns: {sorted(missing)}")
        sub = df[list(JOIN_KEYS) + ["embed"]].copy()
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


def mean_embedding_per_property(
    df: pd.DataFrame,
    kb_labels: set[str] | None,
    *,
    embed_column: str = "embed",
    property_column: str = "property",
) -> dict[str, np.ndarray]:
    """
    Average all mention embeddings per property label (same logic as Linker single-file load).
    """
    if kb_labels is not None:
        work = df[df[property_column].isin(kb_labels)].copy()
    else:
        work = df.copy()
    if len(work) == 0:
        return {}

    work["_emb_arr"] = work[embed_column].apply(
        lambda x: np.asarray(x, dtype=np.float64)
    )
    out: dict[str, np.ndarray] = {}
    for prop_label, group in work.groupby(property_column):
        stacked = np.stack(group["_emb_arr"].tolist())
        out[str(prop_label)] = np.mean(stacked, axis=0)
    return out


def property_mean_vectors_per_parquet(
    path: pathlib.Path,
    kb_labels: set[str] | None,
) -> dict[str, np.ndarray]:
    df = read_parquet_to_dataframe(path)
    return mean_embedding_per_property(df, kb_labels)


def fused_property_vectors_from_paths(
    paths: Sequence[pathlib.Path],
    kb_labels: set[str] | None,
) -> dict[str, np.ndarray]:
    """
    Per-source per-property means, then concatenate vectors for properties in the intersection.

    Source order matches ``paths`` order (must align with ``EmbeddingModelMetadata.sources``).
    """
    if len(paths) == 0:
        raise ValueError("paths must be non-empty")
    per_source = [property_mean_vectors_per_parquet(p, kb_labels) for p in paths]
    common = set(per_source[0].keys())
    for d in per_source[1:]:
        common &= set(d.keys())
    if not common:
        return {}

    fused: dict[str, np.ndarray] = {}
    for prop in sorted(common):
        parts = [per_source[i][prop] for i in range(len(paths))]
        fused[prop] = np.concatenate(parts, axis=0)
    return fused


def property_fused_dataframe_for_linker_order(
    fused_vectors: Mapping[str, np.ndarray],
    labels_map: Mapping[str, str],
) -> pd.DataFrame:
    """
    Rows sorted like Linker._load_embeddings_from_file: sorted property labels that
    resolve to an entity_id in labels_map.
    """
    rows: list[dict[str, object]] = []
    for prop_label in sorted(fused_vectors.keys()):
        entity_id = None
        for eid, label in labels_map.items():
            if label == prop_label:
                entity_id = eid
                break
        if entity_id is None:
            continue
        vec = fused_vectors[prop_label]
        rows.append(
            {"property": prop_label, "embed": vec.astype(np.float32, copy=False)}
        )
    if not rows:
        return pd.DataFrame(columns=["property", "embed"])
    return pd.DataFrame(rows)
