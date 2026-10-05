from __future__ import annotations


import numpy as np
import pandas as pd
import torch


"""Mention-frame shaping helpers shared by fit and search."""


def drop_entities_with_few_mentions(
    frame: pd.DataFrame,
    min_mentions_per_entity: int,
    *,
    negative_label: str | None = None,
) -> pd.DataFrame:
    """
    Drop entities with fewer than ``min_mentions_per_entity`` rows (same rule as
    :func:`~pelinker.search.selection.load_selection_frame` / mention-level selection eval).

    When ``negative_label`` is set, that label is never dropped for low mention count
    (so thin negative tails remain for screener training).
    """
    if "entity" not in frame.columns:
        raise ValueError("frame must contain an 'entity' column")
    mention_count = frame["entity"].value_counts()
    low_count = mention_count[
        ~(mention_count >= min_mentions_per_entity)
    ].index.to_list()
    if negative_label is not None:
        low_count = [e for e in low_count if e != negative_label]
    return frame.loc[~frame["entity"].isin(low_count)].copy()


def embeddings_dict_to_dataframe(
    embeddings_dict: dict[str, tuple[str, torch.Tensor | np.ndarray]],
) -> pd.DataFrame:
    """
    Convert embeddings dictionary to DataFrame format expected by transform artifacts.

    Args:
        embeddings_dict: Dictionary mapping id -> (label, embedding)

    Returns:
        DataFrame with columns: id, label, embed
    """
    embeddings_list = []
    id_list = []
    label_list = []

    for id_val, (label, emb) in embeddings_dict.items():
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy()
        else:
            emb_np = np.array(emb)
        embeddings_list.append(emb_np)
        id_list.append(id_val)
        label_list.append(label)

    return pd.DataFrame({"id": id_list, "label": label_list, "embed": embeddings_list})


def entity_negative_label_mask_01(
    entities: pd.Series | np.ndarray,
    negative_label: str,
) -> np.ndarray:
    """
    Per-row binary labels aligned with ``entities``: ``1`` if the row's ``entity`` equals
    ``negative_label`` (same convention as the negative screener positive class), else ``0``.
    """
    if isinstance(entities, pd.Series):
        s = entities.astype(str).to_numpy()
    else:
        s = np.asarray(entities).astype(str)
    if s.size == 0:
        return np.zeros(0, dtype=np.int64)
    return (s == negative_label).astype(np.int64, copy=False)


def mention_quality_frame(
    dfr: pd.DataFrame,
    *,
    neg_mask: np.ndarray,
    cluster_kb: np.ndarray,
    pca_residuals: np.ndarray,
    pca_mahalanobis: np.ndarray,
    pca_spectral_entropy: np.ndarray,
    negative_label: str,
) -> pd.DataFrame:
    """Per-mention PCA quality and labels for all rows (KB clustered; negatives cluster=-1)."""
    optional = ["pmid", "mention"]
    optional_cols = [c for c in optional if c in dfr.columns]
    out = dfr[["entity", *optional_cols]].copy()
    cluster_full = np.full(len(dfr), -1, dtype=np.int64)
    cluster_full[~neg_mask] = np.asarray(cluster_kb, dtype=np.int64).ravel()
    out["cluster"] = cluster_full
    out["oov_label"] = entity_negative_label_mask_01(dfr["entity"], negative_label)
    out["pca_residual"] = np.asarray(pca_residuals, dtype=np.float64).ravel()
    out["pca_mahalanobis"] = np.asarray(pca_mahalanobis, dtype=np.float64).ravel()
    out["pca_spectral_entropy"] = np.asarray(
        pca_spectral_entropy, dtype=np.float64
    ).ravel()
    ordered = [
        "entity",
        *optional_cols,
        "cluster",
        "oov_label",
        "pca_residual",
        "pca_mahalanobis",
        "pca_spectral_entropy",
    ]
    return out[ordered]
