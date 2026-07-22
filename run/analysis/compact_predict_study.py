#!/usr/bin/env python3
"""Compare legacy vs compact predict paths (ParametricUMAP + MLP) for quality/size gates.

Arms
----
A legacy:       standard UMAP + HDBSCAN approximate_predict
B compact:      ParametricUMAP + MLP (shipped)
C iso-manifold: ParametricUMAP + HDBSCAN approximate_predict
D iso-head:     standard UMAP + MLP
E linear:       ParametricUMAP + LinearSVC (underfit control)

Gates compare **entity-id** decisions (majority-vote cluster→entity on train) vs arm A
on non-noise holdout rows. Prefer a real mention parquet over synthetic blobs.

Example
-------
uv run python run/analysis/compact_predict_study.py \\
  --embeddings-parquet data/derived/res_pubmedbert_1.parquet \\
  --report-dir /tmp/compact_study \\
  --min-cluster-size 20 \\
  --max-rows 20000
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hdbscan
import joblib
import numpy as np
import pandas as pd
from hdbscan import approximate_predict
from sklearn.metrics import adjusted_rand_score

from pelinker.config import TransformConfig
from pelinker.clustering_fit import fit_manifold_clustering
from pelinker.entity_head import (
    EntityHead,
    fit_linear_svc_entity_head,
    fit_mlp_entity_head,
)
from pelinker.onto import NEGATIVE_LABEL
from pelinker.transform import (
    EmbeddingTransformer,
    is_parametric_umap,
    save_clustering_manifold,
)


@dataclass(frozen=True)
class ArmResult:
    name: str
    agreement_vs_a: float
    ari_vs_a: float
    emit_rate: float
    score_corr_vs_a: float
    artifact_bytes: int
    predict_seconds: float
    n_clusters_emergent: int
    noise_fraction: float
    dbcv: float | None


def _load_embeds(path: Path, max_rows: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "embed" not in df.columns or "entity" not in df.columns:
        raise ValueError("parquet must contain embed and entity columns")
    df = df[df["entity"].astype(str) != NEGATIVE_LABEL].reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def _sizeof_joblib(obj: object) -> int:
    buf = __import__("io").BytesIO()
    joblib.dump(obj, buf, compress=3)
    return int(buf.tell())


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _manifold_artifact_bytes(transformer: EmbeddingTransformer) -> int:
    """PCA + clustering manifold size (ParametricUMAP via sidecar save, else joblib)."""
    pca_bytes = _sizeof_joblib(transformer.pca) if transformer.pca is not None else 0
    umap = transformer.umap
    if umap is None:
        return pca_bytes
    if is_parametric_umap(umap):
        with tempfile.TemporaryDirectory(prefix="pumap_size_") as tmp:
            out = Path(tmp) / "manifold"
            save_clustering_manifold(umap, out)
            return pca_bytes + _dir_bytes(out)
    return pca_bytes + _sizeof_joblib(umap)


def _split_idx(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tune = max(1, int(0.15 * n))
    n_hold = max(1, int(0.20 * n))
    tune = idx[:n_tune]
    hold = idx[n_tune : n_tune + n_hold]
    train = idx[n_tune + n_hold :]
    return train, tune, hold


def _majority_cluster_to_entity(
    cluster_labels: np.ndarray, entities: np.ndarray
) -> dict[int, str]:
    mapping: dict[int, str] = {}
    labels = np.asarray(cluster_labels, dtype=np.int64).ravel()
    ents = np.asarray(entities).astype(str)
    for cid in sorted({int(c) for c in labels if int(c) != -1}):
        mask = labels == cid
        counts = Counter(ents[mask].tolist())
        mapping[cid] = counts.most_common(1)[0][0]
    return mapping


def _labels_to_entities(
    cluster_labels: np.ndarray, mapping: dict[int, str]
) -> np.ndarray:
    out: list[str] = []
    for c in np.asarray(cluster_labels, dtype=np.int64).ravel():
        cid = int(c)
        if cid == -1 or cid not in mapping:
            out.append(NEGATIVE_LABEL)
        else:
            out.append(mapping[cid])
    return np.asarray(out, dtype=object)


def _entity_agreement(
    pred_clusters: np.ndarray,
    pred_map: dict[int, str],
    ref_clusters: np.ndarray,
    ref_map: dict[int, str],
) -> float:
    """Agreement of mapped entity ids on rows where the reference is non-noise."""
    ref_ent = _labels_to_entities(ref_clusters, ref_map)
    pred_ent = _labels_to_entities(pred_clusters, pred_map)
    mask = ref_ent != NEGATIVE_LABEL
    if not np.any(mask):
        return float("nan")
    return float(np.mean(pred_ent[mask] == ref_ent[mask]))


def _fit_manifold(
    df: pd.DataFrame,
    *,
    manifold_kind: str,
    pca: int,
    umap_dim: int,
    mcs: int,
    seed: int,
    parametric_epochs: int,
) -> tuple[
    EmbeddingTransformer, hdbscan.HDBSCAN, np.ndarray, np.ndarray, float | None, float
]:
    tc = TransformConfig(
        pca_components=pca,
        umap_components=umap_dim,
        cluster_viz_components=min(3, umap_dim),
        cluster_viz_method="pca",
        manifold_kind=manifold_kind,  # type: ignore[arg-type]
        umap_seed=seed,
        pca_seed=seed,
        parametric_umap_n_training_epochs=parametric_epochs,
    )
    result = fit_manifold_clustering(
        df,
        transform_config=tc,
        min_cluster_size=mcs,
        prediction_data=True,
    )
    labels = np.asarray(result.cluster_labels, dtype=np.int64)
    noise = float(np.mean(labels == -1))
    dbcv = result.fit_metrics.dbcv
    return (
        result.transformer,
        result.clusterer,
        result.umap_clustering,
        labels,
        float(dbcv) if dbcv is not None and dbcv == dbcv else None,
        noise,
    )


def _predict_hdbscan(
    clusterer: hdbscan.HDBSCAN, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lab, prob = approximate_predict(clusterer, X)
    return np.asarray(lab, dtype=np.int64), np.asarray(prob, dtype=np.float64).ravel()


def _emit_rate(labels: np.ndarray, scores: np.ndarray, thr: float) -> float:
    return float(np.mean((labels != -1) & (scores >= thr)))


def _pick_threshold(
    ref_labels: np.ndarray,
    ref_scores: np.ndarray,
    cand_labels: np.ndarray,
    cand_scores: np.ndarray,
) -> float:
    """Pick cand threshold so emit-rate matches legacy emit @ 0.3 on the tune set."""
    target = _emit_rate(ref_labels, ref_scores, 0.3)
    best_thr = 0.3
    best_err = abs(_emit_rate(cand_labels, cand_scores, best_thr) - target)
    for thr in np.linspace(0.05, 0.95, 37):
        err = abs(_emit_rate(cand_labels, cand_scores, float(thr)) - target)
        if err < best_err:
            best_err = err
            best_thr = float(thr)
    return best_thr


def _run_arm_head(
    name: str,
    X_all: np.ndarray,
    tune_idx: np.ndarray,
    hold_idx: np.ndarray,
    a_hold_labels: np.ndarray,
    a_hold_scores: np.ndarray,
    a_tune_labels: np.ndarray,
    a_tune_scores: np.ndarray,
    a_map: dict[int, str],
    head: EntityHead,
    head_map: dict[int, str],
    n_clusters: int,
    noise_fraction: float,
    dbcv: float | None,
    artifact_bytes: int,
) -> ArmResult:
    t0 = time.perf_counter()
    pred_all, score_all = head.predict(X_all)
    dt = time.perf_counter() - t0

    thr = _pick_threshold(
        a_tune_labels,
        a_tune_scores,
        pred_all[tune_idx],
        score_all[tune_idx],
    )
    hold_pred = pred_all[hold_idx]
    hold_scores = score_all[hold_idx]
    hold_eff = hold_pred.copy()
    hold_eff[hold_scores < thr] = -1

    mask = a_hold_labels != -1
    ari = (
        float(adjusted_rand_score(a_hold_labels[mask], hold_pred[mask]))
        if np.any(mask)
        else float("nan")
    )
    corr = float("nan")
    if np.any(mask) and np.std(a_hold_scores[mask]) > 1e-12:
        corr = float(np.corrcoef(a_hold_scores[mask], hold_scores[mask])[0, 1])

    return ArmResult(
        name=name,
        agreement_vs_a=_entity_agreement(hold_pred, head_map, a_hold_labels, a_map),
        ari_vs_a=ari,
        emit_rate=_emit_rate(hold_eff, hold_scores, thr),
        score_corr_vs_a=corr,
        artifact_bytes=artifact_bytes,
        predict_seconds=dt,
        n_clusters_emergent=n_clusters,
        noise_fraction=noise_fraction,
        dbcv=dbcv,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-parquet", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-dim", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--parametric-epochs", type=int, default=5)
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[256, 128, 128])
    args = parser.parse_args()

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    df = _load_embeds(args.embeddings_parquet, args.max_rows, args.seed)
    entities = df["entity"].astype(str).to_numpy()
    train_idx, tune_idx, hold_idx = _split_idx(len(df), args.seed)

    # Arm A: standard UMAP + HDBSCAN
    tx_a, cl_a, X_a, labels_a, dbcv_a, noise_a = _fit_manifold(
        df,
        manifold_kind="umap",
        pca=args.pca_components,
        umap_dim=args.umap_dim,
        mcs=args.min_cluster_size,
        seed=args.seed,
        parametric_epochs=args.parametric_epochs,
    )
    a_map = _majority_cluster_to_entity(labels_a[train_idx], entities[train_idx])
    _, a_all_sc = _predict_hdbscan(cl_a, X_a)
    a_all_lab = labels_a
    a_hold_lab, a_hold_sc = a_all_lab[hold_idx], a_all_sc[hold_idx]
    a_tune_lab, a_tune_sc = a_all_lab[tune_idx], a_all_sc[tune_idx]
    t0 = time.perf_counter()
    _predict_hdbscan(cl_a, X_a[hold_idx])
    a_pred_s = time.perf_counter() - t0
    a_emit = _emit_rate(a_hold_lab, a_hold_sc, 0.3)
    arm_a = ArmResult(
        name="A_legacy",
        agreement_vs_a=1.0,
        ari_vs_a=1.0,
        emit_rate=a_emit,
        score_corr_vs_a=1.0,
        artifact_bytes=_sizeof_joblib(cl_a) + _manifold_artifact_bytes(tx_a),
        predict_seconds=a_pred_s,
        n_clusters_emergent=int(len(set(int(x) for x in labels_a if x != -1))),
        noise_fraction=noise_a,
        dbcv=dbcv_a,
    )

    # Arm B/C/E: ParametricUMAP manifold
    tx_p, cl_p, X_p, labels_p, dbcv_p, noise_p = _fit_manifold(
        df,
        manifold_kind="parametric",
        pca=args.pca_components,
        umap_dim=args.umap_dim,
        mcs=args.min_cluster_size,
        seed=args.seed,
        parametric_epochs=args.parametric_epochs,
    )
    n_clusters_p = int(len(set(int(x) for x in labels_p if x != -1)))
    p_map = _majority_cluster_to_entity(labels_p[train_idx], entities[train_idx])
    manifold_p_bytes = _manifold_artifact_bytes(tx_p)

    # C: ParametricUMAP + HDBSCAN predict
    t0 = time.perf_counter()
    c_hold_lab, c_hold_sc = _predict_hdbscan(cl_p, X_p[hold_idx])
    c_pred_s = time.perf_counter() - t0
    c_tune_lab, c_tune_sc = _predict_hdbscan(cl_p, X_p[tune_idx])
    thr_c = _pick_threshold(a_tune_lab, a_tune_sc, c_tune_lab, c_tune_sc)
    c_eff = c_hold_lab.copy()
    c_eff[c_hold_sc < thr_c] = -1
    mask = a_hold_lab != -1
    arm_c = ArmResult(
        name="C_iso_manifold",
        agreement_vs_a=_entity_agreement(c_hold_lab, p_map, a_hold_lab, a_map),
        ari_vs_a=(
            float(adjusted_rand_score(a_hold_lab[mask], c_hold_lab[mask]))
            if np.any(mask)
            else float("nan")
        ),
        emit_rate=_emit_rate(c_eff, c_hold_sc, thr_c),
        score_corr_vs_a=(
            float(np.corrcoef(a_hold_sc[mask], c_hold_sc[mask])[0, 1])
            if np.any(mask) and np.std(a_hold_sc[mask]) > 1e-12
            else float("nan")
        ),
        artifact_bytes=_sizeof_joblib(cl_p) + manifold_p_bytes,
        predict_seconds=c_pred_s,
        n_clusters_emergent=n_clusters_p,
        noise_fraction=noise_p,
        dbcv=dbcv_p,
    )

    hidden = tuple(int(h) for h in args.mlp_hidden)
    mlp_b = fit_mlp_entity_head(
        X_p[train_idx],
        labels_p[train_idx],
        hidden_layer_sizes=hidden,
        random_state=args.seed,
    )
    arm_b = _run_arm_head(
        "B_compact",
        X_p,
        tune_idx,
        hold_idx,
        a_hold_lab,
        a_hold_sc,
        a_tune_lab,
        a_tune_sc,
        a_map,
        mlp_b,
        p_map,
        n_clusters_p,
        noise_p,
        dbcv_p,
        artifact_bytes=_sizeof_joblib(mlp_b) + manifold_p_bytes,
    )

    mlp_d = fit_mlp_entity_head(
        X_a[train_idx],
        labels_a[train_idx],
        hidden_layer_sizes=hidden,
        random_state=args.seed,
    )
    arm_d = _run_arm_head(
        "D_iso_head",
        X_a,
        tune_idx,
        hold_idx,
        a_hold_lab,
        a_hold_sc,
        a_tune_lab,
        a_tune_sc,
        a_map,
        mlp_d,
        a_map,
        arm_a.n_clusters_emergent,
        noise_a,
        dbcv_a,
        artifact_bytes=_sizeof_joblib(mlp_d) + _manifold_artifact_bytes(tx_a),
    )

    lin_e = fit_linear_svc_entity_head(
        X_p[train_idx], labels_p[train_idx], random_state=args.seed
    )
    arm_e = _run_arm_head(
        "E_linear_control",
        X_p,
        tune_idx,
        hold_idx,
        a_hold_lab,
        a_hold_sc,
        a_tune_lab,
        a_tune_sc,
        a_map,
        lin_e,
        p_map,
        n_clusters_p,
        noise_p,
        dbcv_p,
        artifact_bytes=_sizeof_joblib(lin_e) + manifold_p_bytes,
    )

    arms = [arm_a, arm_b, arm_c, arm_d, arm_e]
    rows = [asdict(a) for a in arms]
    pd.DataFrame(rows).to_csv(report_dir / "arms.csv", index=False)

    # Pass gates relative to A (on B) — entity-id agreement on holdout.
    agree_ok = arm_b.agreement_vs_a >= 0.95
    emit_rel = abs(arm_b.emit_rate - arm_a.emit_rate) / max(arm_a.emit_rate, 1e-9)
    emit_ok = emit_rel <= 0.10
    size_ok = arm_b.artifact_bytes <= 15_000_000 or (
        arm_a.artifact_bytes / max(arm_b.artifact_bytes, 1) >= 5.0
    )
    latency_ok = arm_b.predict_seconds <= 1.5 * max(arm_a.predict_seconds, 1e-9)
    e_only = arm_e.agreement_vs_a >= 0.95 and arm_b.agreement_vs_a < 0.95
    passed = bool(agree_ok and emit_ok and size_ok and latency_ok and not e_only)

    summary: dict[str, Any] = {
        "n_rows": len(df),
        "min_cluster_size": args.min_cluster_size,
        "gates": {
            "agreement_vs_a_ge_0.95": agree_ok,
            "emit_rate_within_10pct": emit_ok,
            "size_ok": size_ok,
            "latency_le_1_5x": latency_ok,
            "linear_not_sole_passer": not e_only,
            "passed": passed,
        },
        "arm_b": asdict(arm_b),
        "arm_a": asdict(arm_a),
        "arm_e": asdict(arm_e),
        "component_bytes": {
            "A_legacy_total": arm_a.artifact_bytes,
            "B_compact_total": arm_b.artifact_bytes,
            "B_mlp_only": _sizeof_joblib(mlp_b),
            "B_manifold": manifold_p_bytes,
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verdict = "PASS" if passed else "FAIL"
    print(f"compact_predict_study: {verdict}")
    print(json.dumps(summary["gates"], indent=2))
    print(f"wrote {report_dir / 'arms.csv'} and {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
