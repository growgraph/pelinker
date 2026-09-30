"""JSON serialization for reports: normalisation, readers and writers."""

from __future__ import annotations

import gzip
import json
import logging
import math
import pathlib
from typing import Any, cast

import numpy as np
import pandas as pd


from pelinker.reports.paths import (
    _FIT_CLUSTER_COMPOSITION_SCHEMA,
    _JSON_CLUSTERING_REPORT_SCHEMA,
    _KB_OUT_SCHEMA,
)
from pelinker.reports.schema import (
    AllScreenerCvResult,
    BinaryClassifierMetrics,
    ClusteringHyperparameters,
    LinkerFitDiagnostics,
    ModelSelectionReport,
    ModelSelectionRunReport,
)

logger = logging.getLogger(__name__)


def _binary_metrics_to_jsonable(
    m: BinaryClassifierMetrics,
) -> dict[str, dict[str, float]]:
    return {
        "precision": {"mean": m.precision.mean, "std": m.precision.std},
        "recall": {"mean": m.recall.mean, "std": m.recall.std},
        "f1": {"mean": m.f1.mean, "std": m.f1.std},
        "auc": {"mean": m.auc.mean, "std": m.auc.std},
    }


def _all_screener_cv_to_jsonable(result: AllScreenerCvResult) -> dict[str, object]:
    return {
        "screener_lda": _binary_metrics_to_jsonable(result.screener_lda),
        "screener_svm": _binary_metrics_to_jsonable(result.screener_svm),
        "screener_best_kind": result.screener_best_kind,
        "screener_best": _binary_metrics_to_jsonable(result.screener_best),
        "oov_winner_kind": result.oov_winner_kind,
        "oov": _binary_metrics_to_jsonable(result.oov),
        "combined": _binary_metrics_to_jsonable(result.combined),
    }


def _json_normalize(obj: object) -> object:
    """Map values to types accepted by :func:`json.dumps` (no NaN/Inf; no numpy scalars)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _json_normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_normalize(x) for x in obj]
    logger.warning(
        "json_normalize: coercing unexpected type %s to str", type(obj).__name__
    )
    return str(obj)


def _dataframe_to_jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(k): _json_normalize(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _ndarray_to_jsonable_nested(arr: np.ndarray) -> Any:
    return _json_normalize(np.asarray(arr).tolist())


def write_cluster_composition_json(
    path: str | pathlib.Path,
    composition_df: pd.DataFrame,
    *,
    top_n: int = 3,
    weighting: str = "inv_sqrt_mention_count",
    exclude_noise: bool = False,
    summary: dict[str, Any] | None = None,
    max_clusters_in_rows: int | None = None,
    indent: int = 2,
) -> None:
    """Serialize a processed cluster-composition table written at fit time."""
    from pelinker.clustering.composition import ENTITY_WEIGHTING_INV_SQRT

    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": _FIT_CLUSTER_COMPOSITION_SCHEMA,
        "top_n": int(top_n),
        "weighting": weighting or ENTITY_WEIGHTING_INV_SQRT,
        "exclude_noise": bool(exclude_noise),
        "rows": _dataframe_to_jsonable_records(composition_df),
    }
    if summary is not None:
        payload["summary"] = _json_normalize(summary)
    if max_clusters_in_rows is not None:
        payload["max_clusters_in_rows"] = int(max_clusters_in_rows)
    with gzip.open(p, mode="wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, indent=indent)


def read_cluster_composition_json(
    path: str | pathlib.Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a composition table and metadata written by :func:`write_cluster_composition_json`."""
    p = pathlib.Path(path).expanduser()
    with gzip.open(p, mode="rt", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    schema = str(raw.get("schema", ""))
    if schema not in (
        "pelinker.fit_cluster_composition.v1",
        _FIT_CLUSTER_COMPOSITION_SCHEMA,
    ):
        raise ValueError(f"Unsupported cluster composition schema: {schema!r}")
    meta = {k: v for k, v in raw.items() if k != "rows"}
    return pd.DataFrame(raw["rows"]), meta


def write_kb_out_json(
    path: str | pathlib.Path,
    payload: dict[str, Any],
    *,
    indent: int = 2,
) -> None:
    """Write :func:`~pelinker.kb.kb_out.build_kb_out_catalog` output."""
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if str(payload.get("schema", "")) != _KB_OUT_SCHEMA:
        raise ValueError(
            f"Expected schema {_KB_OUT_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_normalize(payload), f, indent=indent, ensure_ascii=False)


def read_kb_out_json(path: str | pathlib.Path) -> dict[str, Any]:
    """Load KB-out catalog JSON."""
    p = pathlib.Path(path).expanduser()
    with p.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    if str(raw.get("schema", "")) != _KB_OUT_SCHEMA:
        raise ValueError(f"Unsupported KB-out schema: {raw.get('schema')!r}")
    return raw


def write_model_selection_summary_json(
    path: str | pathlib.Path,
    payload: dict[str, Any],
    *,
    indent: int = 2,
) -> None:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    body = cast(dict[str, Any], _json_normalize(payload))
    with p.open("w", encoding="utf-8") as f:
        json.dump(body, f, indent=indent, ensure_ascii=False)


def _linker_fit_diagnostics_to_jsonable(d: LinkerFitDiagnostics) -> dict[str, Any]:
    return {
        "pca_residual": _ndarray_to_jsonable_nested(d.pca_residual),
        "pca_mahalanobis": _ndarray_to_jsonable_nested(d.pca_mahalanobis),
        "pca_spectral_entropy": _ndarray_to_jsonable_nested(d.pca_spectral_entropy),
        "oov_label": _ndarray_to_jsonable_nested(d.oov_label),
        "screener_decision": _ndarray_to_jsonable_nested(d.screener_decision),
        "projection_score": _ndarray_to_jsonable_nested(d.projection_score),
        "n_total": int(d.n_total),
        "sample_random_state": int(d.sample_random_state),
    }


def _linker_fit_diagnostics_from_jsonable(obj: object) -> LinkerFitDiagnostics | None:
    if obj is None or not isinstance(obj, dict):
        return None
    d = cast(dict[str, Any], obj)
    needed = (
        "pca_residual",
        "pca_mahalanobis",
        "pca_spectral_entropy",
        "oov_label",
        "screener_decision",
        "projection_score",
        "n_total",
        "sample_random_state",
    )
    if not all(k in d for k in needed):
        return None
    pr = np.asarray(d["pca_residual"], dtype=np.float64)
    pm = np.asarray(d["pca_mahalanobis"], dtype=np.float64)
    pe = np.asarray(d["pca_spectral_entropy"], dtype=np.float64)
    ol = np.asarray(d["oov_label"], dtype=np.int64).ravel()
    sd = np.asarray(d["screener_decision"], dtype=np.float64).ravel()
    mo = np.asarray(d["projection_score"], dtype=np.float64).ravel()
    n_tot = int(d["n_total"])
    srs = int(d["sample_random_state"])
    m = len(pr)
    if not (
        len(pm) == m and len(pe) == m and len(ol) == m and len(sd) == m and len(mo) == m
    ):
        return None
    return LinkerFitDiagnostics(
        pca_residual=pr,
        pca_mahalanobis=pm,
        pca_spectral_entropy=pe,
        oov_label=ol,
        screener_decision=sd,
        projection_score=mo,
        n_total=n_tot,
        sample_random_state=srs,
    )


def clustering_report_to_jsonable_dict(report: ModelSelectionReport) -> dict[str, Any]:
    """
    Flatten a :class:`ClusteringReport` into JSON-serializable built-ins (no DataFrames/ndarrays).

    Intended for ``json.dumps`` or for pickling a stable, language-adjacent blob. Schema version
    is stored under ``"schema"`` for forward compatibility.
    """
    ari_out: float | None
    if report.ari is None:
        ari_out = None
    else:
        ari_f = float(report.ari)
        ari_out = None if math.isnan(ari_f) or math.isinf(ari_f) else ari_f

    return {
        "schema": _JSON_CLUSTERING_REPORT_SCHEMA,
        "hyperparameters": {
            "min_cluster_size": int(report.hyperparameters.min_cluster_size),
        },
        "min_cluster_size_provenance": (
            None
            if report.min_cluster_size_provenance is None
            else report.min_cluster_size_provenance.to_jsonable()
        ),
        "distillation_fidelity": (
            None
            if report.distillation_fidelity is None
            else report.distillation_fidelity.to_jsonable()
        ),
        "n_rows_realized": (
            None if report.n_rows_realized is None else int(report.n_rows_realized)
        ),
        "best_score": _json_normalize(float(report.best_score)),
        "number_properties": int(report.number_properties),
        "n_clusters_emergent": int(report.n_clusters_emergent),
        "metrics_df": _dataframe_to_jsonable_records(report.metrics_df),
        "assignments": _dataframe_to_jsonable_records(report.assignments),
        "pca_residuals": _ndarray_to_jsonable_nested(report.pca_residuals),
        "pca_mahalanobis": _ndarray_to_jsonable_nested(report.pca_mahalanobis),
        "pca_spectral_entropy": _ndarray_to_jsonable_nested(
            report.pca_spectral_entropy
        ),
        "oov_label": _ndarray_to_jsonable_nested(report.oov_label),
        "umap_clustering": _ndarray_to_jsonable_nested(report.umap_clustering),
        "cluster_viz": _ndarray_to_jsonable_nested(report.cluster_viz),
        "cluster_viz_method": str(report.cluster_viz_method),
        "pca_reduced": _ndarray_to_jsonable_nested(report.pca_reduced),
        "ari": ari_out,
        "all_screener_cv": (
            None
            if report.all_screener_cv is None
            else _json_normalize(_all_screener_cv_to_jsonable(report.all_screener_cv))
        ),
        "training_diagnostics": (
            None
            if report.training_diagnostics is None
            else _json_normalize(
                _linker_fit_diagnostics_to_jsonable(report.training_diagnostics)
            )
        ),
    }


def write_clustering_report_json(
    path: str | pathlib.Path, report: ModelSelectionReport, *, indent: int = 2
) -> None:
    """
    Serialize ``report`` with :func:`clustering_report_to_jsonable_dict` to UTF-8 JSON.

    Parent directories are created when missing.
    """
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = clustering_report_to_jsonable_dict(report)

    with gzip.open(p, mode="wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, indent=indent)


def read_clustering_report_json(path: str | pathlib.Path) -> ModelSelectionReport:
    """
    Load a :class:`ModelSelectionReport` written by :func:`write_clustering_report_json`.

    Supports schema ``pelinker.clustering_report.v10`` (cluster-space viz coords).
    """
    p = pathlib.Path(path).expanduser()
    with gzip.open(p, mode="rt", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    schema = str(raw.get("schema", ""))
    if schema != "pelinker.clustering_report.v10":
        raise ValueError(f"Unsupported clustering report schema: {schema!r}")

    hp = raw["hyperparameters"]
    h = ClusteringHyperparameters(min_cluster_size=int(hp["min_cluster_size"]))
    metrics_df = pd.DataFrame(raw["metrics_df"])
    assignments = pd.DataFrame(raw["assignments"])

    def _farr(key: str) -> np.ndarray:
        return np.asarray(raw[key], dtype=np.float64)

    def _iarr(key: str) -> np.ndarray:
        return np.asarray(raw[key], dtype=np.int64)

    ari_raw = raw.get("ari")
    ari: float | None
    if ari_raw is None:
        ari = None
    else:
        ari = float(ari_raw)

    # Nested CV summaries are not round-tripped here (linker fit reports use ``None``).
    all_cv: AllScreenerCvResult | None = None

    td_raw = raw.get("training_diagnostics")
    training_diagnostics = _linker_fit_diagnostics_from_jsonable(td_raw)

    return ModelSelectionReport(
        hyperparameters=h,
        best_score=float(raw["best_score"]),
        number_properties=int(raw["number_properties"]),
        n_clusters_emergent=int(raw["n_clusters_emergent"]),
        metrics_df=metrics_df,
        assignments=assignments,
        pca_residuals=_farr("pca_residuals"),
        pca_mahalanobis=_farr("pca_mahalanobis"),
        pca_spectral_entropy=_farr("pca_spectral_entropy"),
        oov_label=_iarr("oov_label"),
        umap_clustering=np.asarray(raw["umap_clustering"], dtype=np.float64),
        cluster_viz=np.asarray(raw["cluster_viz"], dtype=np.float64),
        cluster_viz_method=str(raw["cluster_viz_method"]),
        pca_reduced=np.asarray(raw["pca_reduced"], dtype=np.float64),
        all_screener_cv=all_cv,
        screener_oos_datapoints=None,
        ari=ari,
        training_diagnostics=training_diagnostics,
    )


def model_selection_run_report_to_jsonable_dict(
    report: ModelSelectionRunReport,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _json_normalize(
            {
                "schema": report.schema,
                "generated_at": report.generated_at,
                "run_fingerprint": report.run_fingerprint,
                "run_config": report.run_config,
                "checkpoint": report.checkpoint,
                "combinations": report.combinations,
                "failures": report.failures,
                "best_overall": report.best_overall,
                "best_per_model": report.best_per_model,
            }
        ),
    )


def write_model_selection_run_report_json(
    path: str | pathlib.Path,
    report: ModelSelectionRunReport,
    *,
    indent: int = 2,
) -> None:
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = model_selection_run_report_to_jsonable_dict(report)
    with gzip.open(p, mode="wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(payload, f, indent=indent)
