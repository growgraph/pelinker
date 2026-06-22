from __future__ import annotations

import sys
import tempfile
from collections import defaultdict
from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass, replace
from typing import TypedDict, cast

import pandas as pd
import torch
import joblib
import numpy as np
import hdbscan
from hdbscan import approximate_predict
import pathlib
import logging

from pelinker.config import (
    ClusterCompositionSnapshot,
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingTrainingConfig,
    KBConfig,
    LinkerFitConfig,
    ManifoldOovScreenerConfig,
    NegativeScreenerConfig,
    TransformConfig,
)
from pelinker.screener.projection_screener import (
    ManifoldOovScoreModel,
    build_projection_training_arrays,
    evaluate_projection_cv,
    fit_projection_lda_no_cv,
    fit_projection_score_model,
)
from pelinker.screener.ambient_screener import NegativeClassScreener
from pelinker.analysis import (
    fit_ambient_screener_with_metrics,
    split_by_negative_label,
)
from pelinker.clustering_fit import fit_manifold_clustering
from pelinker.embedder import embed_kb_corpus
from pelinker.embedding_fusion import (
    MENTION_PROVENANCE_COLUMNS,
    fused_property_vectors_from_paths,
    property_fused_dataframe_for_linker_order,
)
from pelinker.sampling import draw_selection_sample, stratified_mention_sample
from pelinker.selection import load_selection_frame
from pelinker.transform import (
    EmbeddingTransformer,
    TransformArtifacts,
    score_transform_artifacts,
)
from pelinker.reporting import (
    ClusteringFitMetrics,
    ClusteringHyperparameters,
    LinkerFitDiagnostics,
    ModelSelectionReport,
    NegativeScreenerInSampleMetrics,
    entity_negative_label_mask_01,
    subsample_diagnostics_stratified,
)
from pelinker.linker_cluster_training import (
    cluster_composition_from_training_frame,
    cluster_derived_labels_map,
    consensus_cluster_names,
    provisional_cluster_assignments_from_training_frame as _provisional_cluster_assignments_from_training_frame,
)
from pelinker.linker_kb_lemma import (
    build_kb_lemma_index,
    enrich_entity_predictions_kb_validation,
    lookup_kb_training_entity_label,
)
from pelinker.onto import (
    MAX_LENGTH,
    MentionCandidate,
    NEGATIVE_LABEL,
    WordGrouping,
)
from pelinker.util import (
    extract_ordered_mention_tensors,
    keep_expression_for_prediction,
    load_models,
    texts_to_vrep,
)

logger = logging.getLogger(__name__)

_ROW_ID_COL = "_pelinker_row_id"

_LINKER_LOAD_DEFAULTS: dict[str, object] = {
    "embedding_metadata": None,
    "kb_config": None,
    "_hf_models_by_type": {},
    "training_cluster_frame": None,
    "training_pca_residuals": None,
    "training_pca_mahalanobis": None,
    "training_pca_spectral_entropy": None,
    "projection": None,
    "training_umap_clustering": None,
    "training_cluster_viz": None,
    "training_pca_reduced": None,
    "cluster_composition": None,
    "cluster_consensus_names": {},
    "cluster_derived_labels_map": {},
    "nlp_model_name": "en_core_web_trf",
    "_nlp": None,
    "screener_in_sample_metrics": None,
    "clustering_fit_metrics": None,
    "_fit_clustering_report": None,
}


@dataclass(frozen=True, slots=True)
class _NegativeScreenerFitStepResult:
    screener: NegativeClassScreener
    in_sample_metrics: NegativeScreenerInSampleMetrics | None
    decision: np.ndarray


@dataclass(frozen=True, slots=True)
class _ManifoldOovFitStepResult:
    model: ManifoldOovScoreModel | None
    cv_payload: dict[str, object] | None
    built_training: tuple[np.ndarray, np.ndarray, np.ndarray] | None


def _screener_training_frame(
    prepared: pd.DataFrame,
    fit_cfg: LinkerFitConfig,
    *,
    negative_label: str,
    clustering_prepared: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stratified subsample for screener fitting (aligned with model-selection draws)."""
    if clustering_prepared is not None and len(clustering_prepared) < len(prepared):
        return clustering_prepared
    cap = fit_cfg.screener_max_rows
    if cap is None or len(prepared) <= cap:
        return prepared
    return stratified_mention_sample(
        prepared,
        n_target=cap,
        negative_label=negative_label,
        random_state=fit_cfg.screener_seed,
    )


def _fit_ambient_screener_step(
    prepared_full: pd.DataFrame,
    screener_fit_frame: pd.DataFrame,
    ns_cfg: NegativeScreenerConfig,
) -> _NegativeScreenerFitStepResult:
    screener, metrics = fit_ambient_screener_with_metrics(screener_fit_frame, ns_cfg)
    x_emb = np.stack(prepared_full["embed"].values).astype(np.float32, copy=False)
    decision = np.asarray(screener.decision_function(x_emb), dtype=np.float64).ravel()
    return _NegativeScreenerFitStepResult(
        screener=screener,
        in_sample_metrics=metrics,
        decision=decision,
    )


def _fit_projection_step(
    prepared: pd.DataFrame,
    manifold_df: pd.DataFrame,
    transformer: EmbeddingTransformer,
    ns_cfg: NegativeScreenerConfig,
    mo_cfg: ManifoldOovScreenerConfig,
) -> _ManifoldOovFitStepResult:
    if not mo_cfg.enabled:
        return _ManifoldOovFitStepResult(
            model=None, cv_payload=None, built_training=None
        )
    built = build_projection_training_arrays(
        prepared,
        manifold_df,
        transformer,
        negative_label=ns_cfg.negative_label,
    )
    if built is None:
        return _ManifoldOovFitStepResult(
            model=None, cv_payload=None, built_training=None
        )
    x_mo, y_mo, _pos = built
    n0 = int(np.sum(y_mo == 0))
    n1 = int(np.sum(y_mo == 1))
    if n0 < 2 or n1 < 2:
        model, cv_pl = fit_projection_lda_no_cv(x_mo, y_mo)
        return _ManifoldOovFitStepResult(
            model=model, cv_payload=cv_pl, built_training=built
        )
    cv_eval = evaluate_projection_cv(x_mo, y_mo, mo_cfg)
    if cv_eval is None:
        return _ManifoldOovFitStepResult(
            model=None, cv_payload=None, built_training=built
        )
    model, cv_pl = fit_projection_score_model(
        x_mo,
        y_mo,
        mo_cfg,
        cv_payload_and_winner=cv_eval,
    )
    return _ManifoldOovFitStepResult(
        model=model, cv_payload=cv_pl, built_training=built
    )


def _linker_fit_diagnostics_full(
    *,
    prepared: pd.DataFrame,
    negative_label: str,
    screener_decision: np.ndarray,
    transformer: EmbeddingTransformer,
    built_mo: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    mo_model: ManifoldOovScoreModel | None,
    sample_random_state: int,
) -> LinkerFitDiagnostics:
    n = len(prepared)
    oov_label = (prepared["entity"].astype(str).values == negative_label).astype(
        np.int64
    )
    full_pca = np.full((n, 3), np.nan, dtype=np.float64)
    full_mo = np.full(n, np.nan, dtype=np.float64)

    if built_mo is not None:
        x_mo, _y_mo, pos = built_mo
        x_mo_f = np.asarray(x_mo, dtype=np.float64)
        for i in range(x_mo_f.shape[0]):
            full_pca[int(pos[i]), :] = x_mo_f[i, :]
        if mo_model is not None:
            scores = mo_model.score(x_mo_f)
            scores_1d = np.asarray(scores, dtype=np.float64).ravel()
            for i in range(len(pos)):
                full_mo[int(pos[i])] = float(scores_1d[i])
    else:
        emb = np.stack(prepared["embed"].values).astype(np.float32, copy=False)
        _u0, _u1, res, mah, ent = transformer.transform(emb)
        full_pca[:, 0] = np.asarray(res, dtype=np.float64).ravel()
        full_pca[:, 1] = np.asarray(mah, dtype=np.float64).ravel()
        full_pca[:, 2] = np.asarray(ent, dtype=np.float64).ravel()

    return LinkerFitDiagnostics(
        pca_residual=full_pca[:, 0].copy(),
        pca_mahalanobis=full_pca[:, 1].copy(),
        pca_spectral_entropy=full_pca[:, 2].copy(),
        oov_label=oov_label.copy(),
        screener_decision=np.asarray(screener_decision, dtype=np.float64)
        .ravel()
        .copy(),
        projection_score=full_mo.copy(),
        n_total=n,
        sample_random_state=sample_random_state,
    )


def _predict_cluster_labels_on_full_manifold(
    manifold_full: pd.DataFrame,
    manifold_fit: pd.DataFrame,
    clusterer: hdbscan.HDBSCAN,
    umap_full: np.ndarray,
    fit_labels: np.ndarray,
    *,
    row_id_col: str = _ROW_ID_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """``approximate_predict`` on full KB rows; training subsample rows keep exact fit labels."""
    if (
        row_id_col not in manifold_fit.columns
        or row_id_col not in manifold_full.columns
    ):
        raise ValueError(f"manifold frames missing {row_id_col!r}")
    cluster_labels_arr, cluster_probs = approximate_predict(clusterer, umap_full)
    cluster_labels = np.asarray(cluster_labels_arr, dtype=np.int64)
    cluster_scores = np.asarray(cluster_probs, dtype=np.float64).ravel()
    fit_ids = manifold_fit[row_id_col].to_numpy(dtype=np.int64, copy=False)
    fit_id_to_label = dict(zip(fit_ids.tolist(), fit_labels.astype(int).tolist()))
    full_ids = manifold_full[row_id_col].to_numpy(dtype=np.int64, copy=False)
    for i, row_id in enumerate(full_ids.tolist()):
        label = fit_id_to_label.get(int(row_id))
        if label is not None:
            cluster_labels[i] = label
    return cluster_labels, cluster_scores


def _build_training_cluster_frame(
    manifold_full: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_scores: np.ndarray,
    screener_decision: np.ndarray,
    manifold_mask: np.ndarray,
    projection_scores: np.ndarray,
) -> pd.DataFrame:
    tc_cols = ["pmid", "entity", "mention"]
    missing = [c for c in tc_cols if c not in manifold_full.columns]
    if missing:
        raise ValueError(
            "Prepared mention frame missing columns required for "
            f"training_cluster_frame: {missing}"
        )
    frame = manifold_full[tc_cols].copy()
    for col in MENTION_PROVENANCE_COLUMNS:
        if col in manifold_full.columns:
            frame[col] = manifold_full[col].values
    frame["cluster"] = cluster_labels
    frame["screener_score"] = screener_decision[manifold_mask]
    frame["projection_score"] = projection_scores[manifold_mask]
    frame["cluster_score"] = cluster_scores
    return frame


def _store_training_manifold_arrays(
    linker: Linker,
    artifacts: TransformArtifacts,
) -> None:
    linker.training_pca_residuals = np.asarray(
        artifacts.pca_residuals, dtype=np.float32
    )
    linker.training_pca_mahalanobis = np.asarray(
        artifacts.pca_mahalanobis, dtype=np.float32
    )
    linker.training_pca_spectral_entropy = np.asarray(
        artifacts.pca_spectral_entropy, dtype=np.float32
    )
    linker.training_umap_clustering = np.asarray(
        artifacts.umap_clustering, dtype=np.float32
    )
    linker.training_cluster_viz = np.asarray(artifacts.cluster_viz, dtype=np.float32)
    linker.training_pca_reduced = np.asarray(artifacts.pca_reduced, dtype=np.float32)


def _finalize_linker_cluster_state(
    linker: Linker,
    *,
    kb_config: KBConfig | None,
    sampled_diag: LinkerFitDiagnostics,
) -> None:
    linker.cluster_composition = cluster_composition_from_training_frame(
        linker.training_cluster_frame
    )
    linker.cluster_consensus_names = consensus_cluster_names(linker.cluster_composition)

    linker.cluster_assignments = _provisional_cluster_assignments_from_training_frame(
        linker.labels_map,
        linker.training_cluster_frame,
    )
    linker.cluster_derived_labels_map = cluster_derived_labels_map(
        linker.labels_map,
        linker.cluster_assignments,
        linker.cluster_composition,
    )
    linker.vocabulary = sorted(linker.cluster_assignments.keys())
    if not linker.vocabulary:
        raise ValueError(
            "No entity_ids received provisional cluster assignments after fit "
            "(check labels_map and training entity labels)"
        )

    if kb_config is not None:
        if kb_config.entity_count is None:
            linker.kb_config = replace(kb_config, entity_count=len(linker.vocabulary))
        else:
            linker.kb_config = kb_config

    fit_report = linker.build_clustering_report(
        training_diagnostics=sampled_diag,
    )
    linker._strip_training_metrics_for_prediction()
    linker._fit_clustering_report = fit_report


class EntityPredictionRow(TypedDict):
    """Row shape for ``predict`` ``entities`` entries from clustering (before optional KB fields)."""

    mention: str
    a: int | None
    b: int | None
    a_abs: int | None
    b_abs: int | None
    itext: int | None
    ichunk: int | None
    word_grouping: WordGrouping | None
    lemma: str
    entity_id_predicted: str
    score: float
    pca_residual: float
    pca_mahalanobis: float
    pca_spectral_entropy: float
    anomaly_score_max_z: float
    projection_score: float


@dataclass(frozen=True, slots=True)
class LinkerPredictResult:
    """Structured return value from :meth:`Linker.predict`.

    ``debug_mentions`` is one row per extracted mention (including screener negatives)
    when debug was requested; it is not filtered by cluster score.
    """

    entities: list[dict[str, object]]
    debug_mentions: list[dict[str, object]] | None = None

    def filter_by_score(self, thr_score: float) -> LinkerPredictResult:
        filtered: list[dict[str, object]] = [
            r for r in self.entities if float(r.get("score", 0.0)) >= thr_score
        ]
        return LinkerPredictResult(
            entities=filtered,
            debug_mentions=self.debug_mentions,
        )

    def to_dict(
        self,
        *,
        include_debug: bool = False,
        include_entity_anomaly_metrics: bool = True,
        strip_mention_source_index: bool = True,
        public_entity_fields: bool = False,
    ) -> dict[str, object]:
        """Serialize for JSON APIs. Debug rows use the legacy key ``mention_anomaly``.

        When ``public_entity_fields`` is true (used by ``/link`` and the link-files CLI
        in default mode), entity rows omit anomaly metrics, KB validation labels, and
        ``word_grouping``; character spans use document-global ``a`` / ``b`` (from
        internal ``a_abs`` / ``b_abs``), not chunk-local coordinates.
        """
        entities_out: list[dict[str, object]] = []
        for r in self.entities:
            e = dict(r)
            if strip_mention_source_index:
                e.pop("mention_source_index", None)
            if public_entity_fields:
                e.pop("pca_residual", None)
                e.pop("pca_mahalanobis", None)
                e.pop("pca_spectral_entropy", None)
                e.pop("anomaly_score_max_z", None)
                e.pop("projection_score", None)
                e.pop("word_grouping", None)
                for k in (
                    "kb_training_entity",
                    "kb_training_entity_from_lemma",
                    "kb_training_entity_for_prediction",
                    "lemma_kb_matches_predicted_entity",
                ):
                    e.pop(k, None)
                chunk_a = e.pop("a", None)
                chunk_b = e.pop("b", None)
                abs_a = e.pop("a_abs", None)
                abs_b = e.pop("b_abs", None)
                e["a"] = abs_a if abs_a is not None else chunk_a
                e["b"] = abs_b if abs_b is not None else chunk_b
            elif not include_entity_anomaly_metrics:
                e.pop("pca_residual", None)
                e.pop("pca_mahalanobis", None)
                e.pop("pca_spectral_entropy", None)
                e.pop("anomaly_score_max_z", None)
                e.pop("projection_score", None)
            entities_out.append(e)
        payload: dict[str, object] = {"entities": entities_out}
        if include_debug and self.debug_mentions is not None:
            payload["mention_anomaly"] = [dict(row) for row in self.debug_mentions]
        return payload


def _linker_artifact_gz_path(file_spec: str | pathlib.Path) -> pathlib.Path:
    """Path to the on-disk ``.gz`` artifact (accepts base path or path already ending in ``.gz``)."""
    p = pathlib.Path(file_spec).expanduser()
    if p.suffix == ".gz":
        return p
    return p.with_name(p.name + ".gz")


class Linker:
    def __init__(
        self,
        transformer: EmbeddingTransformer | None = None,
        clusterer: hdbscan.HDBSCAN | None = None,
        transform_config: TransformConfig | None = None,
        embedding_metadata: EmbeddingModelMetadata | None = None,
        kb_config: KBConfig | None = None,
        **kwargs,
    ):
        self.transformer: EmbeddingTransformer | None = transformer
        self.clusterer: hdbscan.HDBSCAN | None = clusterer
        self.cluster_assignments: dict[str, int] = {}
        self.transform_config: TransformConfig | None = transform_config
        self.embedding_metadata: EmbeddingModelMetadata | None = embedding_metadata
        self.kb_config: KBConfig | None = kb_config

        self.vocabulary: list[str] = []
        self.labels_map: dict[str, str] = kwargs.pop("labels_map", dict())
        self.training_cluster_frame: pd.DataFrame | None = None
        self.training_pca_residuals: np.ndarray | None = None
        self.training_pca_mahalanobis: np.ndarray | None = None
        self.training_pca_spectral_entropy: np.ndarray | None = None
        self.training_umap_clustering: np.ndarray | None = None
        self.training_cluster_viz: np.ndarray | None = None
        self.training_pca_reduced: np.ndarray | None = None
        self.cluster_composition: ClusterCompositionSnapshot | None = None
        self.cluster_consensus_names: dict[int, str] = {}
        self.cluster_derived_labels_map: dict[str, str] = {}
        self.screener: NegativeClassScreener | None = None
        self.screener_in_sample_metrics: NegativeScreenerInSampleMetrics | None = None
        self.clustering_fit_metrics: ClusteringFitMetrics | None = None
        self.projection: ManifoldOovScoreModel | None = kwargs.pop("projection", None)
        self._projection_cv_payload: dict[str, object] | None = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_models_by_type: dict[str, tuple[object, object]] = {}
        self.nlp_model_name: str = kwargs.pop("nlp_model_name", "en_core_web_trf")
        self._nlp: object | None = None
        self._fit_clustering_report: ModelSelectionReport | None = None

    @staticmethod
    def filter_entities(
        entities: list[dict[str, object]], thr_score: float
    ) -> list[dict[str, object]]:
        return [r for r in entities if float(r.get("score", 0.0)) >= thr_score]

    @classmethod
    def filter_report(
        cls, report: dict[str, object], thr_score: float
    ) -> dict[str, object]:
        """Return a shallow copy with ``entities`` filtered by score (does not mutate ``report``)."""
        raw_entities = report.get("entities", [])
        entities = raw_entities if isinstance(raw_entities, list) else []
        filtered = cls.filter_entities(
            cast(list[dict[str, object]], entities), thr_score
        )
        return {**report, "entities": filtered}

    @staticmethod
    def _merge_prediction_fields_into_debug_mentions(
        debug_rows: list[dict[str, object]],
        predictions: list[dict[str, object]],
        *,
        include_kb_validation_fields: bool,
    ) -> None:
        keys_always: tuple[str, ...] = (
            "entity_id_predicted",
            "score",
            "kb_training_entity",
            "pca_spectral_entropy",
            "projection_score",
        )
        keys_when_validation: tuple[str, ...] = (
            "kb_training_entity_from_lemma",
            "kb_training_entity_for_prediction",
            "lemma_kb_matches_predicted_entity",
        )
        for row in predictions:
            mi = row.get("mention_source_index")
            if not isinstance(mi, int) or mi < 0 or mi >= len(debug_rows):
                continue
            target = debug_rows[mi]
            for k in keys_always:
                if k in row:
                    target[k] = row[k]
            if include_kb_validation_fields:
                for k in keys_when_validation:
                    if k in row:
                        target[k] = row[k]

    def dump(self, file_spec: str | pathlib.Path) -> None:
        self._fit_clustering_report = None
        path = _linker_artifact_gz_path(file_spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, file_spec: str | pathlib.Path) -> Linker:
        path = _linker_artifact_gz_path(file_spec)
        pe_model = joblib.load(path)
        for field, default in _LINKER_LOAD_DEFAULTS.items():
            if field not in pe_model.__dict__:
                setattr(pe_model, field, default)
        return pe_model

    def take_fit_clustering_report(self) -> ModelSelectionReport | None:
        """
        Consume the :class:`~pelinker.reporting.ClusteringReport` produced by the last :meth:`fit`.

        Call **before** :meth:`dump` if you need JSON or other persistence: the report is
        not serialized on the linker artifact (only prediction state is pickled).

        Returns ``None`` if :meth:`fit` has not been run, the report was already taken, or
        clustering state was incomplete.
        """
        report = self._fit_clustering_report
        self._fit_clustering_report = None
        return report

    def _strip_training_metrics_for_prediction(self) -> None:
        """Drop mention-level training tables and manifold arrays; keep predict-time fields."""
        self.training_cluster_frame = None
        self.training_pca_residuals = None
        self.training_pca_mahalanobis = None
        self.training_pca_spectral_entropy = None
        self.training_umap_clustering = None
        self.training_cluster_viz = None
        self.training_pca_reduced = None
        self._projection_cv_payload = None

    def build_clustering_report(
        self,
        *,
        training_diagnostics: LinkerFitDiagnostics | None = None,
    ) -> ModelSelectionReport | None:
        """
        Build a :class:`~pelinker.reporting.ClusteringReport` when full training rows exist.

        After a normal :meth:`fit`, heavy training payloads are removed for prediction; use
        :meth:`take_fit_clustering_report` immediately after fitting instead.

        This method remains useful for **legacy** pickled linkers that still embed training
        arrays, or for tests that skip stripping.

        Args:
            training_diagnostics: Optional stratified-sampled mention-level diagnostics
                (PCA quality + screener / manifold OOV scores) attached only to the fit report.
        """
        tcf = self.training_cluster_frame
        if (
            tcf is None
            or self.training_pca_residuals is None
            or self.training_pca_mahalanobis is None
            or self.training_pca_spectral_entropy is None
            or self.training_umap_clustering is None
            or self.training_cluster_viz is None
            or self.training_pca_reduced is None
            or self.clustering_fit_metrics is None
        ):
            return None
        n = len(tcf)
        if (
            len(self.training_pca_residuals) != n
            or len(self.training_pca_mahalanobis) != n
            or len(self.training_pca_spectral_entropy) != n
            or self.training_umap_clustering.shape[0] != n
            or self.training_cluster_viz.shape[0] != n
            or self.training_pca_reduced.shape[0] != n
        ):
            return None

        m = self.clustering_fit_metrics
        dbcv_f = float(m.dbcv) if m.dbcv is not None else float("nan")
        ari_val = float(m.ari) if m.ari is not None else float("nan")
        metrics_df = pd.DataFrame(
            [
                {
                    "min_cluster_size": m.min_cluster_size,
                    "icm": float("nan"),
                    "n_clusters": m.n_clusters_emergent,
                    "dbcv": dbcv_f,
                    "ari": ari_val,
                }
            ]
        )

        base_cols = ["entity", "cluster", "pmid", "mention"]
        optional_cols = [
            *MENTION_PROVENANCE_COLUMNS,
            "screener_score",
            "projection_score",
            "cluster_score",
        ]
        keep = [c for c in base_cols + optional_cols if c in tcf.columns]
        assignments = tcf[keep].copy()

        number_properties = int(tcf["entity"].nunique())

        res_f = np.asarray(self.training_pca_residuals, dtype=np.float64)
        mah_f = np.asarray(self.training_pca_mahalanobis, dtype=np.float64)
        ent_f = np.asarray(self.training_pca_spectral_entropy, dtype=np.float64)

        neg_lbl = (
            self.screener.negative_label
            if self.screener is not None
            else NEGATIVE_LABEL
        )
        y_neg = entity_negative_label_mask_01(tcf["entity"], neg_lbl)

        return ModelSelectionReport(
            hyperparameters=ClusteringHyperparameters(
                min_cluster_size=m.min_cluster_size
            ),
            best_score=dbcv_f,
            number_properties=number_properties,
            n_clusters_emergent=m.n_clusters_emergent,
            metrics_df=metrics_df,
            assignments=assignments,
            pca_residuals=res_f,
            pca_mahalanobis=mah_f,
            pca_spectral_entropy=ent_f,
            oov_label=y_neg,
            umap_clustering=np.asarray(self.training_umap_clustering, dtype=np.float64),
            cluster_viz=np.asarray(self.training_cluster_viz, dtype=np.float64),
            cluster_viz_method=(
                self.transform_config.cluster_viz_method
                if self.transform_config is not None
                else "pca"
            ),
            pca_reduced=np.asarray(self.training_pca_reduced, dtype=np.float64),
            all_screener_cv=None,
            screener_oos_datapoints=None,
            ari=m.ari,
            training_diagnostics=training_diagnostics,
        )

    @staticmethod
    def _normalize_embedding_paths(
        embeddings: pathlib.Path | Sequence[pathlib.Path],
    ) -> list[pathlib.Path]:
        if isinstance(embeddings, pathlib.Path):
            return [embeddings.expanduser()]
        return [pathlib.Path(p).expanduser() for p in embeddings]

    def fit(
        self,
        embeddings: pathlib.Path | Sequence[pathlib.Path] | None,
        transform_config: TransformConfig,
        min_cluster_size: int,
        *,
        fit_config: LinkerFitConfig | None = None,
        embedding_training: EmbeddingTrainingConfig | None = None,
        embedding_metadata: EmbeddingModelMetadata | None = None,
        kb_config: KBConfig | None = None,
    ) -> Linker:
        """
        Fit the Linker model with embeddings.

        This method handles two main parts:
        a) Loading and processing embeddings (from file or direct array)
        b) Fitting the negative screener, then PCA/UMAP + HDBSCAN on non-negative rows

        Args:
            embeddings: Path or sequence of paths to parquet file(s) (mention-level rows:
                        ``pmid``, ``entity``, ``mention``, ``embed``). Multiple files are
                        fused like :func:`~pelinker.selection.load_selection_frame` (inner join
                        on keys, concat embeddings). Order must match
                        ``embedding_metadata.sources``. If None, ``embed_kb_corpus`` is run
                        (one output file per source).
            transform_config: TransformConfig instance
            min_cluster_size: HDBSCAN ``min_cluster_size`` (choose upstream, e.g. via
                ``pelinker.model_selection``).
            fit_config: Parquet read batching, mention load filters, subsample settings, and screener config.
                Defaults to :class:`LinkerFitConfig()`.
            embedding_training: Corpus paths and embedding runtime. Required when embeddings=None.
            embedding_metadata: If provided, overrides or sets ``self.embedding_metadata`` for
                this fit (required when embeddings=None unless already set on the linker).
            kb_config: Knowledge-base metadata stored on the linker; ``entity_count`` is set
                from fitted vocabulary when omitted (None).

        Side effects:
            Sets ``cluster_composition`` (mention-weighted property mass and per-cluster
            mixtures), ``cluster_consensus_names`` (short labels from those mixtures),
            ``screener_in_sample_metrics``, and ``clustering_fit_metrics``. Mention-level
            training tables and manifold arrays used for :class:`~pelinker.reporting.ClusteringReport`
            are stripped after each fit; persist JSON with :meth:`take_fit_clustering_report` and
            :func:`~pelinker.reporting.write_clustering_report_json` at
            :func:`~pelinker.reporting.linker_fit_clustering_report_path` (same layout as
            ``pelinker-fit`` ``report_path``) before :meth:`dump`.

        Returns:
            self
        """
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be >= 2")

        is_temporary = False
        embeddings_paths: list[pathlib.Path] = []

        try:
            if embedding_metadata is not None:
                self.embedding_metadata = embedding_metadata

            fc = fit_config or LinkerFitConfig()
            load_cfg = fc.to_clustering_sample_config()

            if embeddings is None:
                if embedding_training is None:
                    raise ValueError(
                        "embedding_training is required when embeddings is None. "
                        "Provide embeddings path or EmbeddingTrainingConfig(...)."
                    )
                self.nlp_model_name = embedding_training.nlp_model
                if self.embedding_metadata is None:
                    raise ValueError(
                        "embedding_metadata is required when embeddings is None "
                        "(set on Linker(...) or pass embedding_metadata=... to fit())."
                    )
                k = len(self.embedding_metadata.sources)
                for _ in range(k):
                    tf = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
                    embeddings_paths.append(pathlib.Path(tf.name))
                    tf.close()

                logger.info("Stage (a): Embedding corpus (%s source(s))...", k)
                embed_kb_corpus(
                    metadata=self.embedding_metadata,
                    training=embedding_training,
                    output_parquet_paths=embeddings_paths,
                )
                is_temporary = True
            else:
                embeddings_paths = self._normalize_embedding_paths(embeddings)
                if self.embedding_metadata is not None and len(embeddings_paths) != len(
                    self.embedding_metadata.sources
                ):
                    raise ValueError(
                        "Number of embedding parquet paths must match "
                        f"embedding_metadata.sources ({len(self.embedding_metadata.sources)}), "
                        f"got {len(embeddings_paths)}"
                    )
                logger.info(
                    "Stage (A): Using provided embeddings (%s file(s)): %s",
                    len(embeddings_paths),
                    embeddings_paths,
                )

            logger.info(
                "Stage (B): mention-level load from %s parquet file(s)",
                len(embeddings_paths),
            )
            prepared = load_selection_frame(
                file_paths=embeddings_paths,
                config=load_cfg,
                show_embedding_read_progress=True,
            )
            if prepared is None or len(prepared) == 0:
                raise ValueError(
                    "No mention-level embedding rows loaded from parquet (check paths "
                    "and columns pmid, entity, mention, embed)."
                )

            self._fit_clustering_on_prepared_mentions(
                prepared=prepared,
                transform_config=transform_config,
                fit_cfg=fc,
                min_cluster_size=min_cluster_size,
                kb_config=kb_config,
            )

            return self
        finally:
            if is_temporary:
                for p in embeddings_paths:
                    try:
                        p.unlink()
                        logger.debug("Removed temporary parquet file: %s", p)
                    except Exception as e:
                        logger.warning(
                            "Failed to remove temporary parquet file %s: %s", p, e
                        )

    def _fit_clustering_on_prepared_mentions(
        self,
        *,
        prepared: pd.DataFrame,
        transform_config: TransformConfig,
        fit_cfg: LinkerFitConfig,
        min_cluster_size: int,
        kb_config: KBConfig | None,
    ) -> None:
        """Fit screeners, then PCA/UMAP + HDBSCAN on the clustering subsample; label full KB via predict."""
        prepared = prepared.copy()
        prepared[_ROW_ID_COL] = np.arange(len(prepared), dtype=np.int64)

        ns_cfg = fit_cfg.ambient_screener
        mo_cfg = fit_cfg.projection_screener
        neg_label = ns_cfg.negative_label

        sample_cfg = fit_cfg.to_clustering_sample_config()
        clustering_prepared = draw_selection_sample(
            prepared,
            sample_cfg,
            sample_index=fit_cfg.clustering_sample_index,
        )

        screener_prepared = _screener_training_frame(
            prepared,
            fit_cfg,
            negative_label=neg_label,
            clustering_prepared=clustering_prepared,
        )
        _, manifold_screener = split_by_negative_label(screener_prepared, neg_label)

        neg_step = _fit_ambient_screener_step(prepared, screener_prepared, ns_cfg)
        self.screener = neg_step.screener
        self.screener_in_sample_metrics = neg_step.in_sample_metrics

        _, manifold_fit = split_by_negative_label(clustering_prepared, neg_label)
        if len(manifold_fit) == 0:
            raise ValueError(
                "No rows left after excluding negative-label mentions for manifold fit"
            )

        _, manifold_full = split_by_negative_label(prepared, neg_label)
        if len(manifold_full) == 0:
            raise ValueError(
                "No rows left after excluding negative-label mentions for manifold fit"
            )

        self.transform_config = transform_config
        cl_result = fit_manifold_clustering(
            manifold_fit,
            transform_config=transform_config,
            min_cluster_size=min_cluster_size,
            prediction_data=True,
        )
        self.transformer = cl_result.transformer
        self.clusterer = cl_result.clusterer
        self.clustering_fit_metrics = cl_result.fit_metrics

        full_artifacts = score_transform_artifacts(
            manifold_full,
            cl_result.transformer,
            include_umap=True,
        )
        _store_training_manifold_arrays(self, full_artifacts)

        mo_step = _fit_projection_step(
            screener_prepared,
            manifold_screener,
            self.transformer,
            ns_cfg,
            mo_cfg,
        )
        self.projection = mo_step.model
        self._projection_cv_payload = mo_step.cv_payload

        built_mo_diag: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if mo_cfg.enabled:
            built_mo_diag = build_projection_training_arrays(
                prepared,
                manifold_full,
                self.transformer,
                negative_label=neg_label,
            )

        full_diag = _linker_fit_diagnostics_full(
            prepared=prepared,
            negative_label=neg_label,
            screener_decision=neg_step.decision,
            transformer=self.transformer,
            built_mo=built_mo_diag,
            mo_model=self.projection,
            sample_random_state=fit_cfg.diagnostics_random_state,
        )
        sampled_diag = subsample_diagnostics_stratified(
            full_diag,
            max_rows=fit_cfg.diagnostics_sample_size,
            random_state=fit_cfg.diagnostics_random_state,
        )

        neg_mask = prepared["entity"].astype(str).values == neg_label
        manifold_mask = ~neg_mask
        cluster_labels, cluster_scores = _predict_cluster_labels_on_full_manifold(
            manifold_full,
            manifold_fit,
            cl_result.clusterer,
            full_artifacts.umap_clustering,
            cl_result.cluster_labels,
        )
        self.training_cluster_frame = _build_training_cluster_frame(
            manifold_full,
            cluster_labels,
            cluster_scores,
            neg_step.decision,
            manifold_mask,
            full_diag.projection_score,
        )

        _finalize_linker_cluster_state(
            self,
            kb_config=kb_config,
            sampled_diag=sampled_diag,
        )

    def _load_embeddings_from_file(
        self, embeddings_path: pathlib.Path, kb_labels: set[str] | None = None
    ) -> tuple[np.ndarray, list[str]]:
        """Backward-compatible single-file loader; delegates to fused multi-file path."""
        return self._load_fused_embeddings_from_files([embeddings_path], kb_labels)

    def _load_fused_embeddings_from_files(
        self,
        embeddings_paths: Sequence[pathlib.Path],
        kb_labels: set[str] | None = None,
        *,
        read_config: ClusteringOptimizationConfig | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Per-file per-property mean embeddings, intersection across sources, concat features.

        Legacy helper for property-level loads; ``Linker.fit`` uses mention-level fusion instead.
        """
        cfg = read_config or ClusteringOptimizationConfig()
        logger.info(
            "Reading %s parquet source(s) and fusing per-property vectors...",
            len(embeddings_paths),
        )
        fused = fused_property_vectors_from_paths(
            embeddings_paths,
            kb_labels,
            batch_size=cfg.batch_size,
            show_read_progress=sys.stdout.isatty(),
        )
        if not fused:
            raise ValueError("No fused property vectors (empty intersection or inputs)")

        dfr = property_fused_dataframe_for_linker_order(fused, self.labels_map)
        if len(dfr) == 0:
            raise ValueError("No valid embeddings after mapping to entity_ids")

        embeddings = np.stack([np.asarray(e, dtype=np.float64) for e in dfr["embed"]])
        entity_ids = list(dfr["entity_id"])
        logger.info(
            "Embedded %s KB properties into %s-dimensional fused vectors",
            len(embeddings),
            embeddings.shape[1],
        )
        return embeddings, entity_ids

    def _ensure_hf_models_for_sources(self, *, use_gpu: bool = False) -> None:
        """Load tokenizer+encoder once per distinct ``model_type`` in metadata sources."""
        if self.embedding_metadata is None:
            raise ValueError(
                "embedding_metadata is required for predict(); set it during fit() or on the Linker."
            )
        for src in self.embedding_metadata.sources:
            mt = src.model_type
            if mt not in self._hf_models_by_type:
                logger.info("Loading encoder for predict: %s", mt)
                self._hf_models_by_type[mt] = load_models(mt, sentence=False)
        if use_gpu and torch.cuda.is_available():
            for _mt, (_tok, model) in self._hf_models_by_type.items():
                model.to("cuda")
        elif use_gpu:
            logger.warning("CUDA is not available; predict runs on CPU")

    def _ensure_nlp(self) -> object:
        """Lazy-load the spaCy pipeline used for word tokenization (same role as training ``nlp_model``)."""
        if self._nlp is None:
            import spacy

            logger.info("Loading spaCy model %r for predict()", self.nlp_model_name)
            self._nlp = spacy.load(self.nlp_model_name)
        return self._nlp

    @staticmethod
    def _mention_tensor_lists_aligned(
        lists: list[list[torch.Tensor]],
    ) -> None:
        n0 = len(lists[0])
        for i, lst in enumerate(lists[1:], start=1):
            if len(lst) != n0:
                raise ValueError(
                    f"Mention tensor count mismatch between fused sources: "
                    f"source 0 has {n0}, source {i} has {len(lst)}. "
                    "Use the same model_type for all sources if spans must align."
                )

    @staticmethod
    def _zscore(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        if v.size == 0:
            return np.array([], dtype=np.float64)
        mean = float(v.mean())
        std = float(v.std())
        if std <= 1e-12:
            return np.zeros_like(v, dtype=np.float64)
        return (v - mean) / std

    @staticmethod
    def _mention_interval_half_open(
        row: dict[str, object],
    ) -> tuple[int, int, int] | None:
        """Document character interval ``[start, end)`` for overlap tests, or ``None``."""
        it = row.get("itext")
        if it is None:
            return None
        it_i = int(it)
        aa = row.get("a_abs")
        bb = row.get("b_abs")
        if aa is not None and bb is not None:
            return (it_i, int(aa), int(bb))
        aa = row.get("a")
        bb = row.get("b")
        if aa is not None and bb is not None:
            return (it_i, int(aa), int(bb))
        return None

    @staticmethod
    def _char_intervals_overlap(
        u: tuple[int, int, int], v: tuple[int, int, int]
    ) -> bool:
        if u[0] != v[0]:
            return False
        _, a1, b1 = u
        _, a2, b2 = v
        return a1 < b2 and a2 < b1

    @staticmethod
    def _span_extent_chars(row: dict[str, object]) -> int:
        iv = Linker._mention_interval_half_open(row)
        if iv is not None:
            return max(iv[2] - iv[1], 0)
        return len(str(row.get("mention", "")))

    @staticmethod
    def _entity_prediction_row(
        item: MentionCandidate,
        *,
        entity_id_predicted: str,
        cluster_membership_prob: float,
        pca_residual: float,
        pca_mahalanobis: float,
        pca_spectral_entropy: float,
        anomaly_score_max_z: float,
        projection_score: float,
    ) -> EntityPredictionRow:
        """Merge mention span fields with clustering outputs; ``score`` is cluster soft membership."""
        base = dataclasses.asdict(item)
        out = cast(EntityPredictionRow, dict(base))
        out["entity_id_predicted"] = entity_id_predicted
        out["score"] = cluster_membership_prob
        out["pca_residual"] = pca_residual
        out["pca_mahalanobis"] = pca_mahalanobis
        out["pca_spectral_entropy"] = pca_spectral_entropy
        out["anomaly_score_max_z"] = anomaly_score_max_z
        out["projection_score"] = projection_score
        return out

    @staticmethod
    def _dedupe_overlapping_prediction_rows(
        rows: list[EntityPredictionRow],
    ) -> list[EntityPredictionRow]:
        """Drop redundant W1/W2/W3 windows that cover the same text region.

        Rows without a usable ``(itext, …)`` interval are never merged with others.

        Overlap is union of intersecting half-open character intervals on the same
        document. Within each connected component, keep the row with highest
        ``score``; ties prefer a shorter span, then lexicographic ``mention``.

        Returned rows are sorted by ``(itext, a_abs or a)`` for stable output.
        """
        n = len(rows)
        if n <= 1:
            return rows

        intervals: list[tuple[int, int, int] | None] = [
            Linker._mention_interval_half_open(r) for r in rows
        ]
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                ui, uj = intervals[i], intervals[j]
                if ui is None or uj is None:
                    continue
                if Linker._char_intervals_overlap(ui, uj):
                    union(i, j)

        comp_members: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            comp_members[find(i)].append(i)

        chosen: list[EntityPredictionRow] = []
        for members in comp_members.values():

            def rank_key(idx: int) -> tuple[float, int, str]:
                r = rows[idx]
                return (
                    -float(r["score"]),
                    Linker._span_extent_chars(r),
                    str(r["mention"]),
                )

            best = min(members, key=rank_key)
            chosen.append(rows[best])

        def sort_key(r: EntityPredictionRow) -> tuple[int, int]:
            it = r.get("itext")
            it_i = int(it) if it is not None else -1
            aa = r.get("a_abs")
            if aa is not None:
                return it_i, int(aa)
            aa = r.get("a")
            return it_i, int(aa) if aa is not None else -1

        chosen.sort(key=sort_key)
        return chosen

    def training_anomaly_metric_summary(self) -> dict[str, dict[str, float]] | None:
        """Quantile summary from stored per-mention PCA metrics (legacy pickles only after fit)."""
        if (
            self.training_pca_residuals is None
            or self.training_pca_mahalanobis is None
            or self.training_pca_spectral_entropy is None
            or len(self.training_pca_residuals) == 0
            or len(self.training_pca_mahalanobis) == 0
        ):
            return None

        residual = np.asarray(self.training_pca_residuals, dtype=np.float64)
        mahal = np.asarray(self.training_pca_mahalanobis, dtype=np.float64)
        entropy = np.asarray(self.training_pca_spectral_entropy, dtype=np.float64)
        combined = np.maximum.reduce(
            [
                self._zscore(residual),
                self._zscore(mahal),
                self._zscore(entropy),
            ]
        )
        quantiles = [0.5, 0.9, 0.95, 0.99]

        def _q(values: np.ndarray) -> dict[str, float]:
            return {
                f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in quantiles
            }

        return {
            "residual": _q(residual),
            "mahalanobis": _q(mahal),
            "spectral_entropy": _q(entropy),
            "combined_max_z": _q(combined),
        }

    def _encode_mentions(
        self,
        texts: Sequence[str],
        max_length: int | None,
        *,
        use_gpu: bool,
    ) -> tuple[torch.Tensor | None, list[MentionCandidate], object]:
        """Run encoders + spaCy and build the fused mention tensor and mention rows.

        Returns ``(fused_tensor, mentions, primary_report_batch)``. ``fused_tensor`` is
        ``None`` when no mentions were extracted. Each mention row carries
        chunk-local bounds ``a``/``b``, absolute bounds ``a_abs``/``b_abs``, ``itext``,
        ``ichunk``, ``word_grouping`` and ``lemma`` (space-joined token lemmas, used for
        KB-match lookups).

        Mentions are filtered with :func:`~pelinker.util.keep_expression_for_prediction`
        (drop windows containing punctuation; drop windows whose tokens are all stop
        words).
        """
        if self.embedding_metadata is None:
            raise ValueError(
                "embedding_metadata is required; set it during fit() or on the Linker."
            )
        self._ensure_hf_models_for_sources(use_gpu=use_gpu)
        nlp = self._ensure_nlp()
        resolved_max_length = max_length if max_length is not None else MAX_LENGTH

        word_groupings = [WordGrouping.W1, WordGrouping.W2, WordGrouping.W3]
        report_batches: list = []
        for src in self.embedding_metadata.sources:
            tok, model = self._hf_models_by_type[src.model_type]
            rb = texts_to_vrep(
                list(texts),
                tok,
                model,
                src.layers_spec,
                word_groupings,
                nlp,
                max_length=resolved_max_length,
            )
            report_batches.append(rb)

        primary = report_batches[0]
        tt_lists = [
            extract_ordered_mention_tensors(rb, keep=keep_expression_for_prediction)
            for rb in report_batches
        ]
        self._mention_tensor_lists_aligned(tt_lists)

        mentions: list[MentionCandidate] = []
        for wg in word_groupings:
            if wg not in primary.available_groupings():
                continue
            expression_container = primary[wg]
            for expr_holder in expression_container.expression_data:
                for expr, _tt in zip(expr_holder.expressions, expr_holder.tt):
                    if not keep_expression_for_prediction(expr):
                        continue
                    mention_text = ""
                    offset: int | None = None
                    if (
                        expr.itext is not None
                        and expr.itext < len(primary.texts)
                        and expr.a is not None
                        and expr.b is not None
                    ):
                        text = primary.texts[expr.itext]
                        if expr.ichunk is not None:
                            offset = primary.chunk_mapper.map_chunk_to_text(
                                expr.itext, expr.ichunk
                            )
                            mention_text = text[offset + expr.a : offset + expr.b]
                        else:
                            mention_text = text[expr.a : expr.b]
                    lemma = " ".join(t.lemma for t in expr.tokens)
                    mentions.append(
                        MentionCandidate(
                            mention=mention_text,
                            a=expr.a,
                            b=expr.b,
                            a_abs=(
                                expr.a + offset
                                if expr.a is not None and offset is not None
                                else None
                            ),
                            b_abs=(
                                expr.b + offset
                                if expr.b is not None and offset is not None
                                else None
                            ),
                            itext=expr.itext,
                            ichunk=expr.ichunk,
                            word_grouping=wg,
                            lemma=lemma,
                        )
                    )

        if not tt_lists[0]:
            return None, mentions, primary

        fused_rows: list[torch.Tensor] = []
        for i in range(len(tt_lists[0])):
            fused_rows.append(
                torch.cat([tts[i] for tts in tt_lists], dim=-1),
            )
        tt = torch.stack(fused_rows, dim=0)
        return tt, mentions, primary

    def predict(
        self,
        texts: Sequence[str],
        max_length: int | None = None,
        threshold: float = 0.0,
        *,
        use_gpu: bool = False,
        include_mention_anomaly: bool = False,
        include_debug_mentions: bool = False,
        include_prediction_kb_validation: bool = False,
    ) -> LinkerPredictResult:
        """
        Predict entities for input texts.

        With multiple ``embedding_metadata.sources``, runs ``texts_to_vrep`` per source
        (cached by ``model_type``), concatenates mention tensors along the feature axis in
        source order, then applies the fitted transformer and clusterer. Mention counts
        must match across sources (typically the same ``model_type`` for all sources).

        Tokenization uses the spaCy pipeline named by ``nlp_model_name`` (set from
        ``EmbeddingTrainingConfig.nlp_model`` during corpus embedding, else default
        ``en_core_web_trf``).

        Each ``entities`` row includes ``score``: HDBSCAN approximate cluster
        membership probability from ``approximate_predict`` on UMAP coordinates.
        The ``threshold`` argument drops rows whose ``score`` is below that minimum.

        When ``include_mention_anomaly`` or ``include_debug_mentions`` is true,
        :attr:`LinkerPredictResult.debug_mentions` lists one diagnostic row per extracted
        mention (same single encode and PCA→UMAP pass as predictions). Use
        :meth:`LinkerPredictResult.to_dict` with ``include_debug=True`` to emit the legacy
        ``mention_anomaly`` key for JSON consumers.

        ``kb_training_entity`` (human label from ``labels_map`` for the predicted id)
        is attached only when mention-debug or KB validation is requested, not on the
        default prediction path.

        When ``include_prediction_kb_validation`` is true, each row in ``entities`` gains
        validation-only fields comparing mention lemmas to KB training ``entity`` labels
        (same index as training-time matching): ``kb_training_entity_from_lemma``,
        ``kb_training_entity_for_prediction``, ``lemma_kb_matches_predicted_entity``.
        When debug rows are also returned, those fields are copied onto the matching
        mention row via ``mention_source_index``.
        """
        want_debug = include_mention_anomaly or include_debug_mentions
        tt, mentions, primary = self._encode_mentions(
            texts, max_length, use_gpu=use_gpu
        )

        if tt is None:
            return LinkerPredictResult(
                entities=[],
                debug_mentions=[] if want_debug else None,
            )

        kb_lemma_by_wg: dict[WordGrouping, dict[str, str]] | None = None
        if want_debug or include_prediction_kb_validation:
            nlp = self._ensure_nlp()
            kb_lemma_by_wg = self._kb_lemma_index_by_wg(nlp)

        predictions, mention_anomaly = self._predict_with_clustering(
            tt,
            mentions,
            threshold=threshold,
            mention_anomaly_rows=want_debug,
            kb_lemma_by_wg=kb_lemma_by_wg,
        )

        if include_prediction_kb_validation:
            if kb_lemma_by_wg is None:
                raise ValueError(
                    "kb_lemma_by_wg missing for include_prediction_kb_validation "
                    "(internal error: index should have been built)"
                )
            enrich_entity_predictions_kb_validation(
                cast(list[dict[str, object]], predictions),
                kb_lemma_by_wg,
                self.labels_map,
            )

        preds_obj = cast(list[dict[str, object]], predictions)
        if want_debug or include_prediction_kb_validation:
            for row in preds_obj:
                eid = row.get("entity_id_predicted")
                row["kb_training_entity"] = (
                    self.labels_map.get(str(eid)) if eid is not None else None
                )

        if mention_anomaly is not None:
            self._merge_prediction_fields_into_debug_mentions(
                mention_anomaly,
                preds_obj,
                include_kb_validation_fields=include_prediction_kb_validation,
            )

        for item in preds_obj:
            item.pop("lemma", None)

        return LinkerPredictResult(
            entities=preds_obj,
            debug_mentions=mention_anomaly,
        )

    def _build_mention_anomaly_rows(
        self,
        mentions: list[MentionCandidate],
        screener_neg: np.ndarray,
        screener_margin: np.ndarray,
        residuals: np.ndarray,
        mahalanobis: np.ndarray,
        spectral_entropy: np.ndarray,
        combined: np.ndarray,
        projection_scores: np.ndarray,
        kb_lemma_by_wg: dict[WordGrouping, dict[str, str]],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for i, item in enumerate(mentions):
            wg = item.word_grouping
            lemma = item.lemma
            kb_property = lookup_kb_training_entity_label(
                wg if isinstance(wg, WordGrouping) else None,
                str(lemma),
                kb_lemma_by_wg,
            )
            rows.append(
                {
                    "mention": item.mention,
                    "a": item.a,
                    "b": item.b,
                    "a_abs": item.a_abs,
                    "b_abs": item.b_abs,
                    "itext": item.itext,
                    "ichunk": item.ichunk,
                    "word_grouping": wg.name if isinstance(wg, WordGrouping) else None,
                    "lemma": lemma,
                    "is_kb_match": kb_property is not None,
                    "kb_property_match": kb_property,
                    "pca_residual": float(residuals[i]),
                    "pca_mahalanobis": float(mahalanobis[i]),
                    "pca_spectral_entropy": float(spectral_entropy[i]),
                    "anomaly_score_max_z": float(combined[i]),
                    "projection_score": float(projection_scores[i]),
                    "screener_is_negative": bool(screener_neg[i]),
                    "screener_decision": float(screener_margin[i]),
                }
            )
        return rows

    def _predict_with_clustering(
        self,
        embeddings: torch.Tensor,
        mentions: list[MentionCandidate],
        threshold: float = 0.0,
        *,
        mention_anomaly_rows: bool = False,
        kb_lemma_by_wg: dict[WordGrouping, dict[str, str]] | None = None,
    ) -> tuple[list[EntityPredictionRow], list[dict[str, object]] | None]:
        """
        Predict entities using clustering approach.

        Mentions classified as negative by the screener are dropped immediately: no
        PCA/UMAP, no HDBSCAN ``approximate_predict``, and no anomaly metrics for them.

        Each entity row includes ``score``: HDBSCAN soft cluster membership from
        ``approximate_predict`` on UMAP coordinates (same scale as ``threshold``).

        Args:
            embeddings: Tensor of shape (n_mentions, embedding_dim)
            mentions: Extracted mention candidates in row order with ``embeddings``.
            threshold: Minimum cluster membership probability required to return
                a prediction (compared to each row's ``score``).

        Returns:
            ``(entity_predictions, mention_anomaly_rows_or_none)``. Anomaly rows are
            returned only when ``mention_anomaly_rows`` is true (requires
            ``kb_lemma_by_wg``).
        """
        if self.transformer is None or self.clusterer is None or self.screener is None:
            raise ValueError(
                "Screener, Transformer and Clusterer must be fitted before prediction"
            )
        if mention_anomaly_rows and kb_lemma_by_wg is None:
            raise ValueError(
                "kb_lemma_by_wg is required when mention_anomaly_rows is true"
            )

        # Convert to numpy
        embeddings_np = embeddings.detach().cpu().numpy()

        screener_neg = self.screener.predict_is_negative(embeddings_np)
        idx_keep = np.flatnonzero(~screener_neg)
        screener_margin: np.ndarray | None
        if mention_anomaly_rows:
            screener_margin = self.screener.decision_function(embeddings_np)
        else:
            screener_margin = None

        candidates: list[EntityPredictionRow] = []
        n_mentions = len(mentions)

        if len(idx_keep) == 0:
            if mention_anomaly_rows:
                assert screener_margin is not None
                assert kb_lemma_by_wg is not None
                nan_vec = np.full(n_mentions, np.nan, dtype=np.float64)
                return [], self._build_mention_anomaly_rows(
                    mentions,
                    screener_neg,
                    screener_margin,
                    nan_vec,
                    nan_vec,
                    nan_vec,
                    nan_vec,
                    nan_vec,
                    kb_lemma_by_wg,
                )
            return [], None

        emb_k = embeddings_np[idx_keep]
        _umap_k, _, res_k, mah_k, ent_k = self.transformer.transform(emb_k)
        cl_k, cp_k = approximate_predict(self.clusterer, _umap_k)
        cl_arr = cl_k.astype(np.int64, copy=False)
        cp_arr = np.asarray(cp_k, dtype=np.float64).ravel()
        combined_k = np.maximum.reduce(
            [
                self._zscore(res_k),
                self._zscore(mah_k),
                self._zscore(ent_k),
            ]
        )
        mo = self.projection
        if mo is not None:
            X3 = np.column_stack(
                [
                    np.asarray(res_k, dtype=np.float64),
                    np.asarray(mah_k, dtype=np.float64),
                    np.asarray(ent_k, dtype=np.float64),
                ]
            )
            oov_scores_k = mo.score(X3)
            oov_gate_k = mo.is_oov(X3)
        else:
            oov_scores_k = np.full(len(idx_keep), np.nan, dtype=np.float64)
            oov_gate_k = np.zeros(len(idx_keep), dtype=bool)

        for j, mention_i in enumerate(idx_keep):
            item = mentions[int(mention_i)]
            if bool(oov_gate_k[j]):
                continue
            cluster_id = int(cl_arr[j])
            cluster_prob = float(cp_arr[j])
            # Skip HDBSCAN outliers and low-confidence assignments.
            if cluster_id == -1 or cluster_prob < threshold:
                continue

            # Find all entities in the same cluster
            cluster_entities = [
                entity_id
                for entity_id, cid in self.cluster_assignments.items()
                if cid == cluster_id
            ]

            # Skip clusters that have no mapped entities from the training vocabulary.
            if not cluster_entities:
                continue

            # For now, return the first entity in the cluster
            predicted_entity = cluster_entities[0]
            row = self._entity_prediction_row(
                item,
                entity_id_predicted=predicted_entity,
                cluster_membership_prob=cluster_prob,
                pca_residual=float(res_k[j]),
                pca_mahalanobis=float(mah_k[j]),
                pca_spectral_entropy=float(ent_k[j]),
                anomaly_score_max_z=float(combined_k[j]),
                projection_score=float(oov_scores_k[j]),
            )
            cast(dict[str, object], row)["mention_source_index"] = int(mention_i)
            candidates.append(row)

        deduped = self._dedupe_overlapping_prediction_rows(candidates)
        if mention_anomaly_rows:
            assert screener_margin is not None
            assert kb_lemma_by_wg is not None
            residuals = np.full(n_mentions, np.nan, dtype=np.float64)
            mahalanobis = np.full(n_mentions, np.nan, dtype=np.float64)
            spectral_entropy = np.full(n_mentions, np.nan, dtype=np.float64)
            combined_full = np.full(n_mentions, np.nan, dtype=np.float64)
            projection_full = np.full(n_mentions, np.nan, dtype=np.float64)
            residuals[idx_keep] = res_k
            mahalanobis[idx_keep] = mah_k
            spectral_entropy[idx_keep] = ent_k
            combined_full[idx_keep] = combined_k
            projection_full[idx_keep] = oov_scores_k
            return deduped, self._build_mention_anomaly_rows(
                mentions,
                screener_neg,
                screener_margin,
                residuals,
                mahalanobis,
                spectral_entropy,
                combined_full,
                projection_full,
                kb_lemma_by_wg,
            )
        return deduped, None

    def _kb_lemma_index_by_wg(self, nlp: object) -> dict[WordGrouping, dict[str, str]]:
        """Build lemma→KB training-entity index; see :func:`pelinker.linker_kb_lemma.build_kb_lemma_index`."""
        return build_kb_lemma_index(self.labels_map, nlp)

    def compute_mention_anomaly(
        self,
        texts: Sequence[str],
        max_length: int | None = None,
        *,
        use_gpu: bool = False,
    ) -> list[dict[str, object]]:
        """Per-mention PCA residual / Mahalanobis with KB-match and screener fields.

        Delegates to :meth:`predict` with ``include_mention_anomaly=True`` so encoding,
        screening, and ``EmbeddingTransformer.transform`` run once (no duplicate pass).

        Screened-negative rows use NaN for PCA metrics. Each row includes
        ``screener_is_negative``, ``screener_decision``, plus ``is_kb_match`` /
        ``kb_property_match`` (lemma vs KB under :class:`WordGrouping`).
        """
        out = self.predict(
            texts,
            max_length=max_length,
            threshold=0.0,
            use_gpu=use_gpu,
            include_mention_anomaly=True,
        )
        rows = out.debug_mentions
        return list(rows) if rows is not None else []
