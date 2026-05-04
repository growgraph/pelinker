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
    TransformConfig,
)
from pelinker.negative_screener import NegativeClassScreener
from pelinker.analysis import (
    compute_clustering_fit_metrics,
    fit_negative_screener_with_metrics,
    mention_frame_from_embedding_paths,
    split_by_negative_label,
)
from pelinker.reporting import (
    ClusteringFitMetrics,
    ClusteringHyperparameters,
    ClusteringReport,
    NegativeScreenerInSampleMetrics,
)
from pelinker.embedder import embed_kb_corpus
from pelinker.embedding_fusion import (
    fused_property_vectors_from_paths,
    property_fused_dataframe_for_linker_order,
)
from pelinker.linker_cluster_training import (
    cluster_composition_from_training_frame,
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
    WordGrouping,
)
from pelinker.transform import EmbeddingTransformer
from pelinker.util import (
    extract_ordered_mention_tensors,
    keep_expression_for_prediction,
    load_models,
    texts_to_vrep,
)

logger = logging.getLogger(__name__)


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
    anomaly_score_max_z: float


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
    ) -> dict[str, object]:
        """Serialize for JSON APIs. Debug rows use the legacy key ``mention_anomaly``."""
        entities_out: list[dict[str, object]] = []
        for r in self.entities:
            e = dict(r)
            if strip_mention_source_index:
                e.pop("mention_source_index", None)
            if not include_entity_anomaly_metrics:
                e.pop("pca_residual", None)
                e.pop("pca_mahalanobis", None)
                e.pop("anomaly_score_max_z", None)
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


def _parquet_read_config_from_fit(
    fit_cfg: LinkerFitConfig,
) -> ClusteringOptimizationConfig:
    """Map :class:`LinkerFitConfig` to the parquet batch fields of :class:`ClusteringOptimizationConfig`."""
    return ClusteringOptimizationConfig(
        min_class_size=fit_cfg.min_class_size,
        batch_size=fit_cfg.batch_size,
        n_embedding_batches=fit_cfg.n_embedding_batches,
        negative_screener=fit_cfg.negative_screener,
    )


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
        self.training_umap_clustering: np.ndarray | None = None
        self.training_umap_visualization: np.ndarray | None = None
        self.training_pca_reduced: np.ndarray | None = None
        self.cluster_composition: ClusterCompositionSnapshot | None = None
        self.cluster_consensus_names: dict[int, str] = {}
        self.screener: NegativeClassScreener | None = None
        self.screener_in_sample_metrics: NegativeScreenerInSampleMetrics | None = None
        self.clustering_fit_metrics: ClusteringFitMetrics | None = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_models_by_type: dict[str, tuple[object, object]] = {}
        self.nlp_model_name: str = kwargs.pop("nlp_model_name", "en_core_web_trf")
        self._nlp: object | None = None
        self._fit_clustering_report: ClusteringReport | None = None

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
        if "embedding_metadata" not in pe_model.__dict__:
            pe_model.embedding_metadata = None
        if "kb_config" not in pe_model.__dict__:
            pe_model.kb_config = None
        if "_hf_models_by_type" not in pe_model.__dict__:
            pe_model._hf_models_by_type = {}
        if "training_cluster_frame" not in pe_model.__dict__:
            pe_model.training_cluster_frame = None
        if "training_pca_residuals" not in pe_model.__dict__:
            pe_model.training_pca_residuals = None
        if "training_pca_mahalanobis" not in pe_model.__dict__:
            pe_model.training_pca_mahalanobis = None
        if "training_umap_clustering" not in pe_model.__dict__:
            pe_model.training_umap_clustering = None
        if "training_umap_visualization" not in pe_model.__dict__:
            pe_model.training_umap_visualization = None
        if "training_pca_reduced" not in pe_model.__dict__:
            pe_model.training_pca_reduced = None
        if "cluster_composition" not in pe_model.__dict__:
            pe_model.cluster_composition = None
        if "cluster_consensus_names" not in pe_model.__dict__:
            pe_model.cluster_consensus_names = {}
        if "nlp_model_name" not in pe_model.__dict__:
            pe_model.nlp_model_name = "en_core_web_trf"
        if "_nlp" not in pe_model.__dict__:
            pe_model._nlp = None
        if "screener_in_sample_metrics" not in pe_model.__dict__:
            pe_model.screener_in_sample_metrics = None
        if "clustering_fit_metrics" not in pe_model.__dict__:
            pe_model.clustering_fit_metrics = None
        if "_fit_clustering_report" not in pe_model.__dict__:
            pe_model._fit_clustering_report = None
        return pe_model

    def take_fit_clustering_report(self) -> ClusteringReport | None:
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
        self.training_umap_clustering = None
        self.training_umap_visualization = None
        self.training_pca_reduced = None

    def build_clustering_report(self) -> ClusteringReport | None:
        """
        Build a :class:`~pelinker.reporting.ClusteringReport` when full training rows exist.

        After a normal :meth:`fit`, heavy training payloads are removed for prediction; use
        :meth:`take_fit_clustering_report` immediately after fitting instead.

        This method remains useful for **legacy** pickled linkers that still embed training
        arrays, or for tests that skip stripping.
        """
        tcf = self.training_cluster_frame
        if (
            tcf is None
            or self.training_pca_residuals is None
            or self.training_pca_mahalanobis is None
            or self.training_umap_clustering is None
            or self.training_umap_visualization is None
            or self.training_pca_reduced is None
            or self.clustering_fit_metrics is None
        ):
            return None
        n = len(tcf)
        if (
            len(self.training_pca_residuals) != n
            or len(self.training_pca_mahalanobis) != n
            or self.training_umap_clustering.shape[0] != n
            or self.training_umap_visualization.shape[0] != n
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

        assignments = tcf[["entity", "cluster"]].copy()
        for col in ("pmid", "mention"):
            if col in tcf.columns:
                assignments[col] = tcf[col]

        number_properties = int(tcf["entity"].nunique())

        return ClusteringReport(
            hyperparameters=ClusteringHyperparameters(
                min_cluster_size=m.min_cluster_size
            ),
            best_score=dbcv_f,
            number_properties=number_properties,
            n_clusters_emergent=m.n_clusters_emergent,
            metrics_df=metrics_df,
            assignments=assignments,
            pca_residuals=np.asarray(self.training_pca_residuals, dtype=np.float64),
            pca_mahalanobis=np.asarray(self.training_pca_mahalanobis, dtype=np.float64),
            umap_clustering=np.asarray(self.training_umap_clustering, dtype=np.float64),
            umap_visualization=np.asarray(
                self.training_umap_visualization, dtype=np.float64
            ),
            pca_reduced=np.asarray(self.training_pca_reduced, dtype=np.float64),
            negative_screener_cv=None,
            ari=m.ari,
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
                        fused like ``estimate_model_clustering`` (inner join on keys, concat
                        embeddings). Order must match ``embedding_metadata.sources``.
                        If None, ``embed_kb_corpus`` is run (one output file per source).
            transform_config: TransformConfig instance
            min_cluster_size: HDBSCAN ``min_cluster_size`` (choose upstream, e.g. via
                ``run/analysis/clustering_quality.py`` / ``estimate_model_clustering``).
            fit_config: Parquet read batching, mention-per-entity filter, and screener settings.
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
            read_cfg = _parquet_read_config_from_fit(fc)

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
            raw = mention_frame_from_embedding_paths(
                embeddings_paths,
                optimization_config=read_cfg,
                show_read_progress=True,
            )
            if raw is None or len(raw) == 0:
                raise ValueError(
                    "No mention-level embedding rows loaded from parquet (check paths "
                    "and columns pmid, entity, mention, embed)."
                )

            # prepared = drop_entities_with_few_mentions(
            #     raw,
            #     fc.min_class_size,
            #     negative_label=fc.negative_screener.negative_label,
            # )

            self._fit_clustering_on_prepared_mentions(
                prepared=raw,
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
        """Fit screener on all prepared rows, then PCA/UMAP + HDBSCAN on non-negative rows only."""
        ns_cfg = fit_cfg.negative_screener
        self.screener, self.screener_in_sample_metrics = (
            fit_negative_screener_with_metrics(
                prepared,
                ns_cfg,
            )
        )
        _, manifold_df = split_by_negative_label(prepared, ns_cfg.negative_label)
        if len(manifold_df) == 0:
            raise ValueError(
                "No rows left after excluding negative-label mentions for manifold fit"
            )

        embeddings = np.stack(manifold_df["embed"].values).astype(
            np.float32, copy=False
        )
        self.transform_config = transform_config
        self.transformer = EmbeddingTransformer(transform_config)
        umap_clustering, umap_visualization, pca_residuals, pca_mahalanobis = (
            self.transformer.fit_transform(embeddings)
        )
        embeddings_normed = self.transformer._l2_normalize_rows(embeddings)
        pca_reduced = self.transformer.pca.transform(embeddings_normed)
        self.training_pca_residuals = np.asarray(pca_residuals, dtype=np.float32)
        self.training_pca_mahalanobis = np.asarray(pca_mahalanobis, dtype=np.float32)
        self.training_umap_clustering = np.asarray(umap_clustering, dtype=np.float32)
        self.training_umap_visualization = np.asarray(
            umap_visualization, dtype=np.float32
        )
        self.training_pca_reduced = np.asarray(pca_reduced, dtype=np.float32)

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            gen_min_span_tree=True,
            prediction_data=True,
        )
        cluster_labels_arr = self.clusterer.fit_predict(umap_clustering)
        cluster_labels = cluster_labels_arr.astype(int, copy=False)
        self.clustering_fit_metrics = compute_clustering_fit_metrics(
            self.clusterer,
            manifold_df,
            min_cluster_size=min_cluster_size,
            cluster_labels=cluster_labels,
        )

        tc_cols = ["pmid", "entity", "mention"]
        missing = [c for c in tc_cols if c not in manifold_df.columns]
        if missing:
            raise ValueError(
                "Prepared mention frame missing columns required for "
                f"training_cluster_frame: {missing}"
            )
        self.training_cluster_frame = manifold_df[tc_cols].copy()
        self.training_cluster_frame["cluster"] = cluster_labels

        self.cluster_composition = cluster_composition_from_training_frame(
            self.training_cluster_frame
        )
        self.cluster_consensus_names = consensus_cluster_names(self.cluster_composition)

        self.cluster_assignments = _provisional_cluster_assignments_from_training_frame(
            self.labels_map,
            self.training_cluster_frame,
        )
        self.vocabulary = sorted(self.cluster_assignments.keys())
        if not self.vocabulary:
            raise ValueError(
                "No entity_ids received provisional cluster assignments after fit "
                "(check labels_map and training entity labels)"
            )

        if kb_config is not None:
            if kb_config.entity_count is None:
                self.kb_config = replace(kb_config, entity_count=len(self.vocabulary))
            else:
                self.kb_config = kb_config

        fit_report = self.build_clustering_report()
        self._strip_training_metrics_for_prediction()
        self._fit_clustering_report = fit_report

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
            n_embedding_batches=cfg.n_embedding_batches,
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
        anomaly_score_max_z: float,
    ) -> EntityPredictionRow:
        """Merge mention span fields with clustering outputs; ``score`` is cluster soft membership."""
        base = dataclasses.asdict(item)
        out = cast(EntityPredictionRow, dict(base))
        out["entity_id_predicted"] = entity_id_predicted
        out["score"] = cluster_membership_prob
        out["pca_residual"] = pca_residual
        out["pca_mahalanobis"] = pca_mahalanobis
        out["anomaly_score_max_z"] = anomaly_score_max_z
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
            or len(self.training_pca_residuals) == 0
            or len(self.training_pca_mahalanobis) == 0
        ):
            return None

        residual = np.asarray(self.training_pca_residuals, dtype=np.float64)
        mahal = np.asarray(self.training_pca_mahalanobis, dtype=np.float64)
        combined = np.maximum(self._zscore(residual), self._zscore(mahal))
        quantiles = [0.5, 0.9, 0.95, 0.99]

        def _q(values: np.ndarray) -> dict[str, float]:
            return {
                f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in quantiles
            }

        return {
            "residual": _q(residual),
            "mahalanobis": _q(mahal),
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
        combined: np.ndarray,
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
                    "anomaly_score_max_z": float(combined[i]),
                    "screener_is_negative": bool(screener_neg[i]),
                    "screener_decision": float(screener_margin[i]),
                }
            )
        return rows

    def _mention_anomaly_from_full_vectors(
        self,
        mentions: list[MentionCandidate],
        screener_neg: np.ndarray,
        screener_margin: np.ndarray,
        residuals: np.ndarray,
        mahalanobis: np.ndarray,
        combined: np.ndarray,
        kb_lemma_by_wg: dict[WordGrouping, dict[str, str]],
    ) -> list[dict[str, object]]:
        return self._build_mention_anomaly_rows(
            mentions,
            screener_neg,
            screener_margin,
            residuals,
            mahalanobis,
            combined,
            kb_lemma_by_wg,
        )

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
                return [], self._mention_anomaly_from_full_vectors(
                    mentions,
                    screener_neg,
                    screener_margin,
                    nan_vec,
                    nan_vec,
                    nan_vec,
                    kb_lemma_by_wg,
                )
            return [], None

        emb_k = embeddings_np[idx_keep]
        _umap_k, _, res_k, mah_k = self.transformer.transform(emb_k)
        cl_k, cp_k = approximate_predict(self.clusterer, _umap_k)
        cl_arr = cl_k.astype(np.int64, copy=False)
        cp_arr = np.asarray(cp_k, dtype=np.float64).ravel()
        combined_k = np.maximum(self._zscore(res_k), self._zscore(mah_k))

        for j, mention_i in enumerate(idx_keep):
            item = mentions[int(mention_i)]
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
                anomaly_score_max_z=float(combined_k[j]),
            )
            cast(dict[str, object], row)["mention_source_index"] = int(mention_i)
            candidates.append(row)

        deduped = self._dedupe_overlapping_prediction_rows(candidates)
        if mention_anomaly_rows:
            assert screener_margin is not None
            assert kb_lemma_by_wg is not None
            residuals = np.full(n_mentions, np.nan, dtype=np.float64)
            mahalanobis = np.full(n_mentions, np.nan, dtype=np.float64)
            combined_full = np.full(n_mentions, np.nan, dtype=np.float64)
            residuals[idx_keep] = res_k
            mahalanobis[idx_keep] = mah_k
            combined_full[idx_keep] = combined_k
            return deduped, self._mention_anomaly_from_full_vectors(
                mentions,
                screener_neg,
                screener_margin,
                residuals,
                mahalanobis,
                combined_full,
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
