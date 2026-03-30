import tempfile
from collections.abc import Sequence
from dataclasses import replace

import torch
import joblib
import numpy as np
from typing import Optional
import hdbscan
from hdbscan import approximate_predict
import pathlib
import logging

from pelinker.config import (
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingTrainingConfig,
    KBConfig,
    TransformConfig,
)
from pelinker.embedder import embed_kb_corpus
from pelinker.embedding_fusion import (
    fused_property_vectors_from_paths,
    property_fused_dataframe_for_linker_order,
)
from pelinker.onto import WordGrouping
from pelinker.transform import EmbeddingTransformer
from pelinker.util import load_models, texts_to_vrep

logger = logging.getLogger(__name__)


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
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_models_by_type: dict[str, tuple[object, object]] = {}

    @classmethod
    def filter_report(cls, report, thr_score):
        report["entities"] = [r for r in report["entities"] if r["score"] >= thr_score]
        return report

    def dump(self, file_spec):
        joblib.dump(self, f"{file_spec}.gz", compress=3)

    @classmethod
    def load(cls, file_spec):
        pe_model = joblib.load(f"{file_spec}.gz")
        if "embedding_metadata" not in pe_model.__dict__:
            pe_model.embedding_metadata = None
        if "kb_config" not in pe_model.__dict__:
            pe_model.kb_config = None
        if "_hf_models_by_type" not in pe_model.__dict__:
            pe_model._hf_models_by_type = {}
        return pe_model

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
        min_cluster_size: Optional[int] = None,
        kb_labels: Optional[set[str]] = None,
        optimize_clustering: bool = False,
        clustering_optimization_config: Optional[ClusteringOptimizationConfig] = None,
        embedding_training: EmbeddingTrainingConfig | None = None,
        embedding_metadata: EmbeddingModelMetadata | None = None,
        kb_config: KBConfig | None = None,
    ):
        """
        Fit the Linker model with embeddings.

        This method handles two main parts:
        a) Loading and processing embeddings (from file or direct array)
        b) Clustering the embeddings

        Args:
            embeddings: Path or sequence of paths to parquet file(s) (columns ``property``,
                        ``embed``; mention-level rows). Order must match
                        ``embedding_metadata.sources``. Vectors are averaged per property per
                        file, then concatenated across files for each property (intersection).
                        If None, ``embed_kb_corpus`` is run (one output file per source).
            transform_config: TransformConfig instance
            min_cluster_size: Minimum cluster size for HDBSCAN. If None and optimize_clustering=False,
                            uses default from transform_config or 20.
            kb_labels: Set of KB property labels to filter by (only used when embeddings is a file path)
            optimize_clustering: If True, optimize min_cluster_size using grid search
            clustering_optimization_config: Config object for clustering optimization.
            embedding_training: Corpus paths and embedding runtime. Required when embeddings=None.
            embedding_metadata: If provided, overrides or sets ``self.embedding_metadata`` for
                this fit (required when embeddings=None unless already set on the linker).
            kb_config: Knowledge-base metadata stored on the linker; ``entity_count`` is set
                from fitted vocabulary when omitted (None).

        Returns:
            self
        """

        is_temporary = False
        embeddings_paths: list[pathlib.Path] = []

        try:
            if embedding_metadata is not None:
                self.embedding_metadata = embedding_metadata

            if embeddings is None:
                if embedding_training is None:
                    raise ValueError(
                        "embedding_training is required when embeddings is None. "
                        "Provide embeddings path or EmbeddingTrainingConfig(...)."
                    )
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

                logger.info("Step (a): Embedding corpus (%s source(s))...", k)
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
                    "Step (a): Using provided embeddings (%s file(s)): %s",
                    len(embeddings_paths),
                    embeddings_paths,
                )

            if optimize_clustering:
                logger.info("Optimizing clustering parameters...")
                min_cluster_size = self._optimize_clustering(
                    embeddings_paths,
                    transform_config,
                    kb_labels,
                    clustering_optimization_config,
                )

            logger.info(
                "Loading embeddings from %s parquet file(s)", len(embeddings_paths)
            )
            embeddings_data, entity_ids = self._load_fused_embeddings_from_files(
                embeddings_paths, kb_labels
            )
            self.vocabulary = entity_ids

            self.transform_config = transform_config

            self.transformer = EmbeddingTransformer(self.transform_config)
            umap_clustering, _ = self.transformer.fit_transform(embeddings_data)

            if min_cluster_size is None:
                min_cluster_size = 20

            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                gen_min_span_tree=True,
                prediction_data=True,
            )
            cluster_labels = self.clusterer.fit_predict(umap_clustering)

            self.cluster_assignments = {
                entity_id: int(cluster_id)
                for entity_id, cluster_id in zip(self.vocabulary, cluster_labels)
            }

            if kb_config is not None:
                if kb_config.entity_count is None:
                    self.kb_config = replace(
                        kb_config, entity_count=len(self.vocabulary)
                    )
                else:
                    self.kb_config = kb_config

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

    def _load_embeddings_from_file(
        self, embeddings_path: pathlib.Path, kb_labels: set[str] | None = None
    ) -> tuple[np.ndarray, list[str]]:
        """Backward-compatible single-file loader; delegates to fused multi-file path."""
        return self._load_fused_embeddings_from_files([embeddings_path], kb_labels)

    def _load_fused_embeddings_from_files(
        self,
        embeddings_paths: Sequence[pathlib.Path],
        kb_labels: set[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Per-file per-property mean embeddings, intersection across sources, concat features.
        """
        logger.info(
            "Reading %s parquet source(s) and fusing per-property vectors...",
            len(embeddings_paths),
        )
        fused = fused_property_vectors_from_paths(embeddings_paths, kb_labels)
        if not fused:
            raise ValueError("No fused property vectors (empty intersection or inputs)")

        entity_ids: list[str] = []
        embeddings_list: list[np.ndarray] = []
        for prop_label in sorted(fused.keys()):
            entity_id = None
            for eid, label in self.labels_map.items():
                if label == prop_label:
                    entity_id = eid
                    break

            if entity_id is not None:
                entity_ids.append(entity_id)
                embeddings_list.append(fused[prop_label])
            else:
                logger.warning(
                    "Property label '%s' not found in labels_map, skipping", prop_label
                )

        if len(embeddings_list) == 0:
            raise ValueError("No valid embeddings after mapping to entity_ids")

        embeddings = np.stack(embeddings_list)
        logger.info(
            "Embedded %s KB properties into %s-dimensional fused vectors",
            len(embeddings),
            embeddings.shape[1],
        )
        return embeddings, entity_ids

    def _optimize_clustering(
        self,
        embeddings_paths: Sequence[pathlib.Path],
        transform_config: TransformConfig,
        kb_labels: set[str] | None,
        optimization_config: Optional[ClusteringOptimizationConfig] = None,
    ) -> int:
        """
        Optimize min_cluster_size on the same property-level fused matrix as ``fit`` uses
        (not mention-level; differs from ``estimate_model_clustering`` on raw corpora).
        """
        from pelinker.analysis import estimate_clustering_from_frame

        effective_config = optimization_config or ClusteringOptimizationConfig()

        fused = fused_property_vectors_from_paths(embeddings_paths, kb_labels)
        dfr = property_fused_dataframe_for_linker_order(fused, self.labels_map)
        if len(dfr) == 0:
            logger.warning(
                "Clustering optimization skipped (empty fused property frame)"
            )
            return effective_config.min_class_size

        clustering_report = estimate_clustering_from_frame(
            dfr,
            transform_config,
            optimization_config=effective_config,
            selected_labels=None,
            all_metrics_dfs=None,
            aggregation_level="property",
        )

        if clustering_report is None:
            logger.warning(
                "Clustering estimation failed, using default min_cluster_size"
            )
            return effective_config.min_class_size

        logger.info(
            "Optimal min_cluster_size: %s (score: %.3f)",
            clustering_report.hyperparameters.min_cluster_size,
            clustering_report.best_score,
        )
        return clustering_report.hyperparameters.min_cluster_size

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

    @staticmethod
    def _extract_ordered_mention_tensors(report_batch) -> list[torch.Tensor]:
        word_groupings = [WordGrouping.W1, WordGrouping.W2, WordGrouping.W3]
        tt_list: list[torch.Tensor] = []
        for wg in word_groupings:
            if wg not in report_batch.available_groupings():
                continue
            expression_container = report_batch[wg]
            for expr_holder in expression_container.expression_data:
                for _expr, tt in zip(expr_holder.expressions, expr_holder.tt):
                    tt_list.append(tt)
        return tt_list

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

    def predict(
        self,
        texts,
        nlp,
        max_length,
        threshold: float = 0.0,
        *,
        use_gpu: bool = False,
    ):
        """
        Predict entities for input texts.

        With multiple ``embedding_metadata.sources``, runs ``texts_to_vrep`` per source
        (cached by ``model_type``), concatenates mention tensors along the feature axis in
        source order, then applies the fitted transformer and clusterer. Mention counts
        must match across sources (typically the same ``model_type`` for all sources).
        """
        if self.embedding_metadata is None:
            raise ValueError(
                "embedding_metadata is required for predict(); set it during fit() or on the Linker."
            )
        self._ensure_hf_models_for_sources(use_gpu=use_gpu)

        word_groupings = [WordGrouping.W1, WordGrouping.W2, WordGrouping.W3]
        report_batches: list = []
        for src in self.embedding_metadata.sources:
            tok, model = self._hf_models_by_type[src.model_type]
            rb = texts_to_vrep(
                texts,
                tok,
                model,
                layers_spec=src.layers_spec,
                word_modes=word_groupings,
                nlp=nlp,
                max_length=max_length,
            )
            report_batches.append(rb)

        primary = report_batches[0]
        tt_lists = [self._extract_ordered_mention_tensors(rb) for rb in report_batches]
        self._mention_tensor_lists_aligned(tt_lists)

        vocabulary: list[dict[str, object]] = []
        for wg in word_groupings:
            if wg not in primary.available_groupings():
                continue
            expression_container = primary[wg]
            for expr_holder in expression_container.expression_data:
                for expr, _tt in zip(expr_holder.expressions, expr_holder.tt):
                    mention_text = ""
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
                    vocabulary.append(
                        {
                            "mention": mention_text,
                            "a": expr.a,
                            "b": expr.b,
                            "itext": expr.itext,
                            "ichunk": expr.ichunk,
                            "word_grouping": wg,
                        }
                    )

        if not tt_lists[0]:
            return {"entities": [], "word_groupings": {}}

        fused_rows: list[torch.Tensor] = []
        for i in range(len(tt_lists[0])):
            fused_rows.append(
                torch.cat([tts[i] for tts in tt_lists], dim=-1),
            )
        tt = torch.stack(fused_rows, dim=0)

        if self.transformer is not None and self.clusterer is not None:
            kb_items = self._predict_with_clustering(
                tt, vocabulary, threshold=threshold
            )
        else:
            raise TypeError(
                "Neither transformer/clusterer nor index is set. Call fit() first."
            )

        return {
            "entities": kb_items,
            "word_groupings": {
                wg: primary[wg]
                for wg in word_groupings
                if wg in primary.available_groupings()
            },
        }

    def _predict_with_clustering(
        self, embeddings: torch.Tensor, vocabulary: list, threshold: float = 0.0
    ):
        """
        Predict entities using clustering approach.

        Args:
            embeddings: Tensor of shape (n_mentions, embedding_dim)
            vocabulary: List of mention texts
            threshold: Minimum cluster membership probability required to return
                a prediction

        Returns:
            List of entity predictions
        """
        if self.transformer is None or self.clusterer is None:
            raise ValueError(
                "Transformer and clusterer must be fitted before prediction"
            )

        # Convert to numpy
        embeddings_np = embeddings.detach().cpu().numpy()

        # Transform embeddings
        umap_clustering, _ = self.transformer.transform(embeddings_np)

        # Predict clusters for input embeddings using approximate_predict
        cluster_labels, cluster_probs = approximate_predict(
            self.clusterer, umap_clustering
        )

        kb_items = []
        for item_dict, cluster_id, cluster_prob in zip(
            vocabulary, cluster_labels, cluster_probs
        ):
            # Skip HDBSCAN outliers and low-confidence assignments.
            if int(cluster_id) == -1 or float(cluster_prob) < threshold:
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
            score = float(cluster_prob)

            kb_item = {
                **item_dict,
                **{
                    "entity_id_predicted": predicted_entity,
                    "score": score,
                },
            }
            kb_items += [kb_item]

        return kb_items
