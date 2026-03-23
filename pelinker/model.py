import tempfile
from pelinker.util import texts_to_vrep
from pelinker.onto import WordGrouping
import torch
import joblib
import numpy as np
from typing import Optional
import hdbscan
from hdbscan import approximate_predict
import pathlib
import pyarrow.parquet as pq
import logging

from pelinker.config import (
    ClusteringOptimizationConfig,
    EmbeddingModelMetadata,
    EmbeddingTrainingConfig,
    TransformConfig,
)
from pelinker.embedder import embed_kb_corpus
from pelinker.transform import EmbeddingTransformer
from pelinker.util import load_models

logger = logging.getLogger(__name__)


class Linker:
    def __init__(
        self,
        transformer: EmbeddingTransformer | None = None,
        clusterer: hdbscan.HDBSCAN | None = None,
        transform_config: TransformConfig | None = None,
        embedding_metadata: EmbeddingModelMetadata | None = None,
        **kwargs,
    ):
        self.transformer: EmbeddingTransformer | None = transformer
        self.clusterer: hdbscan.HDBSCAN | None = clusterer
        self.cluster_assignments: dict[str, int] = {}
        self.transform_config: TransformConfig | None = transform_config
        self.embedding_metadata: EmbeddingModelMetadata | None = embedding_metadata

        self.vocabulary: list[str] = []
        self.labels_map: dict[str, str] = kwargs.pop("labels_map", dict())
        self._hf_tokenizer = None
        self._hf_model = None

    @classmethod
    def filter_report(cls, report, thr_score, thr_dif):
        report["entities"] = [
            r
            for r in report["entities"]
            if r["dif_to_next"] >= thr_dif and r["score"] >= thr_score
        ]
        return report

    def dump(self, file_spec):
        joblib.dump(self, f"{file_spec}.gz", compress=3)

    @classmethod
    def load(cls, file_spec):
        pe_model = joblib.load(f"{file_spec}.gz")
        if "embedding_metadata" not in pe_model.__dict__:
            pe_model.embedding_metadata = None
        return pe_model

    def fit(
        self,
        embeddings: pathlib.Path | None,
        transform_config: TransformConfig,
        min_cluster_size: Optional[int] = None,
        kb_labels: Optional[set[str]] = None,
        optimize_clustering: bool = False,
        clustering_optimization_config: Optional[ClusteringOptimizationConfig] = None,
        embedding_training: EmbeddingTrainingConfig | None = None,
        embedding_metadata: EmbeddingModelMetadata | None = None,
    ):
        """
        Fit the Linker model with embeddings.

        This method handles two main parts:
        a) Loading and processing embeddings (from file or direct array)
        b) Clustering the embeddings

        Args:
            embeddings: Path to parquet file containing embeddings (with 'property' and
                        'embed' columns). If None, this method will run embedding first.
            transform_config: TransformConfig instance
            min_cluster_size: Minimum cluster size for HDBSCAN. If None and optimize_clustering=False,
                            uses default from transform_config or 20.
            kb_labels: Set of KB property labels to filter by (only used when embeddings is a file path)
            optimize_clustering: If True, optimize min_cluster_size using grid search
            clustering_optimization_config: Config object for clustering optimization.
            embedding_training: Corpus paths and embedding runtime. Required when embeddings=None.
            embedding_metadata: If provided, overrides or sets ``self.embedding_metadata`` for
                this fit (required when embeddings=None unless already set on the linker).

        Returns:
            self
        """

        is_temporary = False
        embeddings_path = None

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

                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                    output_parquet_path = pathlib.Path(f.name)

                logger.info("Step (a): Embedding corpus...")
                embed_kb_corpus(
                    metadata=self.embedding_metadata,
                    training=embedding_training,
                    output_parquet_path=output_parquet_path,
                )
                embeddings_path = output_parquet_path
                is_temporary = True
            else:
                embeddings_path = pathlib.Path(embeddings).expanduser()
                logger.info(
                    "Step (a): Skipping embedding, using provided embeddings: "
                    f"{embeddings_path}"
                )

            # Part b) Optimize clustering first if requested (before loading embeddings)
            if optimize_clustering:
                logger.info("Optimizing clustering parameters...")
                min_cluster_size = self._optimize_clustering(
                    embeddings_path,
                    transform_config,
                    kb_labels,
                    clustering_optimization_config,
                )

            # Load embeddings from file
            logger.info(f"Loading embeddings from file: {embeddings_path}")
            embeddings_data, entity_ids = self._load_embeddings_from_file(
                embeddings_path, kb_labels
            )
            # Update vocabulary to match loaded entity_ids
            self.vocabulary = entity_ids

            # Set transform config
            self.transform_config = transform_config

            # Fit transformer
            self.transformer = EmbeddingTransformer(self.transform_config)
            umap_clustering, _ = self.transformer.fit_transform(embeddings_data)

            # Use optimized or provided min_cluster_size
            if min_cluster_size is None:
                min_cluster_size = 20  # Default

            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                gen_min_span_tree=True,
                prediction_data=True,  # Enable prediction for new points
            )
            cluster_labels = self.clusterer.fit_predict(umap_clustering)

            # Store cluster assignments: entity_id -> cluster_id
            self.cluster_assignments = {
                entity_id: int(cluster_id)
                for entity_id, cluster_id in zip(self.vocabulary, cluster_labels)
            }

            return self
        finally:
            if is_temporary and embeddings_path is not None:
                try:
                    embeddings_path.unlink()
                    logger.debug(f"Removed temporary parquet file: {embeddings_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary parquet file: {e}")

    def _load_embeddings_from_file(
        self, embeddings_path: pathlib.Path, kb_labels: set[str] | None = None
    ) -> tuple[np.ndarray, list[str]]:
        """
        Load embeddings from parquet file, filter to KB entities, and aggregate per property.

        Args:
            embeddings_path: Path to parquet file with 'property' and 'embed' columns
            kb_labels: Optional set of KB property labels to filter by

        Returns:
            Tuple of (embeddings_array, entity_ids_list)
        """
        logger.info("Reading embedded corpus and filtering to KB entities...")
        parquet_table = pq.read_table(embeddings_path)
        df_parquet = parquet_table.to_pandas()

        # Filter to only KB properties if labels provided
        if kb_labels is not None:
            df_kb_filtered = df_parquet[df_parquet["property"].isin(kb_labels)].copy()
            logger.info(
                f"Filtered to {len(df_kb_filtered)} mentions from {len(df_parquet)} total mentions"
            )
        else:
            df_kb_filtered = df_parquet.copy()
            logger.info(f"Using all {len(df_kb_filtered)} mentions (no KB filter)")

        if len(df_kb_filtered) == 0:
            raise ValueError("No mentions found for KB properties in corpus")

        # Aggregate embeddings per property (average)
        logger.info("Aggregating embeddings per property...")
        # Convert embed lists to numpy arrays for aggregation
        df_kb_filtered["embed_array"] = df_kb_filtered["embed"].apply(np.array)

        # Group by property and average embeddings
        property_embeddings = {}
        for prop_label, group in df_kb_filtered.groupby("property"):
            embeddings_list = group["embed_array"].tolist()
            # Stack and average
            embeddings_array = np.stack(embeddings_list)
            avg_embedding = np.mean(embeddings_array, axis=0)
            property_embeddings[prop_label] = avg_embedding

        # Map property labels to entity_ids using labels_map
        entity_ids = []
        embeddings_list = []
        for prop_label in sorted(property_embeddings.keys()):
            # Find entity_id for this property label
            entity_id = None
            for eid, label in self.labels_map.items():
                if label == prop_label:
                    entity_id = eid
                    break

            if entity_id is not None:
                entity_ids.append(entity_id)
                embeddings_list.append(property_embeddings[prop_label])
            else:
                logger.warning(
                    f"Property label '{prop_label}' not found in labels_map, skipping"
                )

        if len(embeddings_list) == 0:
            raise ValueError("No valid embeddings after mapping to entity_ids")

        embeddings = np.stack(embeddings_list)

        logger.info(
            f"Embedded {len(embeddings)} KB properties into {embeddings.shape[1]}-dimensional vectors"
        )

        return embeddings, entity_ids

    def _optimize_clustering(
        self,
        embeddings_path: pathlib.Path,
        transform_config: TransformConfig,
        kb_labels: set[str] | None,
        optimization_config: Optional[ClusteringOptimizationConfig] = None,
    ) -> int:
        """
        Optimize min_cluster_size using grid search.

        Args:
            embeddings_path: Path to parquet file
            transform_config: TransformConfig instance
            kb_labels: Set of KB property labels to filter by

        Returns:
            Optimal min_cluster_size value
        """
        from pelinker.analysis import estimate_model_clustering

        effective_config = optimization_config or ClusteringOptimizationConfig()

        logger.info("Fitting clustering model using estimate_model_clustering...")

        clustering_report = estimate_model_clustering(
            file_path=embeddings_path,
            transform_config=transform_config,
            optimization_config=effective_config,
            selected_labels=kb_labels,
            all_metrics_dfs=None,
        )

        if clustering_report is None:
            logger.warning(
                "Clustering estimation failed, using default min_cluster_size"
            )
            return effective_config.min_class_size

        best_min_cluster_size = clustering_report.best_size
        best_score = clustering_report.best_score
        logger.info(
            f"Optimal min_cluster_size: {best_min_cluster_size} (score: {best_score:.3f})"
        )

        return best_min_cluster_size

    def _ensure_hf_models(self, *, use_gpu: bool = False) -> None:
        """Load HuggingFace tokenizer+encoder once, matching ``embed_kb_corpus`` / first metadata source."""
        if self.embedding_metadata is None:
            raise ValueError(
                "embedding_metadata is required for predict(); set it during fit() or on the Linker."
            )
        if self._hf_tokenizer is None or self._hf_model is None:
            primary = self.embedding_metadata.sources[0]
            if len(self.embedding_metadata.sources) > 1:
                logger.warning(
                    "Multiple embedding sources in metadata; only the first source is used "
                    "(concatenation / fusion not implemented yet)."
                )
            logger.info("Loading encoder for predict: %s", primary.model_type)
            self._hf_tokenizer, self._hf_model = load_models(
                primary.model_type, sentence=False
            )
        if use_gpu:
            if torch.cuda.is_available():
                self._hf_model.to("cuda")
            else:
                logger.warning("CUDA is not available; predict runs on CPU")

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

        """
        self._ensure_hf_models(use_gpu=use_gpu)
        primary = self.embedding_metadata.sources[0]
        report_batch = texts_to_vrep(
            texts,
            self._hf_tokenizer,
            self._hf_model,
            layers_spec=primary.layers_spec,
            word_modes=[WordGrouping.W1, WordGrouping.W2, WordGrouping.W3],
            nlp=nlp,
            max_length=max_length,
        )

        # Extract embeddings and mention metadata across all supported word groupings.
        word_groupings = [WordGrouping.W1, WordGrouping.W2, WordGrouping.W3]
        tt_list = []
        vocabulary = []
        for wg in word_groupings:
            if wg not in report_batch.available_groupings():
                continue

            expression_container = report_batch[wg]
            for expr_holder in expression_container.expression_data:
                # expr_holder is ExpressionHolder with tt (tensor) and expressions
                for expr, tt in zip(expr_holder.expressions, expr_holder.tt):
                    tt_list.append(tt)
                    # Create item dict with mention text and position info
                    mention_text = ""
                    if (
                        expr.itext is not None
                        and expr.itext < len(report_batch.texts)
                        and expr.a is not None
                        and expr.b is not None
                    ):
                        text = report_batch.texts[expr.itext]
                        if expr.ichunk is not None:
                            # Map chunk position to text position
                            offset = report_batch.chunk_mapper.map_chunk_to_text(
                                expr.itext, expr.ichunk
                            )
                            mention_text = text[offset + expr.a : offset + expr.b]
                        else:
                            mention_text = text[expr.a : expr.b]

                    item_dict = {
                        "mention": mention_text,
                        "a": expr.a,
                        "b": expr.b,
                        "itext": expr.itext,
                        "ichunk": expr.ichunk,
                        "word_grouping": wg,
                    }
                    vocabulary.append(item_dict)

        if not tt_list:
            # No mentions found
            report = {"entities": [], "word_groupings": {}}
            return report

        tt = torch.stack(tt_list)

        # Use clustering-based approach if available
        if self.transformer is not None and self.clusterer is not None:
            kb_items = self._predict_with_clustering(
                tt, vocabulary, threshold=threshold
            )
        else:
            raise TypeError(
                "Neither transformer/clusterer nor index is set. Call fit() first."
            )

        # Convert to dict format for compatibility
        report = {
            "entities": kb_items,
            "word_groupings": {
                wg: report_batch[wg]
                for wg in word_groupings
                if wg in report_batch.available_groupings()
            },
        }
        return report

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

            # Calculate dif_to_next (difference to next cluster probability)
            # For simplicity, use 0.0 if only one cluster
            dif_to_next = 0.0
            if len(cluster_entities) > 1:
                # Could compute similarity to other entities in cluster
                dif_to_next = 0.1  # Placeholder

            kb_item = {
                **item_dict,
                **{
                    "entity_id_predicted": predicted_entity,
                    "score": score,
                    "dif_to_next": dif_to_next,
                },
            }
            kb_items += [kb_item]

        return kb_items

    def complement_with_kb_data(self, item, nearest_neighbors, distance, topk):
        distance = distance.tolist()

        candidate_entity = [self.vocabulary[nnx] for nnx in nearest_neighbors]

        dif = round(distance[0] - distance[1], 5)
        item = {
            **item,
            **{
                "entity_id_predicted": candidate_entity[0],
                "score": round(distance[0], 4),
                "dif_to_next": dif,
            },
        }
        if topk is not None:
            item["_leading_candidates"] = candidate_entity[1:topk]
            item["_leading_scores"] = [round(x, 4) for x in distance[1:topk]]

        if self.labels_map:
            item["entity_label"] = (
                self.labels_map[candidate_entity[0]]
                if candidate_entity[0] in self.labels_map
                else "NA"
            )
            if topk is not None:
                item["_leading_candidates_labels"] = [
                    self.labels_map[e] if e in self.labels_map else "NA"
                    for e in candidate_entity[1:topk]
                ]

        return item
