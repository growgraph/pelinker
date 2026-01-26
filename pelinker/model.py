import faiss
from pelinker.util import texts_to_vrep
from pelinker.onto import WordGrouping
import torch
import joblib
import numpy as np
from typing import Optional, Union
import hdbscan
from hdbscan import approximate_predict
import pathlib
import pyarrow.parquet as pq
import logging
from numpy.random import RandomState

from pelinker.transform import EmbeddingTransformer, TransformConfig

logger = logging.getLogger(__name__)


class Linker:
    def __init__(
        self,
        layers,
        nb_nn=10,
        index: faiss.IndexFlatIP | None = None,
        transformer: Optional[EmbeddingTransformer] = None,
        clusterer: Optional[hdbscan.HDBSCAN] = None,
        cluster_assignments: Optional[dict[str, int]] = None,
        transform_config: Optional[TransformConfig] = None,
        **kwargs,
    ):
        self.index: faiss.IndexFlatIP | None = index
        # New clustering-based approach
        self.transformer: Optional[EmbeddingTransformer] = transformer
        self.clusterer: Optional[hdbscan.HDBSCAN] = clusterer
        self.cluster_assignments: dict[str, int] = cluster_assignments or {}
        self.transform_config: Optional[TransformConfig] = transform_config

        self.vocabulary: list[str] = []
        self.labels_map: dict[str, str] = kwargs.pop("labels_map", dict())
        self.ls = layers
        self.nb_nn = nb_nn

    @classmethod
    def filter_report(cls, report, thr_score, thr_dif):
        report["entities"] = [
            r
            for r in report["entities"]
            if r["dif_to_next"] >= thr_dif and r["score"] >= thr_score
        ]
        return report

    def dump(self, file_spec):
        # Save FAISS index if it exists (legacy support)
        if self.index is not None:
            faiss.write_index(self.index, f"{file_spec}.index")
            index_backup = self.index
            self.index = None
        else:
            index_backup = None

        joblib.dump(self, f"{file_spec}.gz", compress=3)

        # Restore index after saving
        if index_backup is not None:
            self.index = index_backup

    @classmethod
    def load(cls, file_spec):
        pe_model = joblib.load(f"{file_spec}.gz")
        # Load FAISS index if it exists (legacy support)
        index_path = f"{file_spec}.index"
        try:
            import pathlib

            if pathlib.Path(index_path).exists():
                index = faiss.read_index(index_path)
                pe_model.index = index
        except Exception:
            # Index file doesn't exist or can't be loaded - that's OK for new models
            pass
        return pe_model

    def fit(
        self,
        embeddings: Union[np.ndarray, pathlib.Path, str],
        transform_config: TransformConfig,
        min_cluster_size: Optional[int] = None,
        kb_labels: Optional[set[str]] = None,
        optimize_clustering: bool = False,
        clustering_optimization_params: Optional[dict] = None,
    ):
        """
        Fit the Linker model with embeddings.

        This method handles two main parts:
        a) Loading and processing embeddings (from file or direct array)
        b) Clustering the embeddings

        Args:
            embeddings: Either:
                - Array of shape (n_samples, n_features) containing KB embeddings
                - Path to parquet file containing embeddings (with 'property' and 'embed' columns)
            transform_config: TransformConfig instance
            min_cluster_size: Minimum cluster size for HDBSCAN. If None and optimize_clustering=False,
                            uses default from transform_config or 20.
            kb_labels: Set of KB property labels to filter by (only used when embeddings is a file path)
            optimize_clustering: If True, optimize min_cluster_size using grid search
            clustering_optimization_params: Optional dict with optimization parameters:
                - min_class_size: Minimum class size for filtering (default: 20)
                - max_scale: Maximum value for grid evaluation (default: 120)
                - rns: RandomState for reproducibility (default: RandomState(seed=13))
                - frac: Fraction of dataset to sample (default: 1.0)
                - head: Number of batches to take (default: None)
                - batch_size: Batch size for reading (default: 1000)

        Returns:
            self
        """
        # Part a) Load and process embeddings
        # Check if embeddings is a file path (before loading)
        is_file_path = isinstance(embeddings, (str, pathlib.Path))

        if is_file_path:
            embeddings_path = pathlib.Path(embeddings)
            # Part b) Optimize clustering first if requested (before loading embeddings)
            if optimize_clustering:
                logger.info("Optimizing clustering parameters...")
                min_cluster_size = self._optimize_clustering(
                    embeddings_path,
                    transform_config,
                    kb_labels,
                    clustering_optimization_params,
                )

            # Load embeddings from file
            logger.info(f"Loading embeddings from file: {embeddings_path}")
            embeddings, entity_ids = self._load_embeddings_from_file(
                embeddings_path, kb_labels
            )
            # Update vocabulary to match loaded entity_ids
            self.vocabulary = entity_ids
        else:
            # Direct embeddings array provided
            if embeddings.ndim != 2:
                raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

            if len(embeddings) != len(self.vocabulary):
                raise ValueError(
                    f"Number of embeddings ({len(embeddings)}) must match vocabulary size ({len(self.vocabulary)})"
                )

            # Optimization not supported for direct embeddings array
            if optimize_clustering:
                logger.warning(
                    "Clustering optimization requires file path, using provided min_cluster_size"
                )

        # Set transform config
        self.transform_config = transform_config

        # Fit transformer
        self.transformer = EmbeddingTransformer(self.transform_config)
        umap_clustering, _ = self.transformer.fit_transform(embeddings)

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

    def _load_embeddings_from_file(
        self, embeddings_path: pathlib.Path, kb_labels: Optional[set[str]] = None
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
        kb_labels: Optional[set[str]],
        optimization_params: Optional[dict],
    ) -> int:
        """
        Optimize min_cluster_size using grid search.

        Args:
            embeddings_path: Path to parquet file
            transform_config: TransformConfig instance
            kb_labels: Set of KB property labels to filter by
            optimization_params: Optional dict with optimization parameters

        Returns:
            Optimal min_cluster_size value
        """
        from pelinker.analysis import estimate_model_clustering

        # Default optimization parameters
        params = {
            "min_class_size": 20,
            "max_scale": 120,
            "rns": RandomState(seed=13),
            "frac": 1.0,
            "head": None,
            "batch_size": 1000,
            "optimization_method": "mean",
        }
        if optimization_params:
            params.update(optimization_params)

        logger.info("Fitting clustering model using estimate_model_clustering...")

        clustering_report = estimate_model_clustering(
            file_path=embeddings_path,
            rns=params["rns"],
            transform_config=transform_config,
            min_class_size=params["min_class_size"],
            max_scale=params["max_scale"],
            frac=params["frac"],
            head=params["head"],
            batch_size=params["batch_size"],
            selected_labels=kb_labels,
            all_metrics_dfs=None,
            optimization_method=params["optimization_method"],
        )

        if clustering_report is None:
            logger.warning(
                "Clustering estimation failed, using default min_cluster_size"
            )
            return params["min_class_size"]

        best_min_cluster_size = clustering_report.best_size
        best_score = clustering_report.best_score
        logger.info(
            f"Optimal min_cluster_size: {best_min_cluster_size} (score: {best_score:.3f})"
        )

        return best_min_cluster_size

    def predict(
        self, texts, tokenizer, model, nlp, max_length, topk=None, extra_context=False
    ):
        """
        Predict entities for input texts.

        Uses clustering-based approach if transformer and clusterer are available,
        otherwise falls back to FAISS index (legacy mode).
        """
        report_batch = texts_to_vrep(
            texts,
            tokenizer,
            model,
            layers_spec=self.ls,
            word_modes=[WordGrouping.W1],
            nlp=nlp,
            max_length=max_length,
        )

        # Extract embeddings and vocabulary from ReportBatch
        wg_current = report_batch[WordGrouping.W1]

        tt_list = []
        vocabulary = []
        for expr_holder in wg_current.expression_data:
            # expr_holder is ExpressionHolder with tt (tensor) and expressions
            for expr, tt in zip(expr_holder.expressions, expr_holder.tt):
                tt_list.append(tt)
                # Create item dict with mention text and position info
                mention_text = ""
                if expr.itext is not None and expr.itext < len(report_batch.texts):
                    text = report_batch.texts[expr.itext]
                    if expr.ichunk is not None:
                        # Map chunk position to text position
                        offset = report_batch.chunk_mapper.map_chunk_to_text(
                            expr.itext, expr.ichunk
                        )
                        if expr.a is not None and expr.b is not None:
                            mention_text = text[offset + expr.a : offset + expr.b]
                    elif expr.a is not None and expr.b is not None:
                        mention_text = text[expr.a : expr.b]

                item_dict = {
                    "mention": mention_text,
                    "a": expr.a,
                    "b": expr.b,
                    "itext": expr.itext,
                    "ichunk": expr.ichunk,
                }
                vocabulary.append(item_dict)

        if not tt_list:
            # No mentions found
            report = {"entities": [], "word_groupings": {}}
            return report

        tt = torch.stack(tt_list)

        # Use clustering-based approach if available
        if self.transformer is not None and self.clusterer is not None:
            kb_items = self._predict_with_clustering(tt, vocabulary, topk=topk)
        elif self.index is not None:
            # Fall back to FAISS index (legacy mode)
            distance_matrix, nearest_neighbors_matrix = self.index.search(
                tt, self.nb_nn
            )
            kb_items = []
            for item, nn, d in zip(
                vocabulary, nearest_neighbors_matrix, distance_matrix
            ):
                item = self.complement_with_kb_data(item, nn, d, topk=topk)
                kb_items += [item]
        else:
            raise TypeError(
                "Neither transformer/clusterer nor index is set. Call fit() first."
            )

        # Convert to dict format for compatibility
        report = {
            "entities": kb_items,
            "word_groupings": {WordGrouping.W1: wg_current},
        }
        return report

    def _predict_with_clustering(
        self, embeddings: torch.Tensor, vocabulary: list, topk=None
    ):
        """
        Predict entities using clustering approach.

        Args:
            embeddings: Tensor of shape (n_mentions, embedding_dim)
            vocabulary: List of mention texts
            topk: Number of top candidates to return

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
            # Find all entities in the same cluster
            cluster_entities = [
                entity_id
                for entity_id, cid in self.cluster_assignments.items()
                if cid == cluster_id
            ]

            # If no entities in cluster (noise point), return empty
            if not cluster_entities:
                kb_item = {
                    **item_dict,
                    **{
                        "entity_id_predicted": None,
                        "score": float(cluster_prob),
                        "dif_to_next": 0.0,
                    },
                }
                kb_items += [kb_item]
                continue

            # For now, return the first entity in the cluster
            # TODO: Could implement similarity-based ranking within cluster
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

            if topk is not None and len(cluster_entities) > 1:
                kb_item["_leading_candidates"] = cluster_entities[1 : topk + 1]
                kb_item["_leading_scores"] = [score * 0.9] * min(
                    len(cluster_entities) - 1, topk
                )
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
