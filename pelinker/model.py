import faiss
from pelinker.util import texts_to_vrep
from pelinker.onto import WordGrouping
import torch
import joblib
import numpy as np
from typing import Optional
import hdbscan
from hdbscan import approximate_predict

from pelinker.transform import EmbeddingTransformer, TransformConfig


class Linker:
    def __init__(
        self,
        vocabulary: list[str],
        layers,
        nb_nn=10,
        index: faiss.IndexFlatIP | None = None,
        transformer: Optional[EmbeddingTransformer] = None,
        clusterer: Optional[hdbscan.HDBSCAN] = None,
        cluster_assignments: Optional[dict[str, int]] = None,
        transform_config: Optional[TransformConfig] = None,
        **kwargs,
    ):
        # Legacy FAISS index support (for backward compatibility)
        self.index: faiss.IndexFlatIP | None = index
        # New clustering-based approach
        self.transformer: Optional[EmbeddingTransformer] = transformer
        self.clusterer: Optional[hdbscan.HDBSCAN] = clusterer
        self.cluster_assignments: dict[str, int] = cluster_assignments or {}
        self.transform_config: Optional[TransformConfig] = transform_config

        self.vocabulary: list[str] = vocabulary
        self.labels_map: dict[str, str] = kwargs.pop("labels_map", dict())
        self.ls = layers
        self.nb_nn = nb_nn

    @classmethod
    def layers2str(cls, layers):
        if isinstance(layers, str):
            layers_str = layers
        else:
            if any(l0 > 0 for l0 in layers):
                raise ValueError(f" there are positive layers: {layers}")
            alayers = sorted([abs(l0) for l0 in layers])
            layers_str = "".join([str(l0) for l0 in alayers])
        return layers_str

    @classmethod
    def str2layers(cls, layers_spec):
        if "," in layers_spec:
            layers_spec = "".join(layers_spec.split(","))
        if layers_spec.isdigit():
            try:
                layers = list(set([-abs(int(x)) for x in layers_spec]))
            except:
                raise ValueError(f"{layers_spec} could not be parsed into layers")
        else:
            layers = layers_spec
        return layers

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
        embeddings: np.ndarray,
        transform_config: Optional[TransformConfig] = None,
        min_cluster_size: Optional[int] = None,
    ):
        """
        Fit the Linker model with embeddings.

        Args:
            embeddings: Array of shape (n_samples, n_features) containing KB embeddings
            transform_config: TransformConfig instance. If None, uses default.
            min_cluster_size: Minimum cluster size for HDBSCAN. If None, uses default optimization.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

        if len(embeddings) != len(self.vocabulary):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match vocabulary size ({len(self.vocabulary)})"
            )

        # Set transform config
        self.transform_config = transform_config or TransformConfig()

        # Fit transformer
        self.transformer = EmbeddingTransformer(self.transform_config)
        umap_clustering, _ = self.transformer.fit_transform(embeddings)

        # Fit clusterer
        if min_cluster_size is None:
            # Use default from analysis module logic
            min_cluster_size = 20

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

            if self.labels_map:
                kb_item["entity_label"] = (
                    self.labels_map[predicted_entity]
                    if predicted_entity in self.labels_map
                    else "NA"
                )
                if topk is not None and "_leading_candidates" in kb_item:
                    kb_item["_leading_candidates_labels"] = [
                        self.labels_map[e] if e in self.labels_map else "NA"
                        for e in kb_item["_leading_candidates"]
                    ]

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
