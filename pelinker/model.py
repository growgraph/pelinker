import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

import pandas as pd
import torch
import joblib
import numpy as np
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
from pelinker.analysis import (
    drop_properties_with_few_mentions,
    estimate_clustering_from_frame,
    filter_mention_frame_by_kb_labels,
    mention_frame_from_embedding_paths,
    metrics_df_with_grid_sample_columns,
)
from pelinker.plotting import plot_metrics_with_error_bars
from pelinker.reporting import (
    ClusteringReport,
    summarize_clustering_reports_for_search,
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


def _clustering_quality_model_layer_from_metadata(
    meta: EmbeddingModelMetadata | None,
) -> tuple[str, str]:
    """Labels aligned with ``run/analysis/clustering_quality.py`` search rows."""
    if meta is None or not meta.sources:
        return "linker", "unknown"
    if len(meta.sources) == 1:
        s = meta.sources[0]
        return s.model_type, s.layers_spec
    layer = "+".join(f"{s.model_type}/{s.layers_spec}" for s in meta.sources)
    return f"fusion{len(meta.sources)}", layer


def _safe_plot_stem(model: str, layer: str) -> str:
    safe_layer = layer.replace("/", "_").replace("+", "__")
    return f"{model}_{safe_layer}"


def _metrics_df_with_grid_sample_columns(
    report: ClusteringReport,
    *,
    model: str,
    layer: str,
    sample_idx: int,
    chosen_min_cluster_size: int | None = None,
) -> pd.DataFrame:
    return metrics_df_with_grid_sample_columns(
        report,
        model=model,
        layer=layer,
        sample_idx=sample_idx,
        chosen_min_cluster_size=chosen_min_cluster_size,
    )


def _fine_clustering_metadata_df(
    report: ClusteringReport,
    *,
    model: str,
    layer: str,
    sample_idx: int,
) -> pd.DataFrame:
    cols = ["model", "layer", "sample_idx", "property", "class"]
    optional_cols = ["pmid", "mention"]
    present_optional = [c for c in optional_cols if c in report.df.columns]
    keep = [
        c for c in ["property", "class", *present_optional] if c in report.df.columns
    ]
    if "property" not in keep or "class" not in keep:
        return pd.DataFrame(columns=cols + present_optional)
    out = report.df[keep].copy()
    out.insert(0, "sample_idx", sample_idx)
    out.insert(0, "layer", layer)
    out.insert(0, "model", model)
    return out


def _write_clustering_validation_artifacts(
    report_dir: pathlib.Path,
    report: ClusteringReport,
    *,
    model: str,
    layer: str,
    sample_idx: int = 0,
) -> None:
    """
    Persist grid metrics, summary row, metric plot, and fine metadata like
    ``run/analysis/clustering_quality.py`` (single sample).
    """
    report_dir = report_dir.expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_clustering_reports_for_search(
        [report], model=model, layer=layer
    )
    pd.DataFrame([summary.to_flat_dict()]).to_csv(
        report_dir / "results.csv", index=False
    )

    grid_detail = _metrics_df_with_grid_sample_columns(
        report, model=model, layer=layer, sample_idx=sample_idx
    )
    grid_detail.to_csv(report_dir / "results_grid_per_sample.csv", index=False)

    fine = _fine_clustering_metadata_df(
        report, model=model, layer=layer, sample_idx=sample_idx
    )
    fine_path = report_dir / "fine_clustering_metadata.pkl.gz"
    tmp_fine = fine_path.with_name(fine_path.name + ".tmp")
    fine.to_pickle(tmp_fine, compression="gzip")
    tmp_fine.replace(fine_path)

    plot_path = report_dir / f"{_safe_plot_stem(model, layer)}.png"
    plot_metrics_with_error_bars(
        [report.metrics_df],
        plot_path,
        chosen_min_cluster_size=float(report.hyperparameters.min_cluster_size),
    )
    logger.info(
        "Wrote clustering validation artifacts under %s (metrics plot: %s)",
        report_dir,
        plot_path.name,
    )


def _modal_cluster_deterministic(clusters: list[int]) -> int | None:
    """Most frequent cluster among ``clusters``, excluding HDBSCAN noise (-1); ties → smallest id."""
    vals = [int(c) for c in clusters if int(c) != -1]
    if not vals:
        return None
    cnt = Counter(vals)
    best_n = max(cnt.values())
    candidates = sorted(k for k, v in cnt.items() if v == best_n)
    return candidates[0]


def _provisional_cluster_assignments_from_training_frame(
    labels_map: dict[str, str],
    training: pd.DataFrame,
) -> dict[str, int]:
    """
    Map each ``entity_id`` to a single cluster id for ``predict`` compatibility.

    Heuristic: modal training cluster among rows whose ``property`` equals
    ``labels_map[entity_id]``, ignoring -1. Interpretation of clusters is otherwise
    left to downstream analysis (see ``Linker.training_cluster_frame``).
    """
    out: dict[str, int] = {}
    if "property" not in training.columns or "cluster" not in training.columns:
        return out
    for entity_id, label in labels_map.items():
        rows = training.loc[training["property"] == label, "cluster"]
        if len(rows) == 0:
            continue
        mode = _modal_cluster_deterministic(rows.astype(int).tolist())
        if mode is None:
            continue
        out[str(entity_id)] = int(mode)
    return out


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
        if "training_cluster_frame" not in pe_model.__dict__:
            pe_model.training_cluster_frame = None
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
        min_cluster_size: int | None = None,
        kb_labels: set[str] | None = None,
        optimize_clustering: bool = False,
        clustering_optimization_config: ClusteringOptimizationConfig | None = None,
        embedding_training: EmbeddingTrainingConfig | None = None,
        embedding_metadata: EmbeddingModelMetadata | None = None,
        kb_config: KBConfig | None = None,
        clustering_report_dir: pathlib.Path | None = None,
    ):
        """
        Fit the Linker model with embeddings.

        This method handles two main parts:
        a) Loading and processing embeddings (from file or direct array)
        b) Clustering the embeddings

        Args:
            embeddings: Path or sequence of paths to parquet file(s) (mention-level rows:
                        ``pmid``, ``property``, ``mention``, ``embed``). Multiple files are
                        fused like ``estimate_model_clustering`` (inner join on keys, concat
                        embeddings). Order must match ``embedding_metadata.sources``.
                        If None, ``embed_kb_corpus`` is run (one output file per source).
            transform_config: TransformConfig instance
            min_cluster_size: Minimum cluster size for HDBSCAN when ``optimize_clustering`` is
                False. If None, uses 20.
            kb_labels: Restrict training rows to these ``property`` labels (optional).
            optimize_clustering: If True, run ``min_cluster_size`` grid search on a
                ``clustering_optimization_config.frac`` subsample (analysis-aligned), then
                fit the serialized model on **all** prepared rows (no subsampling).
            clustering_optimization_config: Parquet batching (``batch_size``,
                ``n_embedding_batches``), grid bounds (``min_class_size``, ``max_scale``),
                grid subsample ``frac``, and RNG ``rns`` / ``frac`` for phase 1 only.
            embedding_training: Corpus paths and embedding runtime. Required when embeddings=None.
            embedding_metadata: If provided, overrides or sets ``self.embedding_metadata`` for
                this fit (required when embeddings=None unless already set on the linker).
            kb_config: Knowledge-base metadata stored on the linker; ``entity_count`` is set
                from fitted vocabulary when omitted (None).
            clustering_report_dir: If set, write clustering grid metrics, ``results.csv``,
                ``results_grid_per_sample.csv``, ``fine_clustering_metadata.pkl.gz``, and a
                DBCV/ICM plot (same layout as ``run/analysis/clustering_quality.py``). When
                ``optimize_clustering`` is False, a diagnostic grid is run on the same
                ``frac`` subsample as configured in ``clustering_optimization_config`` without
                changing the chosen ``min_cluster_size``.

        Returns:
            self
        """

        is_temporary = False
        embeddings_paths: list[pathlib.Path] = []

        try:
            if embedding_metadata is not None:
                self.embedding_metadata = embedding_metadata

            read_cfg = clustering_optimization_config or ClusteringOptimizationConfig()

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
                    "and columns pmid, property, mention, embed)."
                )
            filtered = filter_mention_frame_by_kb_labels(raw, kb_labels)
            if len(filtered) == 0:
                raise ValueError("No rows left after KB property-label filter")
            prepared = drop_properties_with_few_mentions(
                filtered, read_cfg.min_class_size
            )
            if len(prepared) == 0:
                raise ValueError(
                    "No rows left after dropping properties with fewer than "
                    f"{read_cfg.min_class_size} mentions each"
                )
            prepared = prepared.reset_index(drop=True)

            self._fit_clustering_on_prepared_mentions(
                prepared=prepared,
                transform_config=transform_config,
                read_cfg=read_cfg,
                optimize_clustering=optimize_clustering,
                min_cluster_size_arg=min_cluster_size,
                kb_config=kb_config,
                clustering_report_dir=clustering_report_dir,
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
        read_cfg: ClusteringOptimizationConfig,
        optimize_clustering: bool,
        min_cluster_size_arg: int | None,
        kb_config: KBConfig | None,
        clustering_report_dir: pathlib.Path | None,
    ) -> None:
        """Grid search (optional) on subsample; final PCA/UMAP + HDBSCAN on full ``prepared`` frame."""
        need_grid_frame = optimize_clustering or clustering_report_dir is not None
        grid_sample: pd.DataFrame | None = None
        if need_grid_frame:
            grid_sample = prepared.sample(
                frac=read_cfg.frac,
                random_state=read_cfg.rns,
                replace=False,
            )
            if len(grid_sample) == 0:
                logger.warning(
                    "Empty grid subsample (frac=%s); using full prepared frame for grid",
                    read_cfg.frac,
                )
                grid_sample = prepared

        report: ClusteringReport | None = None
        if optimize_clustering:
            assert grid_sample is not None
            report = estimate_clustering_from_frame(
                grid_sample,
                transform_config,
                optimization_config=read_cfg,
                selected_labels=None,
                all_metrics_dfs=None,
                aggregation_level="mention",
            )
            if report is None:
                logger.warning(
                    "Clustering grid failed; falling back to min_class_size=%s",
                    read_cfg.min_class_size,
                )
                best_mcs = int(read_cfg.min_class_size)
            else:
                best_mcs = int(report.hyperparameters.min_cluster_size)
                logger.info(
                    "Grid search selected min_cluster_size=%s (dbcv=%.4f)",
                    best_mcs,
                    report.best_score,
                )
        else:
            best_mcs = int(
                min_cluster_size_arg if min_cluster_size_arg is not None else 20
            )

        if clustering_report_dir is not None:
            diagnostic = report
            if not optimize_clustering:
                assert grid_sample is not None
                diagnostic = estimate_clustering_from_frame(
                    grid_sample,
                    transform_config,
                    optimization_config=read_cfg,
                    selected_labels=None,
                    all_metrics_dfs=None,
                    aggregation_level="mention",
                )
            if diagnostic is not None:
                m, lyr = _clustering_quality_model_layer_from_metadata(
                    self.embedding_metadata
                )
                _write_clustering_validation_artifacts(
                    clustering_report_dir,
                    diagnostic,
                    model=m,
                    layer=lyr,
                    sample_idx=0,
                )
            else:
                logger.warning(
                    "Skipping clustering validation artifacts (grid/diagnostic report is None)"
                )

        embeddings = np.stack(prepared["embed"].values).astype(np.float32, copy=False)
        self.transform_config = transform_config
        self.transformer = EmbeddingTransformer(transform_config)
        umap_clustering, _ = self.transformer.fit_transform(embeddings)

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=best_mcs,
            gen_min_span_tree=True,
            prediction_data=True,
        )
        cluster_labels_arr = self.clusterer.fit_predict(umap_clustering)
        cluster_labels = cluster_labels_arr.astype(int, copy=False)

        tc_cols = ["pmid", "property", "mention"]
        missing = [c for c in tc_cols if c not in prepared.columns]
        if missing:
            raise ValueError(
                "Prepared mention frame missing columns required for "
                f"training_cluster_frame: {missing}"
            )
        self.training_cluster_frame = prepared[tc_cols].copy()
        self.training_cluster_frame["cluster"] = cluster_labels

        self.cluster_assignments = _provisional_cluster_assignments_from_training_frame(
            self.labels_map,
            self.training_cluster_frame,
        )
        self.vocabulary = sorted(self.cluster_assignments.keys())
        if not self.vocabulary:
            raise ValueError(
                "No entity_ids received provisional cluster assignments after fit "
                "(check labels_map and training property labels)"
            )

        if kb_config is not None:
            if kb_config.entity_count is None:
                self.kb_config = replace(kb_config, entity_count=len(self.vocabulary))
            else:
                self.kb_config = kb_config

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
