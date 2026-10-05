from __future__ import annotations

import pathlib


"""Report filenames, schema identifiers and path builders."""

_JSON_CLUSTERING_REPORT_SCHEMA = "pelinker.clustering_report.v10"

MODEL_SELECTION_RUN_REPORT_SCHEMA = "pelinker.search.model_selection.run_report.v2"

# Basenames for artifacts under one report directory (``pelinker-fit`` / clustering search).
LINKER_FIT_CLUSTERING_REPORT_BASENAME = "linker_fit.clustering_report.json.gz"

LINKER_FIT_CLUSTER_COMPOSITION_BASENAME = "linker_fit.cluster_composition.json.gz"

LINKER_FIT_KB_OUT_BASENAME = "linker_fit.kb_out.json"

_FIT_CLUSTER_COMPOSITION_SCHEMA = "pelinker.fit_cluster_composition.v2"

_KB_OUT_SCHEMA = "pelinker.kb.kb_out.v1"

MODEL_SELECTION_RUN_REPORT_BASENAME = "model_selection.run_report.json.gz"

MODEL_SELECTION_SUMMARY_JSON_SCHEMA = "pelinker.search.model_selection.summary.v1"

MODEL_SELECTION_SUMMARY_JSON_BASENAME = "model_selection.summary.json"

CLUSTERING_SEARCH_GRID_PER_SAMPLE_CSV_BASENAME = "results_grid_per_sample.csv"

CLUSTERING_SEARCH_GRID_CHOSEN_JSON_BASENAME = "grid_chosen_hyperparameters.json"

CLUSTERING_SEARCH_FINE_METADATA_BASENAME = "fine_clustering_metadata.jsonl.gz"

FINE_SCREENER_EVAL_BASENAME = "fine_screener_eval.jsonl.gz"

CLUSTERING_SEARCH_SAMPLE_LABELS_BASENAME = "sample_cluster_labels.parquet"
"""Per-bootstrap cluster assignments, written only when label persistence is requested.

Needed to measure cluster *identity* stability across draws
(:mod:`pelinker.clustering.stability`); the grid CSVs carry metrics per draw but no
labels, so without this the reported cluster-count spread cannot be told apart from
wholesale membership churn."""

MODEL_SELECTION_CHECKPOINT_BASENAME = "model_selection.state.json.gz"


def linker_fit_clustering_report_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Filesystem path for the fit-time :class:`ClusteringReport` JSON under ``report_dir``."""
    return pathlib.Path(report_dir).expanduser() / LINKER_FIT_CLUSTERING_REPORT_BASENAME


def linker_fit_cluster_composition_path(
    report_dir: str | pathlib.Path,
) -> pathlib.Path:
    """Filesystem path for the entity-weighted cluster composition artifact."""
    return (
        pathlib.Path(report_dir).expanduser() / LINKER_FIT_CLUSTER_COMPOSITION_BASENAME
    )


def linker_fit_kb_out_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Filesystem path for the KB-out catalog JSON under ``report_dir``."""
    return pathlib.Path(report_dir).expanduser() / LINKER_FIT_KB_OUT_BASENAME


def model_selection_run_report_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Filesystem path for the standardized model-selection aggregate report."""
    return pathlib.Path(report_dir).expanduser() / MODEL_SELECTION_RUN_REPORT_BASENAME


def model_selection_summary_json_path(report_dir: str | pathlib.Path) -> pathlib.Path:
    """Top-level replot summary (rankings, best combos) as plain JSON."""
    return pathlib.Path(report_dir).expanduser() / MODEL_SELECTION_SUMMARY_JSON_BASENAME
