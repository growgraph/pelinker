"""Tests for replot summary JSON (model_selection.summary.json)."""

from __future__ import annotations

import pathlib

import pandas as pd

from run.analysis import model_selection as ms


def test_build_model_selection_summary_payload_rankings() -> None:
    df = pd.DataFrame(
        [
            {
                "combo_key": "1:a/l1",
                "model": "a",
                "layer": "l1",
                "best_score": 0.5,
                "best_score_std": 0.01,
                "best_size": 10.0,
                "best_size_std": 0.0,
                "screener_auc_mean": 0.7,
                "screener_auc_std": 0.02,
                "screener_lda_auc_mean": 0.68,
                "screener_svm_auc_mean": 0.69,
                "combined_auc_mean": 0.75,
                "combined_auc_std": 0.01,
                "oov_auc_mean": 0.72,
                "oov_auc_std": 0.01,
                "screener_best_kind": "lda",
            },
            {
                "combo_key": "1:b/l2",
                "model": "b",
                "layer": "l2",
                "best_score": 0.9,
                "best_score_std": 0.01,
                "best_size": 12.0,
                "best_size_std": 0.0,
                "screener_auc_mean": 0.85,
                "screener_auc_std": 0.01,
                "screener_lda_auc_mean": 0.84,
                "screener_svm_auc_mean": 0.83,
                "combined_auc_mean": 0.8,
                "combined_auc_std": 0.01,
                "oov_auc_mean": 0.78,
                "oov_auc_std": 0.01,
                "screener_best_kind": "svm",
            },
            {
                "combo_key": "1:c/l3",
                "model": "c",
                "layer": "l3",
                "best_score": 0.6,
                "best_score_std": 0.01,
                "best_size": 11.0,
                "best_size_std": 0.0,
                "screener_auc_mean": 0.82,
                "screener_auc_std": 0.01,
                "screener_lda_auc_mean": 0.81,
                "screener_svm_auc_mean": 0.8,
                "combined_auc_mean": 0.88,
                "combined_auc_std": 0.01,
                "oov_auc_mean": 0.86,
                "oov_auc_std": 0.01,
                "screener_best_kind": "lda",
            },
        ]
    )
    payload = ms.build_model_selection_summary_payload(
        pathlib.Path("/tmp/report"),
        df,
        chosen_by_combo=(("a", "l1", 10),),
        generated_at="2020-01-01T00:00:00Z",
    )
    assert payload["schema"] == "pelinker.model_selection.summary.v1"
    assert payload["generated_at"] == "2020-01-01T00:00:00Z"
    top_scr = payload["rankings"]["top_by_screener_auc"]
    assert len(top_scr) == 3
    assert top_scr[0]["model"] == "b"
    assert top_scr[0]["rank"] == 1
    assert top_scr[0]["rank_metric"] == "screener_auc_mean"
    top_comb = payload["rankings"]["top_by_combined_auc"]
    assert top_comb[0]["model"] == "c"
    assert payload["rankings"]["best_by_dbcv"]["model"] == "b"
    assert payload["best_combined_auc"]["model"] == "c"
    assert (
        payload["grid"]["chosen_min_cluster_size_by_combo"][0]["min_cluster_size"] == 10
    )
