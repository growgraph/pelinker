import json

import numpy as np
import pandas as pd
import torch
from click.testing import CliRunner

from pelinker.cli import link_files
from pelinker.model import Linker, LinkerPredictResult
from pelinker.screener.ambient_screener import NegativeClassScreener
from pelinker.onto import MentionCandidate, NEGATIVE_LABEL, WordGrouping
from pelinker.transform import (
    EmbeddingTransformer,
    TransformConfig,
    compute_transform_artifacts,
    score_transform_artifacts,
)


def test_link_files_sanitize_for_json_word_grouping_keys() -> None:
    payload = {"word_groupings": {WordGrouping.W1: {"k": "v"}}}
    safe = link_files._sanitize_for_json(payload)
    json.dumps(safe)
    assert safe["word_groupings"]["W1"]["k"] == "v"


def test_load_documents_from_file_supports_json_wrappers(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"content": "alpha"},
                    {"text": "beta", "ground_truth": [{"a": 0, "b": 1}]},
                ]
            }
        ),
        encoding="utf-8",
    )

    docs = link_files._load_documents_from_file(input_path)
    assert docs == [("alpha", None), ("beta", [{"a": 0, "b": 1}])]


def test_load_documents_from_file_supports_jsonl(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        '{"text":"alpha"}\n{"body":"beta","ground_truth":[{"a":0,"b":4}]}\n',
        encoding="utf-8",
    )

    docs = link_files._load_documents_from_file(input_path)
    assert docs == [("alpha", None), ("beta", [{"a": 0, "b": 4}])]


def test_load_documents_from_file_jsonl_falls_back_to_plain_text_on_invalid_line(
    tmp_path,
) -> None:
    input_path = tmp_path / "broken.jsonl"
    raw = '{"text":"alpha"}\nnot-json\n'
    input_path.write_text(raw, encoding="utf-8")

    docs = link_files._load_documents_from_file(input_path)
    assert docs == [(raw, None)]


def test_load_documents_from_file_falls_back_for_unsupported_json_shape(
    tmp_path,
) -> None:
    input_path = tmp_path / "scalar.json"
    raw = '"hello"'
    input_path.write_text(raw, encoding="utf-8")

    docs = link_files._load_documents_from_file(input_path)
    assert docs == [(raw, None)]


def test_embedding_transformer_transform_returns_dual_pca_metrics() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(12, 8)).astype(np.float32)
    transformer = EmbeddingTransformer(
        TransformConfig(pca_components=4, umap_components=2, cluster_viz_components=2)
    )
    transformer.fit(embeddings)

    (
        umap_clustering,
        cluster_viz,
        pca_residuals,
        pca_mahalanobis,
        pca_spectral_entropy,
    ) = transformer.transform(embeddings)

    assert umap_clustering.shape == (12, 2)
    assert cluster_viz.shape == (12, 2)
    assert pca_residuals.shape == (12,)
    assert pca_mahalanobis.shape == (12,)
    assert pca_spectral_entropy.shape == (12,)
    assert np.all(pca_residuals >= 0.0)
    assert np.all(pca_mahalanobis >= 0.0)
    assert np.all(pca_spectral_entropy >= 0.0)


def test_compute_transform_artifacts_exposes_pca_metric_arrays() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "embed": [row for row in rng.normal(size=(10, 6)).astype(np.float32)],
        }
    )

    artifacts = compute_transform_artifacts(
        df,
        config=TransformConfig(
            pca_components=3, umap_components=2, cluster_viz_components=2
        ),
    )

    assert artifacts.pca_residuals.shape == (len(df),)
    assert artifacts.pca_mahalanobis.shape == (len(df),)
    assert artifacts.pca_spectral_entropy.shape == (len(df),)
    assert (artifacts.pca_residuals >= 0.0).all()
    assert (artifacts.pca_mahalanobis >= 0.0).all()
    assert (artifacts.pca_spectral_entropy >= 0.0).all()


def test_score_transform_artifacts_matches_compute_pca_metrics() -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "embed": [row for row in rng.normal(size=(10, 6)).astype(np.float32)],
        }
    )
    cfg = TransformConfig(pca_components=3, umap_components=2, cluster_viz_components=2)
    fitted = compute_transform_artifacts(df, config=cfg)
    emb = np.stack(df["embed"].values).astype(np.float32, copy=False)
    transformer = EmbeddingTransformer(cfg).fit(emb)
    scored = score_transform_artifacts(df, transformer, include_umap=True)

    np.testing.assert_allclose(scored.pca_residuals, fitted.pca_residuals)
    np.testing.assert_allclose(scored.pca_mahalanobis, fitted.pca_mahalanobis)
    np.testing.assert_allclose(scored.pca_spectral_entropy, fitted.pca_spectral_entropy)


def test_score_transform_artifacts_is_stable_for_same_transformer() -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {"embed": [row for row in rng.normal(size=(10, 6)).astype(np.float32)]}
    )
    cfg = TransformConfig(pca_components=3, umap_components=2, cluster_viz_components=2)
    emb = np.stack(df["embed"].values).astype(np.float32, copy=False)
    transformer = EmbeddingTransformer(cfg).fit(emb)
    a = score_transform_artifacts(df, transformer, include_umap=True)
    b = score_transform_artifacts(df, transformer, include_umap=True)

    np.testing.assert_allclose(a.pca_residuals, b.pca_residuals)
    np.testing.assert_allclose(a.pca_mahalanobis, b.pca_mahalanobis)
    np.testing.assert_allclose(a.pca_spectral_entropy, b.pca_spectral_entropy)


def test_score_transform_artifacts_scores_extra_rows_without_umap() -> None:
    rng = np.random.default_rng(3)
    fit_df = pd.DataFrame(
        {"embed": [row for row in rng.normal(size=(8, 5)).astype(np.float32)]}
    )
    score_df = pd.DataFrame(
        {"embed": [row for row in rng.normal(size=(12, 5)).astype(np.float32)]}
    )
    cfg = TransformConfig(pca_components=3, umap_components=2, cluster_viz_components=2)
    emb_fit = np.stack(fit_df["embed"].values).astype(np.float32, copy=False)
    transformer = EmbeddingTransformer(cfg).fit(emb_fit)
    scored = score_transform_artifacts(score_df, transformer, include_umap=False)

    assert scored.pca_residuals.shape == (12,)
    assert scored.umap_clustering.shape == (12, 0)
    assert scored.cluster_viz.shape == (12, 0)


class _DummyTransformer:
    class _VizStub:
        n_components = 2

    umap = _VizStub()
    cluster_viz_pca = _VizStub()
    config = TransformConfig(
        pca_components=2, umap_components=2, cluster_viz_components=2
    )

    def transform(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = embeddings.shape[0]
        return (
            np.zeros((n, 2), dtype=np.float32),
            np.zeros((n, 2), dtype=np.float32),
            np.array([0.05, 2.5], dtype=np.float32),
            np.array([0.1, 4.0], dtype=np.float32),
            np.array([0.2, 0.3], dtype=np.float32),
        )


def test_predict_with_clustering_adds_anomaly_metrics(monkeypatch) -> None:
    linker = Linker()
    linker.transformer = _DummyTransformer()
    linker.clusterer = object()
    linker.cluster_assignments = {"e1": 0}
    linker.screener = NegativeClassScreener(
        kind="lda", negative_label=NEGATIVE_LABEL, _estimator=None
    )

    def _mock_approximate_predict(_clusterer, _umap_clustering):
        return np.array([0, 0]), np.array([0.9, 0.9])

    monkeypatch.setattr("pelinker.model.approximate_predict", _mock_approximate_predict)
    vocabulary = [
        MentionCandidate(mention="m1", a=0, b=2, itext=0, a_abs=0, b_abs=2),
        MentionCandidate(mention="m2", a=0, b=2, itext=0, a_abs=0, b_abs=2),
    ]
    embeddings = torch.zeros((2, 4), dtype=torch.float32)

    out, _anomaly = linker._predict_with_clustering(
        embeddings,
        vocabulary,
        threshold=0.0,
    )

    assert len(out) == 1
    assert out[0]["mention"] == "m1"
    assert out[0]["entity_id_predicted"] == "e1"
    assert "score" in out[0]
    assert isinstance(out[0]["score"], float)
    assert out[0]["score"] == 0.9
    assert "pca_residual" in out[0]
    assert "pca_mahalanobis" in out[0]
    assert "pca_spectral_entropy" in out[0]
    assert "projection_score" in out[0]
    assert "anomaly_score_max_z" in out[0]
    assert np.isnan(float(out[0]["projection_score"]))


def test_predict_with_clustering_respects_cluster_probability_threshold(
    monkeypatch,
) -> None:
    linker = Linker()
    linker.transformer = _DummyTransformer()
    linker.clusterer = object()
    linker.cluster_assignments = {"e1": 0}
    linker.screener = NegativeClassScreener(
        kind="lda", negative_label=NEGATIVE_LABEL, _estimator=None
    )

    def _mock_approximate_predict(_clusterer, _umap_clustering):
        return np.array([0, 0]), np.array([0.9, 0.9])

    monkeypatch.setattr("pelinker.model.approximate_predict", _mock_approximate_predict)
    vocabulary = [
        MentionCandidate(mention="m1", a=0, b=2, itext=0, a_abs=0, b_abs=2),
        MentionCandidate(mention="m2", a=0, b=2, itext=0, a_abs=0, b_abs=2),
    ]
    embeddings = torch.zeros((2, 4), dtype=torch.float32)

    out, _anomaly = linker._predict_with_clustering(
        embeddings,
        vocabulary,
        threshold=0.95,
    )

    assert len(out) == 0


def test_predict_with_clustering_keeps_disjoint_spans(monkeypatch) -> None:
    linker = Linker()
    linker.transformer = _DummyTransformer()
    linker.clusterer = object()
    linker.cluster_assignments = {"e1": 0}
    linker.screener = NegativeClassScreener(
        kind="lda", negative_label=NEGATIVE_LABEL, _estimator=None
    )

    def _mock_approximate_predict(_clusterer, _umap_clustering):
        return np.array([0, 0]), np.array([0.9, 0.85])

    monkeypatch.setattr("pelinker.model.approximate_predict", _mock_approximate_predict)
    vocabulary = [
        MentionCandidate(mention="left", a=0, b=4, itext=0, a_abs=0, b_abs=4),
        MentionCandidate(mention="right", a=0, b=5, itext=0, a_abs=10, b_abs=15),
    ]
    embeddings = torch.zeros((2, 4), dtype=torch.float32)

    out, _anomaly = linker._predict_with_clustering(
        embeddings,
        vocabulary,
        threshold=0.0,
    )

    assert len(out) == 2
    mentions = {str(r["mention"]) for r in out}
    assert mentions == {"left", "right"}


def test_predict_with_clustering_overlap_prefers_shorter_span_on_score_tie(
    monkeypatch,
) -> None:
    linker = Linker()
    linker.transformer = _DummyTransformer()
    linker.clusterer = object()
    linker.cluster_assignments = {"e1": 0}
    linker.screener = NegativeClassScreener(
        kind="lda", negative_label=NEGATIVE_LABEL, _estimator=None
    )

    def _mock_approximate_predict(_clusterer, _umap_clustering):
        return np.array([0, 0]), np.array([0.9, 0.9])

    monkeypatch.setattr("pelinker.model.approximate_predict", _mock_approximate_predict)
    vocabulary = [
        MentionCandidate(mention="longwin", a=0, b=20, itext=0, a_abs=0, b_abs=20),
        MentionCandidate(mention="short", a=0, b=8, itext=0, a_abs=6, b_abs=14),
    ]
    embeddings = torch.zeros((2, 4), dtype=torch.float32)

    out, _anomaly = linker._predict_with_clustering(
        embeddings,
        vocabulary,
        threshold=0.0,
    )

    assert len(out) == 1
    assert out[0]["mention"] == "short"
    assert "score" in out[0]
    assert isinstance(out[0]["score"], float)
    assert out[0]["score"] == 0.9


def test_predict_with_clustering_overlap_prefers_higher_score_over_shorter_span(
    monkeypatch,
) -> None:
    linker = Linker()
    linker.transformer = _DummyTransformer()
    linker.clusterer = object()
    linker.cluster_assignments = {"e1": 0}
    linker.screener = NegativeClassScreener(
        kind="lda", negative_label=NEGATIVE_LABEL, _estimator=None
    )

    def _mock_approximate_predict(_clusterer, _umap_clustering):
        return np.array([0, 0]), np.array([0.95, 0.5])

    monkeypatch.setattr("pelinker.model.approximate_predict", _mock_approximate_predict)
    vocabulary = [
        MentionCandidate(mention="longwin", a=0, b=20, itext=0, a_abs=0, b_abs=20),
        MentionCandidate(mention="short", a=0, b=8, itext=0, a_abs=6, b_abs=14),
    ]
    embeddings = torch.zeros((2, 4), dtype=torch.float32)

    out, _anomaly = linker._predict_with_clustering(
        embeddings,
        vocabulary,
        threshold=0.0,
    )

    assert len(out) == 1
    assert out[0]["mention"] == "longwin"
    assert out[0]["score"] == 0.95


def test_filter_report_keeps_entities_by_score_threshold() -> None:
    report: dict[str, object] = {
        "entities": [
            {"mention": "hi", "score": 0.8, "entity_id_predicted": "e1"},
            {"mention": "lo", "score": 0.2, "entity_id_predicted": "e2"},
        ],
        "word_groupings": {},
    }
    out = Linker.filter_report(report, thr_score=0.5)
    raw_entities = report["entities"]
    assert isinstance(raw_entities, list)
    assert len(raw_entities) == 2
    entities = out["entities"]
    assert isinstance(entities, list)
    assert len(entities) == 1
    assert entities[0]["mention"] == "hi"
    assert entities[0]["score"] == 0.8


def test_link_files_cli_include_anomaly_metrics(monkeypatch, tmp_path) -> None:
    class _FakeLinker:
        def predict(
            self,
            texts,
            max_length=None,
            threshold=0.0,
            *,
            use_gpu=False,
            include_mention_anomaly=False,
            include_prediction_kb_validation=False,
            **kwargs: object,
        ):
            entities = [
                {
                    "mention": "x",
                    "score": 1.0,
                    "pca_residual": 0.2,
                    "pca_mahalanobis": 1.3,
                    "anomaly_score_max_z": 0.7,
                }
            ]
            debug = (
                [{"mention": "x", "screener_is_negative": False}]
                if include_mention_anomaly
                else None
            )
            return LinkerPredictResult(entities=entities, debug_mentions=debug)

    fake_linker = _FakeLinker()
    monkeypatch.setattr(link_files.Linker, "load", lambda _p: fake_linker)

    input_path = tmp_path / "input.txt"
    input_path.write_text("simple text", encoding="utf-8")
    report_path = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        link_files.main,
        [
            "-m",
            "dummy-model",
            "-o",
            str(report_path),
            "--include-anomaly-metrics",
            str(input_path),
        ],
    )

    assert result.exit_code == 0, result.output
    written = report_path.read_text(encoding="utf-8")
    assert "pca_residual" in written


def test_link_files_cli_excludes_anomaly_metrics_by_default(
    monkeypatch, tmp_path
) -> None:
    class _FakeLinker:
        def predict(
            self,
            texts,
            max_length=None,
            threshold=0.0,
            *,
            use_gpu=False,
            include_mention_anomaly=False,
            include_prediction_kb_validation=False,
            **kwargs: object,
        ):
            entities = [
                {
                    "mention": "x",
                    "score": 1.0,
                    "pca_residual": 0.2,
                    "pca_mahalanobis": 1.3,
                    "anomaly_score_max_z": 0.7,
                }
            ]
            debug = [] if include_mention_anomaly else None
            return LinkerPredictResult(entities=entities, debug_mentions=debug)

    fake_linker = _FakeLinker()
    monkeypatch.setattr(link_files.Linker, "load", lambda _p: fake_linker)

    input_path = tmp_path / "input.txt"
    input_path.write_text("simple text", encoding="utf-8")
    report_path = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        link_files.main,
        [
            "-m",
            "dummy-model",
            "-o",
            str(report_path),
            str(input_path),
        ],
    )

    assert result.exit_code == 0, result.output
    written = report_path.read_text(encoding="utf-8")
    assert "pca_residual" not in written
