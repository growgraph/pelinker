import json

import numpy as np
import pandas as pd
import torch
from click.testing import CliRunner

from pelinker.cli import link_files
from pelinker.model import Linker
from pelinker.negative_screener import NegativeClassScreener
from pelinker.onto import MentionCandidate, NEGATIVE_LABEL, WordGrouping
from pelinker.transform import (
    EmbeddingTransformer,
    TransformConfig,
    compute_transform_artifacts,
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
        TransformConfig(pca_components=4, umap_components=2, umap_viz_components=2)
    )
    transformer.fit(embeddings)

    umap_clustering, umap_visualization, pca_residuals, pca_mahalanobis = (
        transformer.transform(embeddings)
    )

    assert umap_clustering.shape == (12, 2)
    assert umap_visualization.shape == (12, 2)
    assert pca_residuals.shape == (12,)
    assert pca_mahalanobis.shape == (12,)
    assert np.all(pca_residuals >= 0.0)
    assert np.all(pca_mahalanobis >= 0.0)


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
            pca_components=3, umap_components=2, umap_viz_components=2
        ),
    )

    assert artifacts.pca_residuals.shape == (len(df),)
    assert artifacts.pca_mahalanobis.shape == (len(df),)
    assert (artifacts.pca_residuals >= 0.0).all()
    assert (artifacts.pca_mahalanobis >= 0.0).all()


class _DummyTransformer:
    class _UmapStub:
        n_components = 2

    umap = _UmapStub()
    umap_viz = _UmapStub()

    def transform(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = embeddings.shape[0]
        return (
            np.zeros((n, 2), dtype=np.float32),
            np.zeros((n, 2), dtype=np.float32),
            np.array([0.05, 2.5], dtype=np.float32),
            np.array([0.1, 4.0], dtype=np.float32),
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
        MentionCandidate(mention="m1", a=0, b=2),
        MentionCandidate(mention="m2", a=0, b=2),
    ]
    embeddings = torch.zeros((2, 4), dtype=torch.float32)

    out = linker._predict_with_clustering(
        embeddings,
        vocabulary,
        threshold=0.0,
    )

    assert len(out) == 2
    assert out[0]["entity_id_predicted"] == "e1"
    assert "pca_residual" in out[0]
    assert "pca_mahalanobis" in out[0]
    assert "anomaly_score_max_z" in out[0]


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
        MentionCandidate(mention="m1", a=0, b=2),
        MentionCandidate(mention="m2", a=0, b=2),
    ]
    embeddings = torch.zeros((2, 4), dtype=torch.float32)

    out = linker._predict_with_clustering(
        embeddings,
        vocabulary,
        threshold=0.95,
    )

    assert len(out) == 0


def test_link_files_cli_include_anomaly_metrics(monkeypatch, tmp_path) -> None:
    class _FakeLinker:
        def predict(
            self,
            texts,
            max_length=None,
            threshold=0.0,
            *,
            use_gpu=False,
        ):
            return {
                "entities": [
                    {
                        "mention": "x",
                        "score": 1.0,
                        "pca_residual": 0.2,
                        "pca_mahalanobis": 1.3,
                        "anomaly_score_max_z": 0.7,
                    }
                ],
                "word_groupings": {},
            }

    fake_linker = _FakeLinker()
    monkeypatch.setattr(link_files.Linker, "load", lambda _p: fake_linker)

    input_path = tmp_path / "input.txt"
    input_path.write_text("simple text", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        link_files.main,
        [
            "-m",
            "dummy-model",
            "--include-anomaly-metrics",
            str(input_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "pca_residual" in result.output


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
        ):
            return {
                "entities": [
                    {
                        "mention": "x",
                        "score": 1.0,
                        "pca_residual": 0.2,
                        "pca_mahalanobis": 1.3,
                        "anomaly_score_max_z": 0.7,
                    }
                ],
                "word_groupings": {},
            }

    fake_linker = _FakeLinker()
    monkeypatch.setattr(link_files.Linker, "load", lambda _p: fake_linker)

    input_path = tmp_path / "input.txt"
    input_path.write_text("simple text", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        link_files.main,
        [
            "-m",
            "dummy-model",
            str(input_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "pca_residual" not in result.output
