"""Tests for lemma-based KB training-entity index and prediction enrichment."""

from __future__ import annotations

import dataclasses

import torch

from pelinker.linker_kb_lemma import (
    enrich_entity_predictions_kb_validation,
    lookup_kb_training_entity_label,
)
from pelinker.model import Linker
from pelinker.onto import MentionCandidate, WordGrouping


def test_lookup_kb_training_entity_label() -> None:
    index = {WordGrouping.W1: {"a b": "LabelX"}}
    assert lookup_kb_training_entity_label(WordGrouping.W1, "a b", index) == "LabelX"
    assert lookup_kb_training_entity_label(WordGrouping.W1, "missing", index) is None
    assert lookup_kb_training_entity_label(None, "a b", index) is None
    assert lookup_kb_training_entity_label(WordGrouping.W1, "", index) is None


def test_enrich_entity_predictions_kb_validation() -> None:
    index = {WordGrouping.W1: {"tok": "Same"}}
    labels_map = {"id1": "Same", "id2": "Other"}
    rows: list[dict[str, object]] = [
        {
            "word_grouping": WordGrouping.W1,
            "lemma": "tok",
            "entity_id_predicted": "id1",
        },
        {
            "word_grouping": WordGrouping.W1,
            "lemma": "tok",
            "entity_id_predicted": "id2",
        },
        {
            "word_grouping": None,
            "lemma": "tok",
            "entity_id_predicted": "id1",
        },
    ]
    enrich_entity_predictions_kb_validation(rows, index, labels_map)
    assert rows[0]["kb_training_entity_from_lemma"] == "Same"
    assert rows[0]["kb_training_entity_for_prediction"] == "Same"
    assert rows[0]["lemma_kb_matches_predicted_entity"] is True

    assert rows[1]["kb_training_entity_from_lemma"] == "Same"
    assert rows[1]["kb_training_entity_for_prediction"] == "Other"
    assert rows[1]["lemma_kb_matches_predicted_entity"] is False

    assert rows[2]["kb_training_entity_from_lemma"] is None
    assert rows[2]["kb_training_entity_for_prediction"] == "Same"
    assert rows[2]["lemma_kb_matches_predicted_entity"] is False


class _PrimaryStub:
    texts: list[str] = [""]

    def available_groupings(self) -> list[WordGrouping]:
        return []

    def __getitem__(self, wg: WordGrouping) -> object:
        raise AssertionError("unreachable when available_groupings is empty")


def test_predict_include_prediction_kb_validation(monkeypatch) -> None:
    linker = Linker(labels_map={"e1": "Alpha Label", "e2": "Beta Label"})
    cand = MentionCandidate(
        mention="m",
        a=0,
        b=1,
        itext=0,
        a_abs=0,
        b_abs=1,
        word_grouping=WordGrouping.W1,
        lemma="alpha tok",
    )
    pred_row: dict[str, object] = {
        **dataclasses.asdict(cand),
        "entity_id_predicted": "e1",
        "score": 1.0,
        "pca_residual": 0.0,
        "pca_mahalanobis": 0.0,
        "anomaly_score_max_z": 0.0,
    }
    fake_index = {WordGrouping.W1: {"alpha tok": "Alpha Label"}}

    monkeypatch.setattr(
        linker,
        "_encode_mentions",
        lambda texts, max_length, use_gpu=False: (
            torch.zeros(1, 4, dtype=torch.float32),
            [cand],
            _PrimaryStub(),
        ),
    )
    monkeypatch.setattr(linker, "_ensure_nlp", lambda: object())
    monkeypatch.setattr(linker, "_kb_lemma_index_by_wg", lambda _nlp: fake_index)
    monkeypatch.setattr(
        linker,
        "_predict_with_clustering",
        lambda *args, **kwargs: ([pred_row], None),
    )

    out = linker.predict(["x"], include_prediction_kb_validation=True)
    ent = out.entities[0]
    assert ent["kb_training_entity"] == "Alpha Label"
    assert ent["kb_training_entity_from_lemma"] == "Alpha Label"
    assert ent["kb_training_entity_for_prediction"] == "Alpha Label"
    assert ent["lemma_kb_matches_predicted_entity"] is True
    assert "lemma" not in ent


def test_predict_merges_kb_validation_into_debug_mentions(monkeypatch) -> None:
    linker = Linker(labels_map={"e1": "Alpha Label"})
    cand = MentionCandidate(
        mention="m",
        a=0,
        b=1,
        itext=0,
        a_abs=0,
        b_abs=1,
        word_grouping=WordGrouping.W1,
        lemma="alpha tok",
    )
    pred_row: dict[str, object] = {
        **dataclasses.asdict(cand),
        "entity_id_predicted": "e1",
        "score": 1.0,
        "pca_residual": 0.0,
        "pca_mahalanobis": 0.0,
        "anomaly_score_max_z": 0.0,
        "mention_source_index": 0,
    }
    debug_row: dict[str, object] = {
        "mention": "m",
        "lemma": "alpha tok",
        "itext": 0,
        "screener_is_negative": False,
    }
    fake_index = {WordGrouping.W1: {"alpha tok": "Alpha Label"}}

    monkeypatch.setattr(
        linker,
        "_encode_mentions",
        lambda texts, max_length, use_gpu=False: (
            torch.zeros(1, 4, dtype=torch.float32),
            [cand],
            _PrimaryStub(),
        ),
    )
    monkeypatch.setattr(linker, "_ensure_nlp", lambda: object())
    monkeypatch.setattr(linker, "_kb_lemma_index_by_wg", lambda _nlp: fake_index)
    monkeypatch.setattr(
        linker,
        "_predict_with_clustering",
        lambda *args, **kwargs: ([pred_row], [debug_row]),
    )

    out = linker.predict(
        ["x"],
        include_prediction_kb_validation=True,
        include_debug_mentions=True,
    )
    assert out.debug_mentions is not None
    dm0 = out.debug_mentions[0]
    assert dm0["lemma_kb_matches_predicted_entity"] is True
    assert dm0["entity_id_predicted"] == "e1"
    assert dm0["kb_training_entity"] == "Alpha Label"


def test_predict_debug_mentions_length_matches_mention_list(monkeypatch) -> None:
    linker = Linker(labels_map={})
    c1 = MentionCandidate(
        mention="a",
        a=0,
        b=1,
        itext=0,
        a_abs=0,
        b_abs=1,
        word_grouping=WordGrouping.W1,
        lemma="x",
    )
    c2 = MentionCandidate(
        mention="b",
        a=0,
        b=1,
        itext=0,
        a_abs=0,
        b_abs=1,
        word_grouping=WordGrouping.W1,
        lemma="y",
    )
    mentions = [c1, c2]
    dbg = [{"row": 0}, {"row": 1}]
    monkeypatch.setattr(
        linker,
        "_encode_mentions",
        lambda texts, max_length, use_gpu=False: (
            torch.zeros(2, 4, dtype=torch.float32),
            mentions,
            _PrimaryStub(),
        ),
    )
    monkeypatch.setattr(linker, "_ensure_nlp", lambda: object())
    monkeypatch.setattr(linker, "_kb_lemma_index_by_wg", lambda _nlp: {})
    monkeypatch.setattr(
        linker,
        "_predict_with_clustering",
        lambda *args, **kwargs: ([], dbg),
    )
    out = linker.predict(["t"], include_debug_mentions=True)
    assert out.debug_mentions is not None
    assert len(out.debug_mentions) == len(mentions)
