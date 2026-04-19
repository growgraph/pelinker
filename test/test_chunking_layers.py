"""Tests for layer normalization and tokenizer-budget text splitting (no spaCy)."""

import pytest

from pelinker.util import normalize_layers_spec, split_text_into_token_budget


def test_normalize_layers_spec_digit_string() -> None:
    assert normalize_layers_spec("12") == [-2, -1]
    assert normalize_layers_spec("1") == [-1]


def test_normalize_layers_spec_comma_stripped() -> None:
    assert normalize_layers_spec("1,2") == [-2, -1]


def test_normalize_layers_spec_list() -> None:
    assert normalize_layers_spec([-1, -2]) == [-2, -1]


def test_normalize_layers_spec_positive_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        normalize_layers_spec([0, -1])


def test_normalize_layers_spec_sent_rejected() -> None:
    with pytest.raises(ValueError, match="sent"):
        normalize_layers_spec("sent")


def test_normalize_layers_spec_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        normalize_layers_spec([-1, -99], n_hidden_states=13)


def test_split_text_into_token_budget_roundtrip(tokenizer_model_scibert) -> None:
    tokenizer, _ = tokenizer_model_scibert
    text = "Hello world. " * 30
    chunks = split_text_into_token_budget(text, tokenizer, max_tokens=32)
    assert "".join(chunks) == text
    for c in chunks:
        n = len(tokenizer.encode(c, add_special_tokens=False))
        assert n <= 32


def test_split_text_into_token_budget_long(tokenizer_model_scibert) -> None:
    tokenizer, _ = tokenizer_model_scibert
    piece = "word " * 200
    text = (piece * 40).strip()
    chunks = split_text_into_token_budget(text, tokenizer, max_tokens=128)
    assert "".join(chunks) == text
    for c in chunks:
        assert len(tokenizer.encode(c, add_special_tokens=False)) <= 128
