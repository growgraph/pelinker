"""Provider seam and response caching for the gold-pipeline LLM calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pelinker.eval import llm


def test_cache_key_is_stable_and_provider_sensitive() -> None:
    args = dict(model="m", system="s", user="u", json_mode=True)

    a = llm.cache_key(provider="gemini", **args)
    b = llm.cache_key(provider="gemini", **args)
    c = llm.cache_key(provider="anthropic", **args)

    assert a == b
    # Switching provider must not replay another provider's cached answer.
    assert a != c


def test_cache_key_separates_json_mode() -> None:
    args = dict(provider="gemini", model="m", system="s", user="u")

    assert llm.cache_key(json_mode=True, **args) != llm.cache_key(
        json_mode=False, **args
    )


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provider must be one of"):
        llm.complete(system="s", user="u", cache_dir=tmp_path, provider="oracle")


def test_cache_hit_returns_without_calling_the_provider(tmp_path: Path) -> None:
    key = llm.cache_key(
        provider="gemini",
        model=llm.DEFAULT_MODEL,
        system="sys",
        user="usr",
        json_mode=False,
    )
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"text": "cached answer", "provider": "gemini"}), encoding="utf-8"
    )

    # No credentials and no network here: a miss would raise, so returning proves the hit.
    out = llm.complete(system="sys", user="usr", cache_dir=tmp_path)

    assert out == "cached answer"


def test_cache_miss_writes_the_response(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_gemini(*, model, system, user, max_output_tokens, json_mode):
        captured.update(
            model=model, json_mode=json_mode, max_output_tokens=max_output_tokens
        )
        return llm.LLMResponse(
            text="fresh", model=model, provider="gemini", usage={"total_token_count": 7}
        )

    monkeypatch.setattr(llm, "_complete_gemini", fake_gemini)

    out = llm.complete(
        system="sys",
        user="usr",
        cache_dir=tmp_path,
        model="gemini-test",
        json_mode=True,
        max_output_tokens=123,
    )

    assert out == "fresh"
    assert captured == {
        "model": "gemini-test",
        "json_mode": True,
        "max_output_tokens": 123,
    }
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["text"] == "fresh"
    assert payload["provider"] == "gemini"
    assert payload["usage"]["total_token_count"] == 7


def test_second_call_replays_the_cache(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []

    def fake_gemini(*, model, system, user, max_output_tokens, json_mode):
        calls.append(1)
        return llm.LLMResponse(text="once", model=model, provider="gemini")

    monkeypatch.setattr(llm, "_complete_gemini", fake_gemini)

    first = llm.complete(system="s", user="u", cache_dir=tmp_path)
    second = llm.complete(system="s", user="u", cache_dir=tmp_path)

    assert (first, second) == ("once", "once")
    assert len(calls) == 1  # an annotation batch is re-runnable for free
