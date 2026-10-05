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

    def fake_gemini(
        *, model, system, user, max_output_tokens, json_mode, reasoning_effort=None
    ):
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

    def fake_gemini(
        *, model, system, user, max_output_tokens, json_mode, reasoning_effort=None
    ):
        calls.append(1)
        return llm.LLMResponse(text="once", model=model, provider="gemini")

    monkeypatch.setattr(llm, "_complete_gemini", fake_gemini)

    first = llm.complete(system="s", user="u", cache_dir=tmp_path)
    second = llm.complete(system="s", user="u", cache_dir=tmp_path)

    assert (first, second) == ("once", "once")
    assert len(calls) == 1  # an annotation batch is re-runnable for free


def test_openai_is_a_selectable_provider(tmp_path: Path, monkeypatch) -> None:
    """The agreement slice needs a second model family, so the seam has three providers."""
    seen: dict[str, object] = {}

    def fake_openai(
        *, model, system, user, max_output_tokens, json_mode, reasoning_effort=None
    ):
        seen.update(model=model, system=system)
        return llm.LLMResponse(text="[]", model=model, provider="openai")

    monkeypatch.setattr(llm, "_complete_openai", fake_openai)

    out = llm.complete(
        system="sys",
        user="usr",
        cache_dir=tmp_path,
        provider="openai",
        model="a-model",
    )

    assert out == "[]"
    assert seen == {"model": "a-model", "system": "sys"}
    assert "openai" in llm.PROVIDERS


def test_each_provider_caches_separately(tmp_path: Path, monkeypatch) -> None:
    """One provider's answer must never be replayed for another's identical prompt."""
    monkeypatch.setattr(
        llm,
        "_complete_openai",
        lambda **kw: llm.LLMResponse(
            text="from openai", model=kw["model"], provider="openai"
        ),
    )
    monkeypatch.setattr(
        llm,
        "_complete_gemini",
        lambda **kw: llm.LLMResponse(
            text="from gemini", model=kw["model"], provider="gemini"
        ),
    )

    first = llm.complete(
        system="s", user="u", cache_dir=tmp_path, provider="openai", model="m"
    )
    second = llm.complete(
        system="s", user="u", cache_dir=tmp_path, provider="gemini", model="m"
    )

    assert (first, second) == ("from openai", "from gemini")


# ------------------------------------------------------------ truncation and effort


def test_cache_key_without_effort_matches_keys_written_before_it_existed() -> None:
    """Existing caches must keep replaying: an unset effort leaves the key unchanged."""
    key = llm.cache_key(
        provider="gemini", model="m", system="s", user="u", json_mode=True
    )

    assert key == "59f6ef5704f4363ec4b98f4c01f87966"
    assert key == llm.cache_key(
        provider="gemini",
        model="m",
        system="s",
        user="u",
        json_mode=True,
        reasoning_effort=None,
    )


def test_cache_key_separates_reasoning_effort() -> None:
    def key(effort: str | None) -> str:
        return llm.cache_key(
            provider="openai",
            model="m",
            system="s",
            user="u",
            json_mode=True,
            reasoning_effort=effort,
        )

    assert len({key("low"), key("high"), key(None)}) == 3


def test_reasoning_effort_reaches_the_provider(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_openai(**kw):
        seen["reasoning_effort"] = kw["reasoning_effort"]
        return llm.LLMResponse(text="[]", model=kw["model"], provider="openai")

    monkeypatch.setattr(llm, "_complete_openai", fake_openai)

    llm.complete(
        system="s",
        user="u",
        cache_dir=tmp_path,
        provider="openai",
        model="m",
        reasoning_effort="low",
    )

    assert seen == {"reasoning_effort": "low"}


def test_reasoning_effort_is_refused_where_the_provider_has_none(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        llm.complete(
            system="s",
            user="u",
            cache_dir=tmp_path,
            provider="gemini",
            reasoning_effort="low",
        )


def test_a_truncated_answer_is_raised_and_never_cached(
    tmp_path: Path, monkeypatch
) -> None:
    """Cached, a cut-off answer would replay as an empty document on every re-run."""

    def truncated(**kw):
        raise llm.LLMIncompleteError("stopped at max_output_tokens")

    monkeypatch.setattr(llm, "_complete_openai", truncated)

    with pytest.raises(llm.LLMIncompleteError):
        llm.complete(
            system="s", user="u", cache_dir=tmp_path, provider="openai", model="m"
        )

    assert list(tmp_path.glob("*.json")) == []


def test_an_empty_answer_is_raised_and_never_cached(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        llm,
        "_complete_gemini",
        lambda **kw: llm.LLMResponse(text="  ", model=kw["model"], provider="gemini"),
    )

    with pytest.raises(llm.LLMIncompleteError, match="empty"):
        llm.complete(system="s", user="u", cache_dir=tmp_path)

    assert list(tmp_path.glob("*.json")) == []
