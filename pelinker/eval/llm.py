"""Cached LLM calls for the gold pipeline, over a small provider seam.

Only the evaluation path talks to an LLM — the linker itself never does. Two callers use
this module: gold pre-annotation (``run/eval/annotate_llm.py``) and the LLM linking
baseline (:class:`pelinker.eval.baselines.LlmLinkerBaseline`).

Design points that matter for the measurement:

- **Responses are disk-cached** keyed on ``(provider, model, system, user, json_mode)``
  (plus ``reasoning_effort`` when one is set), so re-runs, parser fixes and re-scoring
  cost nothing and an annotation batch is reproducible from the cache alone.
- **Only complete answers are cached.** A response the provider cut off at the output
  limit raises :class:`LLMIncompleteError` instead. Cached, a truncated answer would
  replay forever as an unparsable document with no candidates, and raising the limit
  afterwards would not retry it. Reasoning models make this likely: their hidden
  reasoning counts against the same output limit as the answer.
- **Structured output is requested where the provider supports it** rather than parsed
  out of prose, which removes a whole class of silent annotation loss.
- **The provider is a parameter.** Gold annotated by one model and a baseline answered by
  the same model would be a circular comparison, so the two must be configured to differ —
  and the agreement slice is strongest when its second annotator is a different model
  family, not just a different checkpoint.

Providers and their credentials, all installed by the ``eval`` extra:

============ ====================================== ==================================
provider     credential                             structured output
============ ====================================== ==================================
``gemini``   ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` response mime type (default)
``anthropic`` ``ANTHROPIC_API_KEY``                 prompt-level, with prefix caching
``openai``   ``OPENAI_API_KEY``                     prompt-level
============ ====================================== ==================================

Provider-enforced JSON is used only where it fits the contract. The annotation prompt
answers with a top-level **array**, which the OpenAI and Anthropic JSON-object modes
cannot express, so those two are asked for JSON in the prompt and parsed leniently rather
than being handed a schema that would reshape the answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

PROVIDERS = ("gemini", "anthropic", "openai")

DEFAULT_PROVIDER = "gemini"

DEFAULT_MODEL = "gemini-3.1-flash-lite"
"""Any current model id may be passed instead; nothing here depends on this one."""

REASONING_PROVIDERS = ("openai",)
"""Providers whose calls take a ``reasoning_effort``."""


class LLMIncompleteError(RuntimeError):
    """The provider stopped before the answer was complete (output limit, empty reply).

    Never cached, so the same call is retried on the next run — typically with a larger
    ``max_output_tokens``.
    """


@dataclass(frozen=True)
class LLMResponse:
    """Model text plus whatever usage the provider reported."""

    text: str
    model: str
    provider: str
    usage: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "text": self.text,
            "usage": self.usage,
        }


def cache_key(
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    json_mode: bool,
    reasoning_effort: str | None = None,
) -> str:
    fields: list[object] = [provider, model, system, user, json_mode]
    # Appended only when set, so every key written before the parameter existed still
    # resolves to the same cached answer.
    if reasoning_effort is not None:
        fields.append(reasoning_effort)
    payload = json.dumps(fields, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _complete_gemini(
    *,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    json_mode: bool,
    reasoning_effort: str | None = None,
) -> LLMResponse:
    from google import genai
    from google.genai import types

    # Reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
    client = genai.Client()
    config: dict[str, Any] = {
        "system_instruction": system,
        "max_output_tokens": max_output_tokens,
        # Annotation and linking are extraction tasks, not generation: sampling noise
        # here shows up as annotator disagreement that no one can adjudicate.
        "temperature": 0.0,
    }
    if json_mode:
        config["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(**config),
    )
    candidates = response.candidates or []
    if candidates and candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
        raise LLMIncompleteError(
            f"gemini {model}: stopped at max_output_tokens={max_output_tokens}"
        )
    usage: dict[str, Any] = {}
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        for attr in (
            "prompt_token_count",
            "candidates_token_count",
            "cached_content_token_count",
            "total_token_count",
        ):
            value = getattr(meta, attr, None)
            if value is not None:
                usage[attr] = int(value)
    return LLMResponse(
        text=response.text or "",
        model=model,
        provider="gemini",
        usage=usage,
    )


def _complete_anthropic(
    *,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    json_mode: bool,
    reasoning_effort: str | None = None,
) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        system=[
            {
                "type": "text",
                "text": system,
                # The KB table is a large prefix shared by every request in a batch.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "max_tokens":
        raise LLMIncompleteError(
            f"anthropic {model}: stopped at max_tokens={max_output_tokens}"
        )
    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResponse(
        text=text,
        model=model,
        provider="anthropic",
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", None
            ),
        },
    )


def _complete_openai(
    *,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    json_mode: bool,
    reasoning_effort: str | None = None,
) -> LLMResponse:
    from openai import OpenAI

    # Reads OPENAI_API_KEY from the environment.
    client = OpenAI()
    # No temperature: the reasoning models reject the parameter outright, and the ones
    # that accept it are already near-deterministic on an extraction prompt.
    request: dict[str, Any] = {
        "model": model,
        "instructions": system,
        "input": user,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**request)
    if response.status != "completed":
        details = response.incomplete_details
        reason = details.reason if details is not None else None
        raise LLMIncompleteError(
            f"openai {model}: status={response.status} reason={reason} "
            f"(max_output_tokens={max_output_tokens}; reasoning tokens count against it)"
        )
    usage: dict[str, Any] = {}
    reported = getattr(response, "usage", None)
    if reported is not None:
        for attr in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(reported, attr, None)
            if value is not None:
                usage[attr] = int(value)
        details = getattr(reported, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            usage["cached_tokens"] = int(cached)
        out_details = getattr(reported, "output_tokens_details", None)
        reasoning = getattr(out_details, "reasoning_tokens", None)
        if reasoning is not None:
            usage["reasoning_tokens"] = int(reasoning)
    return LLMResponse(
        text=response.output_text or "",
        model=model,
        provider="openai",
        usage=usage,
    )


def _completion_for(provider: str) -> Callable[..., LLMResponse]:
    """The provider's call, resolved when it is needed.

    Looked up per call rather than bound in a module-level table, so the module attribute
    stays the single definition — a table built at import time would keep calling the
    original function after the attribute is replaced.
    """
    return {
        "gemini": _complete_gemini,
        "anthropic": _complete_anthropic,
        "openai": _complete_openai,
    }[provider]


def complete(
    *,
    system: str,
    user: str,
    cache_dir: pathlib.Path | str,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 8000,
    json_mode: bool = False,
    reasoning_effort: str | None = None,
) -> str:
    """Return the model's text, calling the provider only on a cache miss.

    Raises:
        LLMIncompleteError: the provider cut the answer off or returned nothing; the
            response is not cached.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")
    if reasoning_effort is not None and provider not in REASONING_PROVIDERS:
        raise ValueError(
            f"reasoning_effort is supported for {REASONING_PROVIDERS}, not {provider!r}"
        )

    directory = pathlib.Path(cache_dir)
    key = cache_key(
        provider=provider,
        model=model,
        system=system,
        user=user,
        json_mode=json_mode,
        reasoning_effort=reasoning_effort,
    )
    cache_file = directory / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

    response = _completion_for(provider)(
        model=model,
        system=system,
        user=user,
        max_output_tokens=max_output_tokens,
        json_mode=json_mode,
        reasoning_effort=reasoning_effort,
    )
    if not response.text.strip():
        raise LLMIncompleteError(f"{provider} {model}: empty response")

    directory.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(response.to_jsonable(), ensure_ascii=False), encoding="utf-8"
    )
    return response.text
