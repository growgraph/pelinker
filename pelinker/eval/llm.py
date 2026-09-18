"""Cached LLM calls for the gold pipeline, over a small provider seam.

Only the evaluation path talks to an LLM — the linker itself never does. Two callers use
this module: gold pre-annotation (``run/eval/annotate_llm.py``) and the LLM linking
baseline (:class:`pelinker.eval.baselines.LlmLinkerBaseline`).

Design points that matter for the measurement:

- **Responses are disk-cached** keyed on ``(provider, model, system, user, json_mode)``,
  so re-runs, parser fixes and re-scoring cost nothing and an annotation batch is
  reproducible from the cache alone.
- **Structured output is requested where the provider supports it** rather than parsed
  out of prose, which removes a whole class of silent annotation loss.
- **The provider is a parameter.** Gold annotated by one model and a baseline answered by
  the same model would be a circular comparison, so the two must be configured to differ —
  and the agreement slice is strongest when its second annotator is a different model
  family, not just a different checkpoint.

``gemini`` is the default provider and needs the ``eval`` extra plus ``GEMINI_API_KEY``
(or ``GOOGLE_API_KEY``). ``anthropic`` is supported when that SDK is installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

PROVIDERS = ("gemini", "anthropic")

DEFAULT_PROVIDER = "gemini"

DEFAULT_MODEL = "gemini-3.1-flash-lite"
# Input $0.25 (text / image / video)
# Output price (including thinking tokens)	$1.5

"""Any current model id may be passed instead; nothing here depends on this one."""


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
    *, provider: str, model: str, system: str, user: str, json_mode: bool
) -> str:
    payload = json.dumps([provider, model, system, user, json_mode], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _complete_gemini(
    *, model: str, system: str, user: str, max_output_tokens: int, json_mode: bool
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
    *, model: str, system: str, user: str, max_output_tokens: int, json_mode: bool
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


def complete(
    *,
    system: str,
    user: str,
    cache_dir: pathlib.Path | str,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = 8000,
    json_mode: bool = False,
) -> str:
    """Return the model's text, calling the provider only on a cache miss."""
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")

    directory = pathlib.Path(cache_dir)
    key = cache_key(
        provider=provider, model=model, system=system, user=user, json_mode=json_mode
    )
    cache_file = directory / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

    if provider == "gemini":
        response = _complete_gemini(
            model=model,
            system=system,
            user=user,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
        )
    else:
        response = _complete_anthropic(
            model=model,
            system=system,
            user=user,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
        )

    directory.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(response.to_jsonable(), ensure_ascii=False), encoding="utf-8"
    )
    return response.text
