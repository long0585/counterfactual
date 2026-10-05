"""One adapter for OpenAI and OpenAI-compatible Responses/Chat APIs."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import re
from typing import Any, Mapping

from .config import ProviderConfig


@dataclass(frozen=True)
class ModelResult:
    """Normalized model output while retaining the original SDK response."""

    text: str
    reasoning: str | None
    raw_response: Any


def safe_filename_component(value: str) -> str:
    """Make a provider/model identifier safe as one cross-platform path component."""

    cleaned = re.sub(r'[<>:"/\\|?*]+', "__", value.strip())
    return cleaned.rstrip(" .") or "model"


def result_model_label(provider: str, model: str, effort: str | None) -> str:
    """Keep historical OpenAI filenames and add effort for any reasoning run."""

    label = safe_filename_component(model)
    if effort and (provider != "openai" or model.startswith("gpt-5")):
        label += "_" + safe_filename_component(effort)
    return label


class ModelEngine:
    """Minimal provider-neutral wrapper around the OpenAI Python SDK."""

    def __init__(self, config: ProviderConfig, *, client: Any | None = None) -> None:
        self.config = config
        self.client = client if client is not None else self._create_client()

    def _create_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for live API calls. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.sdk_max_retries,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return OpenAI(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        effort: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = 0,
        top_p: float | None = 0,
        extra_body: Mapping[str, object] | None = None,
    ) -> ModelResult:
        request = self.build_request(
            prompt,
            effort=effort,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
        )
        if self.config.profile.api_style == "responses":
            raw_response = self.client.responses.create(**request)
            text = str(_get_field(raw_response, "output_text") or "")
            reasoning = _extract_responses_reasoning(raw_response)
        else:
            raw_response = self.client.chat.completions.create(**request)
            choices = _get_field(raw_response, "choices") or []
            if not choices:
                raise ValueError("Provider returned no chat completion choices.")
            message = _get_field(choices[0], "message")
            text = str(_get_field(message, "content") or "")
            reasoning = _extract_chat_reasoning(message)

        return ModelResult(text=text, reasoning=reasoning, raw_response=raw_response)

    def build_request(
        self,
        prompt: str,
        *,
        effort: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = 0,
        top_p: float | None = 0,
        extra_body: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Build a request without making a network call (also used by smoke tests)."""

        profile = self.config.profile
        request: dict[str, Any] = {"model": self.config.model}
        if effort and profile.reasoning_style == "none":
            raise ValueError(
                f"Provider {profile.name!r} has no documented --effort mapping. "
                "Omit --effort or use --extra-body-json with a provider-documented field."
            )
        if effort and profile.supported_efforts and effort not in profile.supported_efforts:
            choices = ", ".join(profile.supported_efforts)
            raise ValueError(
                f"Unsupported reasoning effort {effort!r} for provider {profile.name!r}. "
                f"Choose one of: {choices}."
            )
        if profile.api_style == "responses":
            request["input"] = [{"role": "user", "content": prompt}]
        else:
            request["messages"] = [{"role": "user", "content": prompt}]

        if effort and profile.reasoning_style == "responses":
            request["reasoning"] = {"effort": effort}
        elif effort and profile.reasoning_style == "chat":
            request["reasoning_effort"] = effort

        if max_output_tokens is not None:
            request[profile.max_tokens_parameter] = max_output_tokens

        # Reasoning/fixed-sampling models document fixed or ignored sampling values.
        is_openai_reasoning_model = profile.name == "openai" and self.config.model.startswith(
            ("gpt-5", "o1", "o3", "o4")
        )
        if not effort and not profile.fixed_sampling and not is_openai_reasoning_model:
            if temperature is not None:
                request["temperature"] = temperature
            if top_p is not None:
                request["top_p"] = top_p

        merged_extra = copy.deepcopy(dict(profile.default_extra_body or {}))
        if effort == "none" and "thinking" in merged_extra:
            merged_extra["thinking"] = {"type": "disabled"}
        _deep_update(merged_extra, dict(extra_body or {}))
        if merged_extra:
            request["extra_body"] = merged_extra
        return request


def _get_field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(name)
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, Mapping):
        return model_extra.get(name)
    return None


def _extract_chat_reasoning(message: Any) -> str | None:
    reasoning = _get_field(message, "reasoning_content")
    if reasoning:
        return str(reasoning)

    details = _get_field(message, "reasoning_details") or []
    texts: list[str] = []
    for detail in details:
        text = _get_field(detail, "text") or _get_field(detail, "content")
        if text:
            texts.append(str(text))
    return "\n".join(texts) or None


def _extract_responses_reasoning(response: Any) -> str | None:
    texts: list[str] = []
    for item in _get_field(response, "output") or []:
        if _get_field(item, "type") != "reasoning":
            continue
        for summary in _get_field(item, "summary") or []:
            text = _get_field(summary, "text")
            if text:
                texts.append(str(text))
    return "\n".join(texts) or None


def diagnose_empty_response(raw_response: Any) -> str:
    """Best-effort diagnostic suffix for why a call produced no final answer text.

    Empty output isn't always a transient network blip -- it can also be a
    model/provider consistently truncating before emitting an answer (e.g.
    hitting a token limit mid-reasoning). Surfacing status/finish_reason and
    the shape of the response here means the next failure log actually says
    *why* it was empty instead of just that it was, so a persistent,
    non-transient failure doesn't have to be diagnosed blind.
    """

    parts: list[str] = []

    status = _get_field(raw_response, "status")
    if status:
        parts.append(f"status={status}")

    incomplete = _get_field(raw_response, "incomplete_details")
    reason = _get_field(incomplete, "reason") if incomplete is not None else None
    if reason:
        parts.append(f"incomplete_reason={reason}")

    output = _get_field(raw_response, "output")
    if output is not None:
        types = [str(_get_field(item, "type")) for item in output]
        parts.append(f"output_item_types={types}")

    choices = _get_field(raw_response, "choices") or []
    if choices:
        finish_reason = _get_field(choices[0], "finish_reason")
        if finish_reason:
            parts.append(f"finish_reason={finish_reason}")

    return " (" + ", ".join(parts) + ")" if parts else ""


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], dict(value))
        else:
            target[key] = value
