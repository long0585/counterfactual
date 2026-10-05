"""Configuration profiles for OpenAI and OpenAI-compatible model APIs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Mapping


ApiStyle = Literal["responses", "chat"]
ReasoningStyle = Literal["responses", "chat", "none"]


@dataclass(frozen=True)
class ProviderProfile:
    """Static, non-secret defaults for one API provider."""

    name: str
    api_style: ApiStyle
    api_key_env: str
    base_url_env: str
    default_base_url: str | None
    default_model: str | None
    reasoning_style: ReasoningStyle = "none"
    max_tokens_parameter: str = "max_tokens"
    fixed_sampling: bool = False
    default_extra_body: Mapping[str, object] | None = None
    supported_efforts: tuple[str, ...] | None = None


HF_API_KEY_ENV = "HF_TOKEN"
HF_BASE_URL_ENV = "HF_BASE_URL"
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
HF_DEFAULT_INFERENCE_PROVIDER = "deepinfra"


def _hf_profile(
    name: str,
    model_id: str,
    *,
    api_style: ApiStyle = "responses",
    reasoning_style: ReasoningStyle = "responses",
    supported_efforts: tuple[str, ...] | None = None,
) -> ProviderProfile:
    """Create one logical model profile routed through the shared HF gateway."""

    return ProviderProfile(
        name=name,
        api_style=api_style,
        api_key_env=HF_API_KEY_ENV,
        base_url_env=HF_BASE_URL_ENV,
        default_base_url=HF_ROUTER_BASE_URL,
        default_model=f"{model_id}:{HF_DEFAULT_INFERENCE_PROVIDER}",
        reasoning_style=reasoning_style,
        max_tokens_parameter="max_output_tokens" if api_style == "responses" else "max_tokens",
        fixed_sampling=True,
        supported_efforts=supported_efforts,
    )


PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "openai": ProviderProfile(
        name="openai",
        api_style="responses",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_base_url=None,
        default_model="gpt-4o",
        reasoning_style="responses",
        max_tokens_parameter="max_output_tokens",
        supported_efforts=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    ),
    "qwen": _hf_profile(
        "qwen",
        "Qwen/Qwen3.8-27B",
        supported_efforts=("low", "medium", "xhigh"),
    ),
    "glm": _hf_profile("glm", "zai-org/GLM-5.3"),
    "minimax": _hf_profile(
        "minimax",
        "MiniMaxAI/MiniMax-M3",
        reasoning_style="none",
    ),
    "kimi": _hf_profile(
        "kimi",
        "moonshotai/Kimi-K3",
        supported_efforts=("low", "high", "max"),
    ),
    "deepseek": _hf_profile(
        "deepseek",
        "deepseek-ai/DeepSeek-V4-Pro",
        api_style="chat",
        reasoning_style="chat",
        supported_efforts=("low", "high", "max"),
    ),
    "huggingface": _hf_profile("huggingface", "Qwen/Qwen3.8-27B"),
    "custom": ProviderProfile(
        name="custom",
        api_style="chat",
        api_key_env="MODEL_API_KEY",
        base_url_env="MODEL_BASE_URL",
        default_base_url=None,
        default_model=None,
        reasoning_style="none",
    ),
}


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider settings, including the secret read from the environment."""

    profile: ProviderProfile
    model: str
    api_key: str
    api_key_env: str
    base_url: str | None
    timeout: float = 120.0
    sdk_max_retries: int = 2

    @classmethod
    def from_env(
        cls,
        provider: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 120.0,
        sdk_max_retries: int = 2,
        environ: Mapping[str, str] | None = None,
        require_api_key: bool = True,
    ) -> "ProviderConfig":
        try:
            profile = PROVIDER_PROFILES[provider]
        except KeyError as exc:
            choices = ", ".join(sorted(PROVIDER_PROFILES))
            raise ValueError(f"Unknown provider {provider!r}. Choose one of: {choices}") from exc

        env = os.environ if environ is None else environ
        resolved_key_env = api_key_env or profile.api_key_env
        resolved_key = env.get(resolved_key_env, "")
        if require_api_key and not resolved_key:
            raise ValueError(
                f"Missing API key. Set environment variable {resolved_key_env} "
                f"for provider {provider!r}."
            )

        resolved_model = model or profile.default_model
        if not resolved_model:
            raise ValueError(f"Provider {provider!r} requires an explicit model name.")

        resolved_base_url = base_url or env.get(profile.base_url_env) or profile.default_base_url
        if provider == "custom" and not resolved_base_url:
            raise ValueError("Provider 'custom' requires --base-url or MODEL_BASE_URL.")

        return cls(
            profile=profile,
            model=resolved_model,
            api_key=resolved_key,
            api_key_env=resolved_key_env,
            base_url=resolved_base_url,
            timeout=timeout,
            sdk_max_retries=sdk_max_retries,
        )
