"""Runtime settings and chat-model construction."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "anthropic:claude-opus-5-5"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    """Everything that configures one agent session."""

    model: str = DEFAULT_MODEL
    effort: str = "high"
    workspace: Path = field(default_factory=lambda: Path("ml_workspace"))
    max_time_budget: int = 900
    python_timeout: int = 300
    auto_approve: bool = False
    max_model_calls: int = 80
    anthropic_fallbacks: bool = True

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Build settings from ``ML_AGENT_*`` environment variables, then apply overrides.

        Overrides whose value is ``None`` are ignored so CLI flags can be passed straight through.
        """
        settings = cls(
            model=os.environ.get("ML_AGENT_MODEL") or DEFAULT_MODEL,
            effort=os.environ.get("ML_AGENT_EFFORT") or "high",
            workspace=Path(os.environ.get("ML_AGENT_WORKSPACE") or "ml_workspace"),
            max_time_budget=_env_int("ML_AGENT_MAX_TIME_BUDGET", 900),
            python_timeout=_env_int("ML_AGENT_PYTHON_TIMEOUT", 300),
            auto_approve=_env_bool("ML_AGENT_AUTO_APPROVE", False),
            max_model_calls=_env_int("ML_AGENT_MAX_MODEL_CALLS", 80),
            anthropic_fallbacks=_env_bool("ML_AGENT_ANTHROPIC_FALLBACKS", True),
        )
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if "workspace" in overrides:
            overrides["workspace"] = Path(overrides["workspace"])
        settings = replace(settings, **overrides)
        if settings.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {settings.effort!r}")
        return settings

    @property
    def provider(self) -> str:
        provider, sep, _ = self.model.partition(":")
        return provider if sep else ""


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the LLM that drives the agent from a ``provider:model`` string.

    Claude models get adaptive thinking at the configured effort, streaming (agent turns can be
    long), and the server-side refusal fallback. Ollama models run fully on this machine.
    Any other provider string is handed to LangChain's ``init_chat_model`` unchanged.
    """
    if settings.provider == "anthropic":
        kwargs: dict[str, Any] = {
            "max_tokens": 64000,
            "streaming": True,
            "reasoning_effort": settings.effort,
        }
        if settings.anthropic_fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["model_kwargs"] = {"fallbacks": "default"}
        return init_chat_model(settings.model, **kwargs)
    if settings.provider == "ollama":
        # Ollama's default context window is too small for the tool schemas plus a working session.
        return init_chat_model(settings.model, num_ctx=32768, temperature=0)
    return init_chat_model(settings.model)
