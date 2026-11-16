"""
Model settings for the Docs AI use cases, read from env.

Each use case has its own model and token limit so the frequent, latency
sensitive completion can run on a fast model. An unset model falls back to the
adapter's default model.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSettings:
    model: Optional[str]
    max_tokens: int


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name) or default)
    except ValueError:
        return default
    return value if value > 0 else default


def ask_settings() -> ModelSettings:
    return ModelSettings(os.getenv("ARES_DOCS_ASK_MODEL") or None, _int_env("ARES_DOCS_ASK_MAX_TOKENS", 4000))


def complete_settings() -> ModelSettings:
    return ModelSettings(
        os.getenv("ARES_DOCS_COMPLETE_MODEL") or "claude-haiku-4-5-20251001",
        _int_env("ARES_DOCS_COMPLETE_MAX_TOKENS", 80),
    )


def review_settings() -> ModelSettings:
    return ModelSettings(os.getenv("ARES_DOCS_REVIEW_MODEL") or None, _int_env("ARES_DOCS_REVIEW_MAX_TOKENS", 8000))
