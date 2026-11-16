"""
One prompt sent through an AI adapter, with failures raised instead of returned.

The adapters report failures as {"success": False, "error": ...}; the use-case
services want either the model's text or an exception they can surface as an
HTTP error, never an empty or made-up result.
"""

from app.adapters.base_ai_adapter import BaseAIAdapter
from app.services.docs_settings import ModelSettings


class AIProviderError(Exception):
    """The AI provider failed, or returned output the service can't use."""


async def ask_model(adapter: BaseAIAdapter, settings: ModelSettings, system: str, user: str,
                    allow_truncated: bool = False) -> str:
    """Send one user message with a system prompt and return the model's text.

    A reply cut off by the token limit is a failure unless allow_truncated is set,
    so a half-written answer never reaches the writer.
    """
    result = await adapter.chat(
        [{"role": "user", "content": user}],
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=system,
    )
    if not result.get("success"):
        raise AIProviderError(result.get("error") or "AI provider error")
    if result.get("stop_reason") == "max_tokens" and not allow_truncated:
        raise AIProviderError("AI provider reply was cut off by the token limit")
    return result.get("response") or ""
