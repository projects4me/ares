"""
Docs "complete": continue the sentence the writer is typing.
"""

from app.adapters.base_ai_adapter import BaseAIAdapter
from app.schemas.docs import CompleteRequest, CompleteResponse
from app.services import docs_prompts
from app.services.ai_call import ask_model
from app.services.docs_settings import complete_settings
from app.services.model_output import strip_code_fence

MAX_COMPLETION_CHARS = 200


def trim_completion(text: str) -> str:
    """Keep the first line, cut to MAX_COMPLETION_CHARS at a word boundary where there is one."""
    lines = strip_code_fence(text).lstrip("\r\n").splitlines()
    text = lines[0] if lines else ""
    if len(text) > MAX_COMPLETION_CHARS:
        cut = text[:MAX_COMPLETION_CHARS]
        space = MAX_COMPLETION_CHARS if text[MAX_COMPLETION_CHARS] == " " else cut.rfind(" ")
        text = cut[:space] if space > 0 else cut
    return text.rstrip()


async def complete(adapter: BaseAIAdapter, request: CompleteRequest) -> CompleteResponse:
    """Return plain text continuing the sentence, possibly empty."""
    if not request.before.strip():
        return CompleteResponse(text="")
    reply = await ask_model(
        adapter,
        complete_settings(),
        docs_prompts.COMPLETE_SYSTEM,
        docs_prompts.complete_message(request.title, request.headingPath, request.before, request.after),
        allow_truncated=True,
    )
    return CompleteResponse(text=trim_completion(reply))
