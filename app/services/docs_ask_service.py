"""
Docs "ask": answer a writer's instruction about the selected text with replacement HTML.
"""

from app.adapters.base_ai_adapter import BaseAIAdapter
from app.schemas.docs import AskRequest, AskResponse
from app.services import docs_prompts
from app.services.ai_call import AIProviderError, ask_model
from app.services.docs_settings import ask_settings
from app.services.html_sanitizer import sanitize_html
from app.services.model_output import strip_code_fence


async def ask(adapter: BaseAIAdapter, request: AskRequest) -> AskResponse:
    """Return the selection's replacement, reduced to the tags the editor accepts."""
    context = request.context
    reply = await ask_model(
        adapter,
        ask_settings(),
        docs_prompts.ASK_SYSTEM,
        docs_prompts.ask_message(request.instruction, request.selectionHtml, context.title,
                                 context.headingPath, context.before, context.after),
    )
    html = sanitize_html(strip_code_fence(reply).strip()).strip()
    if not html:
        # An empty replacement would silently delete the selection.
        raise AIProviderError("AI provider returned no content for the selection")
    return AskResponse(html=html)
