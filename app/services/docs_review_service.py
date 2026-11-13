"""
Docs "review": findings about a whole document, each tied to an exact quote from it.
"""

import json
from typing import Any, List

from pydantic import ValidationError

from app.adapters.base_ai_adapter import BaseAIAdapter
from app.schemas.docs import ReviewFinding, ReviewRequest, ReviewResponse
from app.services import docs_prompts
from app.services.ai_call import AIProviderError, ask_model
from app.services.docs_settings import review_settings
from app.services.model_output import strip_code_fence

MAX_FINDINGS = 20


def _parse_findings(reply: str) -> List[Any]:
    text = strip_code_fence(reply).strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1]) if start != -1 and end > start else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
        raise AIProviderError("AI provider returned a review that is not valid JSON")
    return data["findings"]


def keep_valid_findings(raw_findings: List[Any], text: str) -> List[ReviewFinding]:
    """Drop malformed findings and those whose quote is not in the text; keep at most MAX_FINDINGS."""
    findings: List[ReviewFinding] = []
    seen = set()
    for raw in raw_findings:
        try:
            finding = ReviewFinding.model_validate(raw)
        except ValidationError:
            continue
        key = (finding.quote, finding.category)
        if not finding.quote.strip() or finding.quote not in text or key in seen:
            continue
        seen.add(key)
        if finding.replacement == finding.quote:
            finding.replacement = None
        findings.append(finding)
        if len(findings) == MAX_FINDINGS:
            break
    return findings


async def review(adapter: BaseAIAdapter, request: ReviewRequest) -> ReviewResponse:
    """Return the review findings for the document."""
    reply = await ask_model(
        adapter,
        review_settings(),
        docs_prompts.REVIEW_SYSTEM,
        docs_prompts.review_message(request.title, request.text),
    )
    return ReviewResponse(findings=keep_valid_findings(_parse_findings(reply), request.text))
