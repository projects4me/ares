"""
Small helpers that clean a model's raw text before a service parses it.
"""

import re

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Return the body of a reply wrapped in one Markdown code fence, else the reply unchanged."""
    match = _FENCE.match(text)
    return match.group(1) if match else text
