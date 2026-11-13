"""
Request and response models for the Docs AI endpoints.

These models are the contract between Gaia (core/services/LLM.php) and the
/docs endpoints. The routers validate requests against them and the services
read from them, so they live outside both layers.
"""

from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, StringConstraints

TITLE_MAX = 500
HEADING_MAX = 500
HEADING_PATH_MAX = 20
BEFORE_MAX = 2000
AFTER_MAX = 500


class DocsContext(BaseModel):
    """Where the writer is in the document: its title, the headings above the cursor and the text around it."""

    title: str = Field("", max_length=TITLE_MAX)
    headingPath: List[Annotated[str, StringConstraints(max_length=HEADING_MAX)]] = Field(
        default_factory=list, max_length=HEADING_PATH_MAX)
    before: str = Field("", max_length=BEFORE_MAX)
    after: str = Field("", max_length=AFTER_MAX)


class AskRequest(BaseModel):
    """A writer's instruction about the selected HTML."""

    instruction: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    selectionHtml: str = Field("", max_length=8000)
    context: DocsContext = Field(default_factory=DocsContext)


class AskResponse(BaseModel):
    """The HTML that replaces the selection."""

    html: str


class CompleteRequest(DocsContext):
    """The text around the cursor of the sentence being written."""


class CompleteResponse(BaseModel):
    """Plain text continuing the sentence; may be empty."""

    text: str


class ReviewRequest(BaseModel):
    """A whole document as plain text, headings on their own lines."""

    title: str = Field("", max_length=TITLE_MAX)
    text: str = Field(..., min_length=1, max_length=60000)


class FindingCategory(str, Enum):
    unclear = "unclear"
    open_question = "open_question"
    inconsistency = "inconsistency"
    missing = "missing"
    wording = "wording"


class ReviewFinding(BaseModel):
    """One review finding; with a replacement it is a concrete fix for quote, without it a note."""

    quote: str = Field(..., min_length=1)
    category: FindingCategory
    message: str = Field(..., min_length=1)
    replacement: Optional[str] = None


class ReviewResponse(BaseModel):
    findings: List[ReviewFinding]
