"""
What the semantic search gateway returns: a record's search state, the outcome of storing
an embedding, and a page of the pending list. The services work on these; the Gaia
gateway (adapters.gaia_client) builds them.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SearchState:
    """What Gaia holds for semantic search: the text to embed, its desired hash and the actual hash last embedded."""
    id: str
    semantic_text: Optional[str]
    desired_hash: Optional[str]
    actual_hash: Optional[str]


@dataclass(frozen=True)
class EmbeddingStored:
    """Gaia stored the embedding for this hash."""
    id: str
    actual_hash: str


@dataclass(frozen=True)
class RecordNotFound:
    """Gaia has no such record, or it is deleted (404)."""
    resource: str
    id: str


@dataclass(frozen=True)
class StaleHash:
    """The record's text changed since the hash was read (409); nothing was stored."""
    resource: str
    id: str
    text_hash: str


@dataclass(frozen=True)
class PendingRecord:
    """A record whose embedding is missing or outdated: its search type, REST resource and id."""
    type: str
    resource: str
    id: str


@dataclass(frozen=True)
class PendingPage:
    """One page of the pending list; next_cursor is None on the last page."""
    records: List[PendingRecord]
    next_cursor: Optional[str]
