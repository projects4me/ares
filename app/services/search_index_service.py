"""
Semantic search indexing use case: bring one record's embedding up to date.

Every trigger (a Hermes event, a reconciled pending record) ends here. It reads the
text Gaia stores for the record and its hashes; when the embedding is current there
is nothing to do. Otherwise it embeds that text as a document and stores the vector
with the desired hash it read, so Gaia refuses it (409) when the text changed since.
Ares builds no text of its own. A 409 is dropped, not retried: the change that made it
stale announces itself, and the next reconcile lists the record if it is still pending.
"""

import enum
from typing import Protocol, Union

from app.adapters.base_embedding_adapter import BaseEmbeddingAdapter
from app.adapters.gaia_errors import GaiaUnavailable
from app.schemas.embeddings import EmbeddingTask
from app.schemas.search import EmbeddingStored, RecordNotFound, SearchState, StaleHash
from app.services.embedding_service import EmbeddingUnavailableError, embed


class SearchGateway(Protocol):
    """The Gaia calls the indexer needs (adapters.gaia_client.GaiaClient)."""

    async def get_search_state(self, resource: str, record_id: str) -> Union[SearchState, RecordNotFound]: ...

    async def store_embedding(self, resource: str, record_id: str, embedding,
                              text_hash: str) -> Union[EmbeddingStored, RecordNotFound, StaleHash]: ...


class IndexOutcome(str, enum.Enum):
    """What indexing one record did."""
    stored = "stored"
    current = "current"          # the embedding already matches the text
    no_text = "no_text"          # Gaia has no text or no desired hash for it yet
    not_found = "not_found"      # the record is gone (404)
    stale = "stale"              # the text changed while it was embedded (409); dropped


async def index_record(gaia: SearchGateway, embedder: BaseEmbeddingAdapter, resource: str,
                       record_id: str) -> IndexOutcome:
    """Embed the record's stored text and store it when its embedding is missing or outdated.

    Gaia failures (gaia_errors) and EmbeddingUnavailableError propagate; nothing is stored then.
    """
    state = await gaia.get_search_state(resource, record_id)
    if isinstance(state, RecordNotFound):
        return IndexOutcome.not_found
    if not state.semantic_text or not state.desired_hash:
        return IndexOutcome.no_text
    if state.desired_hash == state.actual_hash:
        return IndexOutcome.current

    embeddings = await embed(embedder, [state.semantic_text], EmbeddingTask.document)
    result = await gaia.store_embedding(resource, record_id, embeddings.vectors[0], state.desired_hash)
    if isinstance(result, StaleHash):
        return IndexOutcome.stale
    if isinstance(result, RecordNotFound):
        return IndexOutcome.not_found
    return IndexOutcome.stored


def is_transient(error: BaseException) -> bool:
    """Whether a failed index_record may well succeed soon: Gaia was unreachable or the model failed."""
    return isinstance(error, (GaiaUnavailable, EmbeddingUnavailableError))
