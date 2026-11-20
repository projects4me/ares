import asyncio

import pytest

from app.adapters.base_embedding_adapter import BaseEmbeddingAdapter
from app.adapters.gaia_client import EmbeddingStored, RecordNotFound, SearchState, StaleHash
from app.adapters.gaia_errors import GaiaAuthError, GaiaUnavailable
from app.services.embedding_service import EmbeddingUnavailableError
from app.services.search_index_service import IndexOutcome, index_record, is_transient

ISSUE_ID = "1f6b0c3e-0000-4000-8000-000000000001"
DESIRED = "a" * 64
ACTUAL = "b" * 64


class FakeEmbedder(BaseEmbeddingAdapter):
    """Embeds every text as [1, 0, 0]; never loads a model."""

    def __init__(self):
        self.loaded = True
        self.error = None
        self.texts = []

    def model_name(self):
        return "fake"

    def dimensions(self):
        return 3

    def is_loaded(self):
        return self.loaded

    def load_error(self):
        return None

    async def load(self):
        pass

    async def embed(self, texts):
        self.texts.extend(texts)
        if self.error:
            raise self.error
        return [[2.0, 0.0, 0.0] for _ in texts]


class FakeGaia:
    """Answers get_search_state and store_embedding from canned results, recording each store."""

    def __init__(self, state, store_result=None):
        self.state = state
        self.store_result = store_result
        self.stored = []

    async def get_search_state(self, resource, record_id):
        if isinstance(self.state, Exception):
            raise self.state
        return self.state

    async def store_embedding(self, resource, record_id, embedding, text_hash):
        self.stored.append((resource, record_id, embedding, text_hash))
        return self.store_result or EmbeddingStored(record_id, text_hash)


def run(coro):
    return asyncio.run(coro)


def test_dirty_record_is_embedded_as_a_document_and_stored_with_the_hash_read():
    gaia = FakeGaia(SearchState(ISSUE_ID, "Login fails", DESIRED, ACTUAL))
    embedder = FakeEmbedder()

    outcome = run(index_record(gaia, embedder, "issue", ISSUE_ID))

    assert outcome is IndexOutcome.stored
    assert embedder.texts == ["search_document: Login fails"]
    assert gaia.stored == [("issue", ISSUE_ID, [1.0, 0.0, 0.0], DESIRED)]


def test_record_never_embedded_is_stored():
    gaia = FakeGaia(SearchState(ISSUE_ID, "Login fails", DESIRED, None))

    assert run(index_record(gaia, FakeEmbedder(), "issue", ISSUE_ID)) is IndexOutcome.stored
    assert gaia.stored[0][3] == DESIRED


def test_clean_record_does_nothing():
    gaia = FakeGaia(SearchState(ISSUE_ID, "Login fails", DESIRED, DESIRED))
    embedder = FakeEmbedder()

    assert run(index_record(gaia, embedder, "issue", ISSUE_ID)) is IndexOutcome.current
    assert embedder.texts == [] and gaia.stored == []


def test_stale_hash_is_dropped_without_a_retry():
    gaia = FakeGaia(SearchState(ISSUE_ID, "Login fails", DESIRED, ACTUAL), StaleHash("issue", ISSUE_ID, DESIRED))

    assert run(index_record(gaia, FakeEmbedder(), "issue", ISSUE_ID)) is IndexOutcome.stale
    assert len(gaia.stored) == 1


def test_missing_record_is_skipped():
    gaia = FakeGaia(RecordNotFound("issue", ISSUE_ID))
    embedder = FakeEmbedder()

    assert run(index_record(gaia, embedder, "issue", ISSUE_ID)) is IndexOutcome.not_found
    assert embedder.texts == [] and gaia.stored == []


def test_record_deleted_while_embedded_is_skipped():
    gaia = FakeGaia(SearchState(ISSUE_ID, "Login fails", DESIRED, None), RecordNotFound("issue", ISSUE_ID))

    assert run(index_record(gaia, FakeEmbedder(), "issue", ISSUE_ID)) is IndexOutcome.not_found


@pytest.mark.parametrize("text, desired", [("", DESIRED), (None, DESIRED), ("Login fails", None)])
def test_record_without_text_or_desired_hash_is_skipped(text, desired):
    gaia = FakeGaia(SearchState(ISSUE_ID, text, desired, None))
    embedder = FakeEmbedder()

    assert run(index_record(gaia, embedder, "issue", ISSUE_ID)) is IndexOutcome.no_text
    assert embedder.texts == [] and gaia.stored == []


def test_embedder_failure_stores_nothing():
    gaia = FakeGaia(SearchState(ISSUE_ID, "Login fails", DESIRED, None))
    embedder = FakeEmbedder()
    embedder.error = RuntimeError("onnx")

    with pytest.raises(EmbeddingUnavailableError):
        run(index_record(gaia, embedder, "issue", ISSUE_ID))
    assert gaia.stored == []


def test_unloaded_model_stores_nothing():
    gaia = FakeGaia(SearchState(ISSUE_ID, "Login fails", DESIRED, None))
    embedder = FakeEmbedder()
    embedder.loaded = False

    with pytest.raises(EmbeddingUnavailableError):
        run(index_record(gaia, embedder, "issue", ISSUE_ID))
    assert gaia.stored == []


def test_gaia_failure_propagates():
    gaia = FakeGaia(GaiaUnavailable("down"))

    with pytest.raises(GaiaUnavailable):
        run(index_record(gaia, FakeEmbedder(), "issue", ISSUE_ID))


@pytest.mark.parametrize("error, transient", [
    (GaiaUnavailable("down"), True),
    (EmbeddingUnavailableError("not loaded"), True),
    (GaiaAuthError("403", 403), False),
    (ValueError("bad resource"), False),
])
def test_only_gaia_outages_and_model_failures_are_transient(error, transient):
    assert is_transient(error) is transient
