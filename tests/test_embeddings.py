import asyncio
import math

import pytest
from fastapi.testclient import TestClient

from app.adapters.base_embedding_adapter import BaseEmbeddingAdapter
from app.adapters.fastembed_adapter import FastEmbedAdapter
from app.dependencies import get_embedding_adapter, get_embedding_adapter_or_none
from app.factories.ai_service_factory import AIServiceFactory
from app.factories.embedding_adapter_factory import EmbeddingAdapterFactory
from app.main import app
from tests.conftest import FakeAdapter


class FakeEmbeddingAdapter(BaseEmbeddingAdapter):
    """Records each embed call and returns un-normalised vectors derived from the text; never loads a model."""

    def __init__(self, dimensions=4):
        self.loaded = True
        self.error = None
        self.raises = None
        self.vectors = None
        self._dimensions = dimensions
        self.calls = []

    def model_name(self):
        return "nomic-embed-text-v1.5"

    def dimensions(self):
        return self._dimensions

    def is_loaded(self):
        return self.loaded

    def load_error(self):
        return self.error

    async def load(self):
        pass

    async def embed(self, texts):
        self.calls.append(texts)
        if self.raises:
            raise self.raises
        if self.vectors is not None:
            return self.vectors
        return [[float(len(text))] + [1.0] * (self._dimensions - 1) for text in texts]


@pytest.fixture
def embedder():
    return FakeEmbeddingAdapter()


@pytest.fixture
def client(embedder):
    app.dependency_overrides[get_embedding_adapter] = lambda: embedder
    app.dependency_overrides[get_embedding_adapter_or_none] = lambda: embedder
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_query_texts_get_the_query_prefix(client, embedder):
    response = client.post("/embeddings", json={"texts": ["open bugs"], "task": "query"})

    assert response.status_code == 200
    assert embedder.calls == [["search_query: open bugs"]]


def test_document_texts_get_the_document_prefix(client, embedder):
    client.post("/embeddings", json={"texts": ["a", "b"], "task": "document"})

    assert embedder.calls == [["search_document: a", "search_document: b"]]


def test_returns_one_normalised_vector_per_text_in_order(client):
    texts = ["x", "a much longer text", "mid text"]

    body = client.post("/embeddings", json={"texts": texts, "task": "document"}).json()

    assert body["model"] == "nomic-embed-text-v1.5"
    assert body["dimensions"] == 4
    assert len(body["embeddings"]) == 3
    for text, vector in zip(texts, body["embeddings"]):
        assert len(vector) == 4
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)
        raw = [len("search_document: " + text), 1.0, 1.0, 1.0]
        norm = math.sqrt(sum(v * v for v in raw))
        assert vector == pytest.approx([v / norm for v in raw])


@pytest.mark.parametrize("payload", [
    {"texts": [], "task": "query"},
    {"texts": [""], "task": "query"},
    {"texts": ["x" * 8001], "task": "query"},
    {"texts": ["x"] * 65, "task": "query"},
    {"texts": ["x"], "task": "summary"},
    {"texts": ["x"]},
    {"task": "query"},
    {"texts": "x", "task": "query"},
])
def test_rejects_invalid_requests_with_422(client, embedder, payload):
    response = client.post("/embeddings", json=payload)

    assert response.status_code == 422
    assert embedder.calls == []


def test_accepts_the_limits(client):
    response = client.post("/embeddings", json={"texts": ["x" * 8000] * 64, "task": "document"})

    assert response.status_code == 200
    assert len(response.json()["embeddings"]) == 64


def test_model_still_loading_is_503(client, embedder):
    embedder.loaded = False

    response = client.post("/embeddings", json={"texts": ["x"], "task": "query"})

    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"]
    assert embedder.calls == []


def test_model_that_failed_to_load_is_503_with_the_reason(client, embedder):
    embedder.loaded = False
    embedder.error = "OSError: weights not found"

    response = client.post("/embeddings", json={"texts": ["x"], "task": "query"})

    assert response.status_code == 503
    assert "weights not found" in response.json()["detail"]


def test_inference_failure_is_503(client, embedder):
    embedder.raises = RuntimeError("onnx exploded")

    response = client.post("/embeddings", json={"texts": ["x"], "task": "query"})

    assert response.status_code == 503
    assert "failed" in response.json()["detail"]


@pytest.mark.parametrize("vectors", [
    [],
    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    [[1.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0, 0.0]],
    [[float("nan"), 0.0, 0.0, 0.0]],
])
def test_unusable_model_output_is_503_never_a_vector(client, embedder, vectors):
    embedder.vectors = vectors

    response = client.post("/embeddings", json={"texts": ["x"], "task": "query"})

    assert response.status_code == 503
    assert "embeddings" not in response.json()


def test_health_reports_the_embedding_model(client, embedder, monkeypatch):
    embedder.loaded = False
    embedder.error = "OSError: weights not found"
    monkeypatch.setattr(AIServiceFactory, "create_ai_service", classmethod(lambda cls, provider=None: FakeAdapter()))

    body = client.get("/").json()

    assert body["embedding_health"] == {
        "loaded": False, "model": "nomic-embed-text-v1.5", "error": "OSError: weights not found"}


def test_health_answers_when_the_embedding_provider_is_broken(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "nope")
    monkeypatch.setattr("app.dependencies._embedding_adapter", None)
    monkeypatch.setattr(AIServiceFactory, "create_ai_service", classmethod(lambda cls, provider=None: FakeAdapter()))

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.json()["embedding_health"]["loaded"] is False


def test_startup_loads_the_model_in_the_background(monkeypatch):
    embedder = FakeEmbeddingAdapter()
    embedder.loaded = False

    async def load():
        embedder.loaded = True

    embedder.load = load
    monkeypatch.setattr("app.dependencies._embedding_adapter", embedder)

    with TestClient(app) as client:
        response = client.post("/embeddings", json={"texts": ["x"], "task": "query"})

    assert response.status_code == 200
    assert embedder.loaded


def test_fastembed_adapter_falls_back_to_a_download_when_the_cache_misses():
    calls = []

    def text_embedding(model_id, **kwargs):
        calls.append(kwargs.get("local_files_only", False))
        if kwargs.get("local_files_only"):
            raise ValueError("not cached")
        return "model"

    assert FastEmbedAdapter()._create(text_embedding) == "model"
    assert calls == [True, False]


def test_factory_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        EmbeddingAdapterFactory.create_embedding_adapter("nope")


def test_fastembed_adapter_keeps_the_load_failure_and_stays_unloaded(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "acme/not-a-model")
    adapter = FastEmbedAdapter()

    asyncio.run(adapter.load())

    assert adapter.model_name() == "not-a-model"
    assert not adapter.is_loaded()
    assert adapter.load_error()
