import asyncio
import json
import logging
from urllib.parse import parse_qs

import httpx
import pytest

from app.adapters.gaia_auth import GaiaAuth
from app.adapters.gaia_client import EmbeddingStored, GaiaClient, RecordNotFound, SearchState, StaleHash
from app.adapters.gaia_errors import (
    GaiaAuthError, GaiaConfigError, GaiaRequestRejected, GaiaUnavailable)

BASE = "https://gaia.test"
SECRET = "s3cret-value"
ISSUE_ID = "1f6b0c3e-0000-4000-8000-000000000001"
HASH = "a" * 64


class FakeGaia:
    """A mocked Gaia: answers the token grant and the issue routes from canned responses, recording every request."""

    def __init__(self):
        self.requests = []
        self.tokens_issued = 0
        self.expires_in = 3600
        self.token_status = 200
        self.responses = []  # queued (status, json) for the non-token calls
        self.raises = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raises:
            raise self.raises
        if request.url.path == "/api/v1/token":
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": "invalid_client"})
            self.tokens_issued += 1
            return httpx.Response(200, json={"access_token": f"token-{self.tokens_issued}",
                                             "expires_in": self.expires_in, "token_type": "Bearer"})
        status, body = self.responses.pop(0)
        return httpx.Response(status, json=body)

    def api_requests(self):
        return [r for r in self.requests if r.url.path != "/api/v1/token"]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def gaia():
    return FakeGaia()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def client(gaia, clock):
    http = httpx.AsyncClient(transport=httpx.MockTransport(gaia.handler))
    return GaiaClient(http, GaiaAuth(http, BASE + "/", "ares-client", SECRET, clock=clock))


def run(coro):
    return asyncio.run(coro)


def issue_doc(**attributes):
    return {"data": {"type": "issue", "id": ISSUE_ID, "attributes": attributes}}


def test_get_search_state_reads_the_three_fields(client, gaia):
    gaia.responses.append((200, issue_doc(searchSemanticText="Login fails", searchDesiredHash=HASH,
                                          searchActualHash=None)))

    state = run(client.get_search_state("issue", ISSUE_ID))

    assert state == SearchState(ISSUE_ID, "Login fails", HASH, None)
    request = gaia.api_requests()[0]
    assert request.method == "GET"
    assert request.url.path == f"/api/v1/issue/{ISSUE_ID}"
    assert request.url.params["fields"] == (
        "Issue.id,Issue.searchSemanticText,Issue.searchDesiredHash,Issue.searchActualHash")
    assert request.headers["Authorization"] == "Bearer token-1"


def test_get_search_state_keeps_desired_and_actual_hash_apart(client, gaia):
    old_hash = "b" * 64
    gaia.responses.append((200, issue_doc(searchSemanticText="Login fails", searchDesiredHash=HASH,
                                          searchActualHash=old_hash)))

    state = run(client.get_search_state("issue", ISSUE_ID))

    assert (state.desired_hash, state.actual_hash) == (HASH, old_hash)


def test_sign_in_uses_client_credentials(client, gaia):
    gaia.responses.append((200, issue_doc()))

    run(client.get_search_state("issue", ISSUE_ID))

    token_request = gaia.requests[0]
    assert token_request.method == "POST"
    assert str(token_request.url) == BASE + "/api/v1/token"
    assert parse_qs(token_request.content.decode()) == {
        "grant_type": ["client_credentials"], "client_id": ["ares-client"], "client_secret": [SECRET]}


def test_store_embedding_patches_vector_and_hash(client, gaia):
    gaia.responses.append((200, issue_doc(searchActualHash=HASH)))

    result = run(client.store_embedding("issue", ISSUE_ID, [0.1, 0.2], HASH))

    assert result == EmbeddingStored(ISSUE_ID, HASH)
    request = gaia.api_requests()[0]
    assert request.method == "PATCH"
    assert request.url.path == f"/api/v1/issue/{ISSUE_ID}"
    assert json.loads(request.content) == {"data": {"type": "issue", "id": ISSUE_ID, "attributes": {
        "searchEmbedding": [0.1, 0.2], "searchActualHash": HASH}}}


def test_stale_hash_is_a_result(client, gaia):
    gaia.responses.append((409, {"errors": [{"detail": "The search text changed"}]}))

    result = run(client.store_embedding("issue", ISSUE_ID, [0.1], HASH))

    assert result == StaleHash("issue", ISSUE_ID, HASH)


@pytest.mark.parametrize("call", ["get", "store"])
def test_missing_record_is_a_result(client, gaia, call):
    gaia.responses.append((404, {"errors": [{"detail": "Not found"}]}))

    if call == "get":
        result = run(client.get_search_state("issue", ISSUE_ID))
    else:
        result = run(client.store_embedding("issue", ISSUE_ID, [0.1], HASH))

    assert result == RecordNotFound("issue", ISSUE_ID)


def test_token_is_reused_until_it_nears_expiry(client, gaia, clock):
    gaia.responses += [(200, issue_doc())] * 3

    run(client.get_search_state("issue", ISSUE_ID))
    clock.now += 3000
    run(client.get_search_state("issue", ISSUE_ID))
    assert gaia.tokens_issued == 1

    clock.now += 600  # within a minute of the 3600s lifetime
    run(client.get_search_state("issue", ISSUE_ID))
    assert gaia.tokens_issued == 2
    assert gaia.api_requests()[-1].headers["Authorization"] == "Bearer token-2"


def test_401_signs_in_again_and_retries_once(client, gaia):
    gaia.responses += [(401, {"error": "invalid_token"}), (200, issue_doc(searchDesiredHash=HASH))]

    state = run(client.get_search_state("issue", ISSUE_ID))

    assert state.desired_hash == HASH
    assert gaia.tokens_issued == 2
    assert [r.headers["Authorization"] for r in gaia.api_requests()] == ["Bearer token-1", "Bearer token-2"]


def test_401_after_a_fresh_token_raises_auth_error(client, gaia):
    gaia.responses += [(401, {"error": "invalid_token"}), (401, {"error": "invalid_token"})]

    with pytest.raises(GaiaAuthError) as raised:
        run(client.get_search_state("issue", ISSUE_ID))

    assert raised.value.status == 401
    assert len(gaia.api_requests()) == 2


def test_403_raises_auth_error_without_retry(client, gaia):
    gaia.responses.append((403, {"errors": [{"detail": "Forbidden"}]}))

    with pytest.raises(GaiaAuthError) as raised:
        run(client.store_embedding("issue", ISSUE_ID, [0.1], HASH))

    assert raised.value.status == 403
    assert gaia.tokens_issued == 1


def test_5xx_raises_unavailable(client, gaia):
    gaia.responses.append((503, {"errors": [{"detail": "Down"}]}))

    with pytest.raises(GaiaUnavailable) as raised:
        run(client.get_search_state("issue", ISSUE_ID))

    assert raised.value.status == 503


def test_400_raises_request_rejected(client, gaia):
    gaia.responses.append((400, {"errors": [{"detail": "searchEmbedding must have 768 numbers"}]}))

    with pytest.raises(GaiaRequestRejected) as raised:
        run(client.store_embedding("issue", ISSUE_ID, [0.1], HASH))

    assert "768 numbers" in str(raised.value)


def test_unreachable_gaia_raises_unavailable(client, gaia):
    gaia.raises = httpx.ConnectError("connection refused")

    with pytest.raises(GaiaUnavailable):
        run(client.get_search_state("issue", ISSUE_ID))


def test_refused_sign_in_raises_auth_error_without_the_secret(client, gaia, caplog):
    gaia.token_status = 401
    caplog.set_level(logging.DEBUG)

    with pytest.raises(GaiaAuthError) as raised:
        run(client.get_search_state("issue", ISSUE_ID))

    assert raised.value.status == 401
    assert SECRET not in str(raised.value)
    assert SECRET not in caplog.text


def test_secret_is_never_logged_or_shown(client, gaia, caplog):
    caplog.set_level(logging.DEBUG)
    gaia.responses.append((200, issue_doc()))

    run(client.get_search_state("issue", ISSUE_ID))

    assert "Signed in to Gaia" in caplog.text
    assert SECRET not in caplog.text
    assert SECRET not in repr(client._auth)


def test_missing_config_raises_config_error(gaia, monkeypatch):
    for name in ("GAIA_URL", "GAIA_CLIENT_ID", "GAIA_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    http = httpx.AsyncClient(transport=httpx.MockTransport(gaia.handler))
    client = GaiaClient(http, GaiaAuth.from_env(http))

    with pytest.raises(GaiaConfigError):
        run(client.get_search_state("issue", ISSUE_ID))

    assert gaia.requests == []


def test_from_env_reads_the_gaia_settings(monkeypatch):
    monkeypatch.setenv("GAIA_URL", BASE)
    monkeypatch.setenv("GAIA_CLIENT_ID", "ares-client")
    monkeypatch.setenv("GAIA_CLIENT_SECRET", SECRET)

    client = GaiaClient.from_env()

    assert client._auth.base_url == BASE
    assert SECRET not in repr(client._auth)
    run(client.aclose())


@pytest.mark.parametrize("resource, record_id", [("../token", ISSUE_ID), ("Issue", ISSUE_ID), ("issue", "")])
def test_rejects_unsafe_paths(client, gaia, resource, record_id):
    with pytest.raises(ValueError):
        run(client.get_search_state(resource, record_id))

    assert gaia.requests == []


def test_record_id_is_escaped(client, gaia):
    gaia.responses.append((404, {}))

    run(client.get_search_state("issue", "a/../b"))

    assert gaia.api_requests()[0].url.raw_path.startswith(b"/api/v1/issue/a%2F..%2Fb")


def test_second_401_drops_the_refused_token(client, gaia):
    gaia.responses += [(401, {}), (401, {}), (200, issue_doc(searchDesiredHash=HASH))]

    with pytest.raises(GaiaAuthError):
        run(client.get_search_state("issue", ISSUE_ID))
    run(client.get_search_state("issue", ISSUE_ID))

    assert gaia.tokens_issued == 3
    assert gaia.api_requests()[-1].headers["Authorization"] == "Bearer token-3"


def test_concurrent_calls_sign_in_once(client, gaia):
    gaia.responses += [(200, issue_doc()) for _ in range(5)]

    async def many():
        return await asyncio.gather(*(client.get_search_state("issue", ISSUE_ID) for _ in range(5)))

    run(many())

    assert gaia.tokens_issued == 1


def test_invalidating_an_old_token_keeps_the_new_one(gaia, clock):
    http = httpx.AsyncClient(transport=httpx.MockTransport(gaia.handler))
    auth = GaiaAuth(http, BASE, "ares-client", SECRET, clock=clock)

    async def flow():
        old = await auth.token()
        await auth.invalidate(old)
        new = await auth.token()
        await auth.invalidate(old)
        return new, await auth.token()

    new, current = run(flow())

    assert new == current == "token-2"
    assert gaia.tokens_issued == 2


def test_rate_limited_sign_in_raises_unavailable(client, gaia):
    gaia.token_status = 429

    with pytest.raises(GaiaUnavailable) as raised:
        run(client.get_search_state("issue", ISSUE_ID))

    assert raised.value.status == 429


def test_malformed_success_reply_raises_unavailable(client, gaia):
    gaia.responses.append((200, ["not", "a", "document"]))

    with pytest.raises(GaiaUnavailable):
        run(client.get_search_state("issue", ISSUE_ID))


def test_unreadable_patch_reply_still_counts_as_stored(client, gaia):
    gaia.responses.append((200, ["not", "a", "document"]))

    stored = run(client.store_embedding("issue", ISSUE_ID, [0.1], HASH))

    assert stored == EmbeddingStored(ISSUE_ID, HASH)
