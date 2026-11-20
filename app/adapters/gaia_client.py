"""
Gaia gateway for semantic search: reads a record's stored search text and hashes
and stores its embedding, as the `ares` service client (Gaia D-072).

- get_search_state(resource, id): GET /api/v1/<resource>/<id> with the three search
  fields Gaia opens to ares through `fields=`.
- store_embedding(resource, id, embedding, text_hash): PATCH /api/v1/<resource>/<id>
  with {searchEmbedding, searchActualHash}; Gaia stores it only when the hash is
  the record's current searchDesiredHash and answers 409 otherwise.
- list_pending(after, limit): one page of GET /api/v1/searchpending, the records whose
  embedding is missing or outdated (Gaia search D-009), as ids only.

A missing record (404) and a stale hash (409) are results the caller acts on; other
failures raise the errors in gaia_errors. A 401 signs in again and retries once.
"""

import os
import re
from typing import List, Optional, Union
from urllib.parse import quote

import httpx

from app.adapters.gaia_auth import GaiaAuth
from app.adapters.gaia_errors import GaiaAuthError, GaiaRequestRejected, GaiaUnavailable
# The results are defined with the search schemas, so the services need not import this gateway.
from app.schemas.search import (  # noqa: F401  (re-exported for callers of this gateway)
    EmbeddingStored, PendingPage, PendingRecord, RecordNotFound, SearchState, StaleHash)

API_PREFIX = "/api/v1/"
PENDING_PATH = "searchpending"
SEARCH_FIELDS = ("searchSemanticText", "searchDesiredHash", "searchActualHash")
DEFAULT_TIMEOUT_SECONDS = 10.0
# A Gaia resource is its lower-case model name, e.g. "issue".
_RESOURCE = re.compile(r"[a-z][a-z0-9]*")


class GaiaClient:
    """Owns the HTTP transport to Gaia for the semantic search fields."""

    def __init__(self, http: httpx.AsyncClient, auth: GaiaAuth):
        self._http = http
        self._auth = auth

    @classmethod
    def from_env(cls) -> "GaiaClient":
        """A client on its own connection pool, configured from GAIA_URL, GAIA_CLIENT_ID, GAIA_CLIENT_SECRET
        and GAIA_TIMEOUT_SECONDS. Close it with aclose()."""
        http = httpx.AsyncClient(timeout=_timeout_env())
        return cls(http, GaiaAuth.from_env(http))

    @property
    def auth(self) -> GaiaAuth:
        """The ares sign-in this client uses; the Hermes listener connects with the same token."""
        return self._auth

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_search_state(self, resource: str, record_id: str) -> Union[SearchState, RecordNotFound]:
        model = _model_name(resource)
        fields = ",".join([f"{model}.id"] + [f"{model}.{field}" for field in SEARCH_FIELDS])
        response = await self._send("GET", _path(resource, record_id), params={"fields": fields})
        if response.status_code == 404:
            return RecordNotFound(resource, record_id)
        _raise_for_status(response, "GET", resource)
        attributes = _attributes(response)
        return SearchState(
            id=record_id,
            semantic_text=attributes.get("searchSemanticText"),
            desired_hash=attributes.get("searchDesiredHash"),
            actual_hash=attributes.get("searchActualHash"),
        )

    async def store_embedding(self, resource: str, record_id: str, embedding: List[float],
                              text_hash: str) -> Union[EmbeddingStored, RecordNotFound, StaleHash]:
        body = {"data": {"type": resource, "id": record_id,
                         "attributes": {"searchEmbedding": list(embedding), "searchActualHash": text_hash}}}
        response = await self._send("PATCH", _path(resource, record_id), json=body)
        if response.status_code == 404:
            return RecordNotFound(resource, record_id)
        if response.status_code == 409:
            return StaleHash(resource, record_id, text_hash)
        _raise_for_status(response, "PATCH", resource)
        # Gaia stored the embedding; an unreadable reply does not undo that.
        try:
            stored = _attributes(response).get("searchActualHash") or text_hash
        except GaiaUnavailable:
            stored = text_hash
        return EmbeddingStored(record_id, stored)

    async def list_pending(self, after: Optional[str] = None, limit: Optional[int] = None) -> PendingPage:
        """The page of pending records after the cursor (None: from the start); limit None is Gaia's default."""
        params = {}
        if after:
            params["after"] = after
        if limit is not None:
            params["limit"] = str(limit)
        response = await self._send("GET", PENDING_PATH, params=params)
        _raise_for_status(response, "GET", PENDING_PATH)
        return _pending_page(response)

    async def _send(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = self._auth.base_url + API_PREFIX + path
        token = await self._auth.token()
        response = await self._request(method, url, token, **kwargs)
        if response.status_code == 401:
            # The token expired or was revoked: sign in again once and retry.
            await self._auth.invalidate(token)
            token = await self._auth.token()
            response = await self._request(method, url, token, **kwargs)
            if response.status_code == 401:
                # Refused again: drop this token too, so the next call does not start with it.
                await self._auth.invalidate(token)
        return response

    async def _request(self, method: str, url: str, token: str, **kwargs) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.api+json"}
        try:
            return await self._http.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as e:
            raise GaiaUnavailable(f"Gaia {method} {_safe_path(url)} failed: {type(e).__name__}") from None


def _model_name(resource: str) -> str:
    return resource[:1].upper() + resource[1:]


def _path(resource: str, record_id: str) -> str:
    if not _RESOURCE.fullmatch(resource or ""):
        raise ValueError(f"Not a Gaia resource name: {resource!r}")
    if not record_id:
        raise ValueError("A record id is required")
    return f"{resource}/{quote(record_id, safe='')}"


def _safe_path(url: str) -> str:
    return url.split(API_PREFIX, 1)[-1]


def _raise_for_status(response: httpx.Response, method: str, resource: str) -> None:
    status = response.status_code
    if status < 400:
        return
    message = f"Gaia {method} {resource} returned {status}: {_error_detail(response)}"
    if status in (401, 403):
        raise GaiaAuthError(message, status)
    if status >= 500:
        raise GaiaUnavailable(message, status)
    raise GaiaRequestRejected(message, status)


def _attributes(response: httpx.Response) -> dict:
    try:
        data = response.json().get("data") or {}
    except (ValueError, AttributeError):
        # A malformed success reply is a fault on Gaia's side, not a refusal.
        raise GaiaUnavailable(f"Gaia answered {response.status_code} without a JSON:API document",
                              response.status_code)
    attributes = data.get("attributes") if isinstance(data, dict) else None
    return attributes if isinstance(attributes, dict) else {}


def _pending_page(response: httpx.Response) -> PendingPage:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    data = payload.get("data") if isinstance(payload, dict) else None
    next_cursor = payload.get("next") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not (next_cursor is None or isinstance(next_cursor, str)):
        raise GaiaUnavailable(f"Gaia answered {response.status_code} without a pending list", response.status_code)
    records = []
    for item in data:
        fields = [item.get(key) if isinstance(item, dict) else None for key in ("type", "resource", "id")]
        if not all(isinstance(value, str) and value for value in fields):
            raise GaiaUnavailable("Gaia's pending list holds a record without a type, resource or id",
                                  response.status_code)
        records.append(PendingRecord(*fields))
    return PendingPage(records, next_cursor or None)


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return "no JSON reply"
    if isinstance(payload, dict):
        errors = payload.get("errors")
        if isinstance(errors, list) and errors and isinstance(errors[0], dict):
            first = errors[0]
            return str(first.get("detail") or first.get("title") or first)[:300]
        return str(payload.get("message") or payload.get("error") or payload)[:300]
    return str(payload)[:300]


def _timeout_env() -> float:
    try:
        value = float(os.getenv("GAIA_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS
