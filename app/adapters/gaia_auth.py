"""
Gaia sign-in for Ares: the OAuth client_credentials grant as the `ares` service client.

Configured from env only (a Gaia service client, Gaia D-065/D-066):
- GAIA_URL: Gaia's base URL, e.g. https://api.local-projects4.me
- GAIA_CLIENT_ID, GAIA_CLIENT_SECRET: the ares Serviceclient's OAuth credentials

The secret is never logged, never put in an error message and never shown by repr().
"""

import asyncio
import logging
import os
import time
from typing import Optional

import httpx

from app.adapters.gaia_errors import GaiaAuthError, GaiaConfigError, GaiaUnavailable

logger = logging.getLogger(__name__)

TOKEN_PATH = "/api/v1/token"
# Sign in again this long before the token expires, so a request never goes out with a token about to lapse.
REFRESH_MARGIN_SECONDS = 60
DEFAULT_LIFETIME_SECONDS = 3600


class GaiaAuth:
    """Owns the ares service token: signs in, caches the token and signs in again before expiry or after a 401."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, client_id: str, client_secret: str,
                 clock=time.monotonic):
        self._http = http
        self._base_url = (base_url or "").rstrip("/")
        self._client_id = client_id or ""
        self._client_secret = client_secret or ""
        self._clock = clock
        self._token: Optional[str] = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls, http: httpx.AsyncClient) -> "GaiaAuth":
        return cls(http, os.getenv("GAIA_URL", ""), os.getenv("GAIA_CLIENT_ID", ""),
                   os.getenv("GAIA_CLIENT_SECRET", ""))

    @property
    def base_url(self) -> str:
        return self._base_url

    def __repr__(self) -> str:
        return f"GaiaAuth(base_url={self._base_url!r}, client_id={self._client_id!r})"

    async def token(self) -> str:
        """A valid access token, signing in when there is none or it is about to expire."""
        async with self._lock:
            if self._token is None or self._clock() >= self._expires_at:
                await self._sign_in()
            return self._token

    async def invalidate(self, token: str) -> None:
        """Drop the token Gaia refused (401), unless a concurrent request already replaced it."""
        async with self._lock:
            if self._token == token:
                self._token = None

    async def _sign_in(self) -> None:
        if not self._base_url or not self._client_id or not self._client_secret:
            raise GaiaConfigError("GAIA_URL, GAIA_CLIENT_ID and GAIA_CLIENT_SECRET must be set to sign in to Gaia")
        form = {"grant_type": "client_credentials", "client_id": self._client_id,
                "client_secret": self._client_secret}
        try:
            response = await self._http.post(self._base_url + TOKEN_PATH, data=form)
        except httpx.HTTPError as e:
            raise GaiaUnavailable(f"Gaia sign-in failed: {type(e).__name__}") from None
        if response.status_code >= 500 or response.status_code == 429:
            raise GaiaUnavailable(f"Gaia sign-in failed ({response.status_code})", response.status_code)
        payload = _json_or_empty(response)
        token = payload.get("access_token")
        if response.status_code != 200 or not token:
            # Only Gaia's error code: never the request, which holds the secret.
            reason = payload.get("error") or "no access token"
            raise GaiaAuthError(f"Gaia sign-in as {self._client_id} failed ({response.status_code}): {reason}",
                                response.status_code)
        lifetime = _lifetime(payload.get("expires_in"))
        self._token = token
        self._expires_at = self._clock() + max(lifetime - REFRESH_MARGIN_SECONDS, lifetime / 2)
        logger.info("Signed in to Gaia as %s; the token is valid for %ss", self._client_id, lifetime)


def _lifetime(expires_in) -> float:
    try:
        value = float(expires_in)
    except (TypeError, ValueError):
        return DEFAULT_LIFETIME_SECONDS
    return value if value > 0 else DEFAULT_LIFETIME_SECONDS


def _json_or_empty(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}
