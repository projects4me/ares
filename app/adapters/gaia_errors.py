"""
Errors raised by the Gaia gateway (gaia_auth, gaia_client).

Expected outcomes of a call (record not found, stale hash) are results, not
errors: see gaia_client. These are the failures a caller can't act on except by
reporting them or trying again later.
"""

from typing import Optional


class GaiaError(Exception):
    """A Gaia call failed; status is Gaia's HTTP status, or None when Gaia gave no answer."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class GaiaConfigError(GaiaError):
    """GAIA_URL, GAIA_CLIENT_ID or GAIA_CLIENT_SECRET is missing."""


class GaiaAuthError(GaiaError):
    """Gaia refused the ares client: sign-in failed, a fresh token was still 401, or the call was 403."""


class GaiaUnavailable(GaiaError):
    """Gaia could not be reached or answered 5xx; worth trying again later."""


class GaiaRequestRejected(GaiaError):
    """Gaia refused the request itself (4xx other than 401/403/404/409), e.g. a malformed embedding (400)."""
