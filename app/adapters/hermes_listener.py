"""
Hermes listener: takes search.semantic.changed on the ares service scope.

Gaia publishes the event to Hermes with projectId "service:ares" and the record's REST
resource and id in the envelope's resource (Hermes D-024). Hermes lets an ares service
token open a socket and subscribe to exactly that intent.

Its URL is HERMES_URL, e.g. https://hermes.local-projects4.me (read by factories.search_sync_factory).

Each connect sends a fresh ares token: Hermes disconnects a socket when its token
expires (Hermes D-002), and python-socketio does not reconnect after a server-side
disconnect, so this listener runs its own reconnect loop with backoff. A connect refused
as unauthorized drops the cached token, so the next try signs in again. Tokens and
event contents are never logged.
"""

import asyncio
import logging
import random
import re
import time
from typing import Awaitable, Callable, Optional, Protocol

import socketio

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
SCOPE = "service:ares"
EVENT_NAME = "search.semantic.changed"
INTENT = {"projectId": SCOPE, "eventName": EVENT_NAME}
SUBSCRIBE_TIMEOUT_SECONDS = 10
CONNECT_TIMEOUT_SECONDS = 10
BACKOFF_MIN_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0
# A session must stay up this long to reset the backoff, so a socket dropped at once does not reconnect every second.
STABLE_SESSION_SECONDS = 30.0
_RESOURCE = re.compile(r"[a-z][a-z0-9]*")


class TokenSource(Protocol):
    """The ares service token (adapters.gaia_auth.GaiaAuth)."""

    async def token(self) -> str: ...

    async def invalidate(self, token: str) -> None: ...


ChangeHandler = Callable[[str, str], None]
SubscribedHandler = Callable[[], None]


class HermesListener:
    """Owns the Socket.IO connection to Hermes: connect, subscribe, hand each change on, reconnect."""

    def __init__(self, url: str, tokens: TokenSource, on_change: ChangeHandler, on_subscribed: SubscribedHandler,
                 client_factory: Callable[[], socketio.AsyncClient] = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 backoff_min: float = BACKOFF_MIN_SECONDS, backoff_max: float = BACKOFF_MAX_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self._url = (url or "").rstrip("/")
        self._tokens = tokens
        self._on_change = on_change
        self._on_subscribed = on_subscribed
        self._client_factory = client_factory or _new_client
        self._sleep = sleep
        self._backoff_min = backoff_min
        self._backoff_max = backoff_max
        self._clock = clock
        self._revision = 0
        self._client: Optional[socketio.AsyncClient] = None
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="hermes-listener")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        await self._disconnect()

    async def run(self) -> None:
        """Stay connected and subscribed until cancelled, backing off after every lost or failed connection."""
        failures = 0
        while True:
            started = self._clock()
            subscribed = await self._session()
            # Only a subscribed session that stayed up resets the backoff.
            stable = subscribed and self._clock() - started >= STABLE_SESSION_SECONDS
            failures = 0 if stable else failures + 1
            await self._sleep(self._delay(failures))

    def _delay(self, failures: int) -> float:
        """Exponential backoff with jitter, so a restarted Hermes is not hit by every client at once."""
        return min(self._backoff_min * 2 ** min(failures, 16), self._backoff_max) * random.uniform(0.5, 1.0)

    async def _session(self) -> bool:
        """One connection: connect, subscribe, and wait until it ends. True when it was subscribed."""
        client = self._client_factory()
        self._client = client
        used = {"token": None, "unauthorized": False}

        async def auth():
            used["token"] = await self._tokens.token()
            return {"token": used["token"]}

        async def on_connect_error(data):
            used["unauthorized"] = _message(data) == "unauthorized"

        async def on_event(envelope):
            self._handle_event(envelope)

        client.on("connect_error", on_connect_error)
        client.on("domain:event", on_event)
        try:
            await client.connect(self._url, auth=auth, transports=["websocket"],
                                 wait_timeout=CONNECT_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if used["unauthorized"] and used["token"]:
                await self._tokens.invalidate(used["token"])
            logger.warning("Hermes connect failed%s: %s", " (unauthorized)" if used["unauthorized"] else "",
                           type(e).__name__)
            await self._disconnect()
            return False

        try:
            if not await self._subscribe(client):
                return False
            logger.info("Listening to Hermes for %s", EVENT_NAME)
            self._on_subscribed()
            await client.wait()
            logger.info("Hermes disconnected; reconnecting")
            return True
        finally:
            await self._disconnect()

    async def _subscribe(self, client: socketio.AsyncClient) -> bool:
        self._revision += 1
        payload = {"protocolVersion": PROTOCOL_VERSION, "revision": self._revision, "intents": [INTENT]}
        try:
            ack = await client.call("intents:set", payload, timeout=SUBSCRIBE_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Hermes did not acknowledge the subscription: %s", type(e).__name__)
            return False
        accepted = ack.get("accepted") if isinstance(ack, dict) else None
        if not isinstance(accepted, list) or INTENT not in accepted:
            rejected = ack.get("rejected") if isinstance(ack, dict) else ack
            logger.error("Hermes refused the %s subscription: %s", EVENT_NAME, rejected)
            return False
        return True

    def _handle_event(self, envelope) -> None:
        if not isinstance(envelope, dict) or envelope.get("eventName") != EVENT_NAME \
                or envelope.get("projectId") != SCOPE:
            return
        resource = envelope.get("resource")
        resource_type = resource.get("type") if isinstance(resource, dict) else None
        record_id = resource.get("id") if isinstance(resource, dict) else None
        if not isinstance(resource_type, str) or not _RESOURCE.fullmatch(resource_type) \
                or not isinstance(record_id, str) or not record_id:
            logger.warning("Ignored a %s event without a usable resource", EVENT_NAME)
            return
        self._on_change(resource_type, record_id)

    async def _disconnect(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await client.disconnect()
        except Exception as e:
            logger.debug("Hermes disconnect failed: %s", type(e).__name__)


def _new_client() -> socketio.AsyncClient:
    # The reconnect loop is ours; the library's would not run after a server-side disconnect.
    return socketio.AsyncClient(reconnection=False, handle_sigint=False)


def _message(data) -> Optional[str]:
    if isinstance(data, dict):
        return data.get("message")
    return data if isinstance(data, str) else None
