import asyncio

import pytest
import socketio

from app.adapters.hermes_listener import EVENT_NAME, SCOPE, HermesListener

ISSUE_ID = "1f6b0c3e-0000-4000-8000-000000000001"
ACCEPTED = {"revision": 1, "accepted": [{"projectId": SCOPE, "eventName": EVENT_NAME}], "rejected": []}


class Done(Exception):
    """Raised by the client factory when the scripted sessions run out, ending run()."""


class FakeTokens:
    def __init__(self):
        self.issued = 0
        self.invalidated = []

    async def token(self):
        self.issued += 1
        return f"token-{self.issued}"

    async def invalidate(self, token):
        self.invalidated.append(token)


class FakeClient:
    """One scripted Socket.IO session: refuse the connect, ack the intents, deliver events, then disconnect."""

    def __init__(self, refuse=None, ack=ACCEPTED, events=()):
        self.refuse = refuse
        self.ack = ack
        self.events = list(events)
        self.handlers = {}
        self.connected_with = None
        self.calls = []
        self.disconnected = False

    def on(self, event, handler):
        self.handlers[event] = handler

    async def connect(self, url, auth=None, transports=None, wait_timeout=None):
        self.connected_with = {"url": url, "auth": await auth(), "transports": transports}
        if self.refuse:
            await self.handlers["connect_error"]({"message": self.refuse})
            raise socketio.exceptions.ConnectionError("One or more namespaces failed to connect: /")

    async def call(self, event, data=None, timeout=None):
        self.calls.append((event, data))
        if isinstance(self.ack, Exception):
            raise self.ack
        return self.ack

    async def wait(self):
        for envelope in self.events:
            await self.handlers["domain:event"](envelope)

    async def disconnect(self):
        self.disconnected = True


def envelope(**overrides):
    payload = {"schemaVersion": 1, "eventId": "e1", "eventName": EVENT_NAME, "occurredAt": "2026-11-20T10:00:00Z",
               "projectId": SCOPE, "resource": {"type": "issue", "id": ISSUE_ID}, "actorId": None,
               "changes": {"type": "issue", "resource": "issue", "recordId": ISSUE_ID}}
    payload.update(overrides)
    return payload


class Clock:
    """Each session lasts session_seconds: the listener reads the clock once before and once after it."""

    def __init__(self, session_seconds):
        self.now = 0.0
        self.reads = 0
        self.session_seconds = session_seconds

    def __call__(self):
        self.reads += 1
        if self.reads % 2 == 0:
            self.now += self.session_seconds
        return self.now


def listen(clients, session_seconds=0):
    """Run the listener over the scripted sessions; returns (listener, tokens, changes, subscribes, sleeps)."""
    tokens = FakeTokens()
    changes, subscribes, sleeps = [], [], []
    scripted = list(clients)

    def factory():
        if not scripted:
            raise Done()
        return scripted.pop(0)

    async def sleep(seconds):
        sleeps.append(seconds)

    listener = HermesListener("https://hermes.test/", tokens, on_change=lambda r, i: changes.append((r, i)),
                              on_subscribed=lambda: subscribes.append(True), client_factory=factory, sleep=sleep,
                              backoff_min=1, backoff_max=8, clock=Clock(session_seconds))
    with pytest.raises(Done):
        asyncio.run(listener.run())
    return listener, tokens, changes, subscribes, sleeps


def test_connects_with_a_fresh_token_and_subscribes_to_the_ares_scope():
    client = FakeClient()

    _, tokens, _, subscribes, _ = listen([client])

    assert client.connected_with == {"url": "https://hermes.test", "auth": {"token": "token-1"},
                                     "transports": ["websocket"]}
    assert client.calls == [("intents:set", {"protocolVersion": 1, "revision": 1,
                                             "intents": [{"projectId": SCOPE, "eventName": EVENT_NAME}]})]
    assert subscribes == [True]
    assert client.disconnected


def test_each_change_event_is_handed_on_as_resource_and_id():
    _, _, changes, _, _ = listen([FakeClient(events=[envelope(), envelope(resource={"type": "issue", "id": "2"})])])

    assert changes == [("issue", ISSUE_ID), ("issue", "2")]


@pytest.mark.parametrize("bad", [
    envelope(eventName="issue.created"),
    envelope(projectId="service:runtime"),
    envelope(resource={"type": "issue"}),
    envelope(resource={"type": "../token", "id": "1"}),
    envelope(resource="issue"),
    "not an envelope",
])
def test_other_or_malformed_events_are_ignored(bad):
    _, _, changes, _, _ = listen([FakeClient(events=[bad])])

    assert changes == []


def test_a_reconnect_sends_a_new_token_and_asks_for_another_reconcile():
    first, second = FakeClient(), FakeClient()

    listener, tokens, _, subscribes, sleeps = listen([first, second], session_seconds=60)

    assert first.connected_with["auth"] == {"token": "token-1"}
    assert second.connected_with["auth"] == {"token": "token-2"}
    assert second.calls[0][1]["revision"] == 2
    assert subscribes == [True, True]
    # After a subscribed session that stayed up the backoff starts over (with jitter).
    assert all(0.5 <= s <= 1 for s in sleeps)


def test_sessions_dropped_at_once_back_off_like_failures():
    _, _, _, subscribes, sleeps = listen([FakeClient() for _ in range(3)], session_seconds=1)

    assert subscribes == [True, True, True]
    assert 1 <= sleeps[0] <= 2 and 2 <= sleeps[1] <= 4 and 4 <= sleeps[2] <= 8


def test_an_unauthorized_connect_drops_the_token_and_backs_off():
    _, tokens, _, subscribes, sleeps = listen([FakeClient(refuse="unauthorized"), FakeClient(refuse="unauthorized")])

    assert tokens.invalidated == ["token-1", "token-2"]
    assert subscribes == []
    assert 1 <= sleeps[0] <= 2 and 2 <= sleeps[1] <= 4


def test_another_connect_failure_keeps_the_token():
    _, tokens, _, _, _ = listen([FakeClient(refuse="xhr poll error")])

    assert tokens.invalidated == []


def test_a_refused_subscription_does_not_reconcile_and_disconnects():
    client = FakeClient(ack={"revision": 1, "accepted": [], "rejected": [{"reason": "project_not_authorized"}]})

    _, _, _, subscribes, sleeps = listen([client])

    assert subscribes == []
    assert client.disconnected
    assert 1 <= sleeps[0] <= 2


def test_an_unacknowledged_subscription_does_not_reconcile():
    client = FakeClient(ack=TimeoutError())

    _, _, _, subscribes, _ = listen([client])

    assert subscribes == []
    assert client.disconnected


def test_backoff_is_capped():
    _, _, _, _, sleeps = listen([FakeClient(refuse="x") for _ in range(6)])

    assert max(sleeps) <= 8


def test_stop_while_connected_disconnects():
    class Blocking(FakeClient):
        async def wait(self):
            await asyncio.Event().wait()

    async def scenario():
        client = Blocking()
        subscribed = asyncio.Event()
        listener = HermesListener("https://hermes.test", FakeTokens(), on_change=lambda r, i: None,
                                  on_subscribed=subscribed.set, client_factory=lambda: client)
        listener.start()
        await subscribed.wait()
        await listener.stop()
        return client

    assert asyncio.run(scenario()).disconnected
