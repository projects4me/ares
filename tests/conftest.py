import pytest
from fastapi.testclient import TestClient

from app.adapters.base_ai_adapter import BaseAIAdapter
from app.dependencies import get_ai_adapter
from app.main import app


class FakeAdapter(BaseAIAdapter):
    """Records each chat call and answers with a canned reply; never calls a provider."""

    def __init__(self):
        self.reply = ""
        self.error = None
        self.stop_reason = "end_turn"
        self.calls = []

    async def chat(self, message, **kwargs):
        self.calls.append({"message": message, **kwargs})
        if self.error:
            return {"success": False, "error": self.error, "provider": "fake"}
        return {"success": True, "response": self.reply, "provider": "fake", "stop_reason": self.stop_reason}

    def get_provider_name(self):
        return "fake"

    async def health_check(self):
        return {"success": True, "provider": "fake"}

    async def authenticate(self):
        return {"success": True, "provider": "fake"}


@pytest.fixture
def fake_adapter():
    return FakeAdapter()


@pytest.fixture
def client(fake_adapter):
    app.dependency_overrides[get_ai_adapter] = lambda: fake_adapter
    yield TestClient(app)
    app.dependency_overrides.clear()
