from fastapi.testclient import TestClient

from app.factories.ai_service_factory import AIServiceFactory
from app.main import app
from tests.conftest import FakeAdapter


def test_planissue_still_awaits_the_adapter(monkeypatch):
    fake = FakeAdapter()
    fake.reply = '```json\n{"tasks": []}\n```'
    monkeypatch.setattr(AIServiceFactory, "create_ai_service", classmethod(lambda cls, provider=None: fake))

    response = TestClient(app).post("/planissue/", json={"issueDetails": {
        "projectName": "A Gaia", "issueSubject": "Subject", "issueDescription": "Description"}})

    assert response.status_code == 200
    assert len(fake.calls) == 1
