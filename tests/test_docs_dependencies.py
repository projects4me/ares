from fastapi.testclient import TestClient

from app import dependencies
from app.main import app


def test_docs_reports_an_unavailable_provider(monkeypatch):
    monkeypatch.setattr(dependencies, "_adapter", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AI_PROVIDER", "anthropic")

    response = TestClient(app).post("/docs/complete", json={"before": "Hello"})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
