from app.services.docs_complete_service import MAX_COMPLETION_CHARS

BODY = {"title": "Doc", "headingPath": ["Intro"], "before": "The quick brown", "after": ""}


def test_complete_returns_the_continuation(client, fake_adapter):
    fake_adapter.reply = " fox jumps."

    response = client.post("/docs/complete", json=BODY)

    assert response.status_code == 200
    assert response.json() == {"text": " fox jumps."}


def test_complete_uses_the_fast_model_with_a_low_token_limit(client, fake_adapter, monkeypatch):
    monkeypatch.delenv("ARES_DOCS_COMPLETE_MODEL", raising=False)
    monkeypatch.delenv("ARES_DOCS_COMPLETE_MAX_TOKENS", raising=False)

    client.post("/docs/complete", json=BODY)

    assert fake_adapter.calls[0]["model"] == "claude-haiku-4-5-20251001"
    assert fake_adapter.calls[0]["max_tokens"] <= 100


def test_complete_model_comes_from_env(client, fake_adapter, monkeypatch):
    monkeypatch.setenv("ARES_DOCS_COMPLETE_MODEL", "fast-model")
    monkeypatch.setenv("ARES_DOCS_COMPLETE_MAX_TOKENS", "40")

    client.post("/docs/complete", json=BODY)

    assert fake_adapter.calls[0]["model"] == "fast-model"
    assert fake_adapter.calls[0]["max_tokens"] == 40


def test_complete_truncates_long_output_at_a_word(client, fake_adapter):
    fake_adapter.reply = " word" * 100

    text = client.post("/docs/complete", json=BODY).json()["text"]

    assert 0 < len(text) <= MAX_COMPLETION_CHARS
    assert text.endswith("word")


def test_complete_keeps_only_the_first_line(client, fake_adapter):
    fake_adapter.reply = " fox.\nAnother paragraph"

    assert client.post("/docs/complete", json=BODY).json() == {"text": " fox."}


def test_complete_may_be_empty(client, fake_adapter):
    fake_adapter.reply = ""

    assert client.post("/docs/complete", json=BODY).json() == {"text": ""}


def test_complete_skips_the_model_without_text_before_the_cursor(client, fake_adapter):
    assert client.post("/docs/complete", json={"before": "  "}).json() == {"text": ""}
    assert fake_adapter.calls == []


def test_complete_rejects_oversized_fields(client):
    assert client.post("/docs/complete", json={"before": "x" * 2001}).status_code == 422
    assert client.post("/docs/complete", json={"before": "x", "after": "x" * 501}).status_code == 422
    assert client.post("/docs/complete", json={"before": "x", "headingPath": "not a list"}).status_code == 422


def test_complete_reports_a_provider_failure(client, fake_adapter):
    fake_adapter.error = "Anthropic API error: timeout"

    response = client.post("/docs/complete", json=BODY)

    assert response.status_code == 502
    assert response.json() == {"detail": "Anthropic API error: timeout"}


def test_complete_falls_back_on_empty_or_invalid_env(client, fake_adapter, monkeypatch):
    monkeypatch.setenv("ARES_DOCS_COMPLETE_MODEL", "")
    monkeypatch.setenv("ARES_DOCS_COMPLETE_MAX_TOKENS", "abc")

    client.post("/docs/complete", json=BODY)

    assert fake_adapter.calls[0]["model"] == "claude-haiku-4-5-20251001"
    assert fake_adapter.calls[0]["max_tokens"] == 80


def test_complete_accepts_a_reply_cut_off_by_the_token_limit(client, fake_adapter):
    fake_adapter.reply = " fox jumps over"
    fake_adapter.stop_reason = "max_tokens"

    assert client.post("/docs/complete", json=BODY).json() == {"text": " fox jumps over"}


def test_complete_rejects_long_headings(client):
    assert client.post("/docs/complete", json={"before": "x", "headingPath": ["x" * 501]}).status_code == 422
