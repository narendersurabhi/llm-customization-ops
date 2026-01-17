from fastapi.testclient import TestClient

from llm_customization_ops.serving.app import create_app


def test_healthz(monkeypatch) -> None:
    monkeypatch.setenv("LLM_OPS_FAKE_MODEL", "1")
    app = create_app()
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200


def test_generate_fake(monkeypatch) -> None:
    monkeypatch.setenv("LLM_OPS_FAKE_MODEL", "1")
    app = create_app()
    client = TestClient(app)
    payload = {"prompt": "hello", "template_id": "summarization"}
    response = client.post("/v1/generate", json=payload)
    assert response.status_code == 200
    assert "text" in response.json()
