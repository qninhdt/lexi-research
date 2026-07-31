from fastapi.testclient import TestClient

from lexi_research.format import BandConfig, default_config_path
from serve.app import create_app


class Backend:
    async def grade(self, target, sense, text):
        return {"correction": text, "meaning": 4, "feedback": "Good sentence."}


def test_health_ready_and_completion() -> None:
    client = TestClient(
        create_app(Backend(), BandConfig.from_json(default_config_path()), adapter_revision="abc")
    )
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/readyz").json()["adapter_revision"] == "abc"
    response = client.post(
        "/v1/chat/completions",
        json={
            "target": "bright",
            "definition": "full of light",
            "pos": "adjective",
            "text": "The room is bright.",
        },
    )
    assert response.status_code == 200
    assert set(response.json()) >= {"correction", "meaning", "grammar", "naturalness", "feedback"}
