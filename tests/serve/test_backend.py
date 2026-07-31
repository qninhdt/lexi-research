import httpx
import pytest

from lexi_research.teacher.schemas import SenseRef
from serve.backend import OpenAIBackend


async def test_backend_posts_shared_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    async def post(self, url, *, json, headers):
        assert url == "http://backend/v1/chat/completions"
        assert json["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"correction": null, "meaning": 0, '
                                '"feedback": "Unreadable sentence."}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    result = await OpenAIBackend("http://backend/v1", "model").grade(
        "x", SenseRef(definition="d", pos="noun"), "text"
    )
    assert result["meaning"] == 0
