"""OpenAI-compatible completion backend for the private shim."""

from __future__ import annotations

import json

import httpx

from lexi_research.teacher import render_grader_prompt
from lexi_research.teacher.schemas import SenseRef


class OpenAIBackend:
    def __init__(
        self, base_url: str, model: str, api_key: str = "", timeout_s: float = 60.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    async def grade(self, target: str, sense: SenseRef, text: str) -> dict[str, object]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "messages": render_grader_prompt(target, sense, text),
            "temperature": 0,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", json=body, headers=headers
            )
            response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("backend completion content is not text")
        decoded = json.loads(content)
        if not isinstance(decoded, dict):
            raise ValueError("backend completion is not a JSON object")
        return decoded
