"""The shim's backend, built from an engine adapter rather than a fixed URL.

The contract `serve/app.py` exposes does not change — the existing `tests/serve/`
suite is what says so. What changes is where the completion comes from: any
engine the bench harness can launch can also serve the shim, which is what makes
a latency number and a served product the same configuration rather than two.
"""

from __future__ import annotations

from typing import Any

from lexi_research.teacher.schemas import SenseRef

from .backend import OpenAIBackend


class EngineBackend:
    """An `OpenAIBackend` pointed at an engine this process launched.

    Thin on purpose: the shim already speaks to an OpenAI-compatible endpoint, and
    every engine here exposes one. Re-implementing the client per engine would be
    three chances to disagree about the contract.
    """

    def __init__(self, engine: Any, *, quantisation: str = "bf16", **options: Any) -> None:
        self._engine = engine
        self._launched = engine.launch(quantisation=quantisation, **options)
        self._backend = OpenAIBackend(self._launched.base_url, model="lexi")

    @property
    def launched(self) -> Any:
        return self._launched

    async def grade(self, target: str, sense: SenseRef, text: str) -> dict[str, object]:
        return await self._backend.grade(target, sense, text)

    def close(self) -> None:
        self._engine.shutdown()


__all__ = ["EngineBackend"]
