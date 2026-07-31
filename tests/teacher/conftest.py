"""Fakes and fixtures for the teacher tests. Nothing here touches the network.

The client's contract is what these fakes exist to pin down: a cache hit must not
reach the provider, a transient failure must be retried, and the concurrency cap
must actually bind. `FakeLLM` therefore records its own peak in-flight count —
asserting on the semaphore's internals would test the implementation, whereas
observed overlap tests the property.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from lexi_research.teacher import (
    GraderOutput,
    ResponseCache,
    SenseRef,
    TeacherClient,
    TeacherConfig,
)

_T = TypeVar("_T", bound=BaseModel)


class FakeLLM:
    """A `StructuredLLM` that always returns the same payload.

    `delay` makes a call take measurable time so concurrency is observable;
    `max_in_flight` is the peak overlap seen, which is what the cap is asserted
    against.
    """

    def __init__(
        self,
        payload: Any,
        *,
        usage: tuple[int, int] = (100, 20),
        delay: float = 0.0,
    ) -> None:
        self.payload = payload
        self.last_usage = usage
        self.delay = delay
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def parse(self, messages: list[dict[str, str]], schema: type[_T]) -> _T:
        self.calls += 1
        self.messages.append(messages)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self._respond(schema)
        finally:
            self.in_flight -= 1

    def _respond(self, schema: type[_T]) -> _T:
        if isinstance(self.payload, schema):
            return self.payload
        return schema.model_validate(self.payload)


class FlakyLLM(FakeLLM):
    """Fails the first `failures` calls, then succeeds.

    `fail_on_text` fails only the calls whose user turn contains a marker, which
    is how `map`'s "quarantine one row, keep the rest" behaviour gets tested
    without depending on call ordering.
    """

    def __init__(
        self,
        payload: Any,
        *,
        failures: int = 0,
        fail_on_text: str | None = None,
        usage: tuple[int, int] = (100, 20),
        delay: float = 0.0,
    ) -> None:
        super().__init__(payload, usage=usage, delay=delay)
        self.failures = failures
        self.fail_on_text = fail_on_text
        self._failed = 0

    def _respond(self, schema: type[_T]) -> _T:
        if self.fail_on_text is not None:
            latest = self.messages[-1]
            if any(self.fail_on_text in message.get("content", "") for message in latest):
                raise RuntimeError("provider rejected this request")
        if self._failed < self.failures:
            self._failed += 1
            raise RuntimeError("transient provider failure")
        return super()._respond(schema)


@pytest.fixture
def config() -> TeacherConfig:
    """A config that cannot accidentally reach a real provider."""
    return TeacherConfig(
        base_url="http://localhost:1/v1",
        api_key="test-key",
        model="test-model",
        # Retries must not actually sleep: the client jitters over
        # `base_delay * 2**attempt`, so zero keeps the suite fast.
        base_delay=0.0,
        max_retries=3,
        concurrency=4,
        prompt_cost_per_mtok=1.0,
        completion_cost_per_mtok=2.0,
    )


@pytest.fixture
def client_factory(
    config: TeacherConfig,
    tmp_path: Path,
) -> Callable[..., TeacherClient]:
    """Build a `TeacherClient` over a fake LLM and a throwaway cache directory."""

    def build(
        llm: FakeLLM,
        *,
        cache: ResponseCache | None = None,
        prompt_hash: str = "test-prompt-hash",
        **overrides: Any,
    ) -> TeacherClient:
        from dataclasses import replace

        return TeacherClient(
            replace(config, **overrides) if overrides else config,
            cache=cache if cache is not None else ResponseCache(tmp_path / "cache"),
            llm=llm,
            prompt_hash=prompt_hash,
        )

    return build


@pytest.fixture
def sense() -> SenseRef:
    """The worked example from the design doc, so prompt assertions read clearly."""
    return SenseRef(definition="full of light", pos="adjective")


@pytest.fixture
def grader_output() -> GraderOutput:
    return GraderOutput(
        correction="The room [have>has:agr] bright light in [>the:art] morning.",
        meaning=4,
        feedback="Good use of 'bright', but check the verb agreement.",
    )
