"""Serving core: validate a backend completion before returning derived bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from lexi_research.format import BandConfig, ValidationError, validate_output
from lexi_research.teacher.schemas import SenseRef


class Backend(Protocol):
    async def grade(self, target: str, sense: SenseRef, text: str) -> dict[str, object]: ...


class GradeUnavailable(RuntimeError):
    """The backend repeatedly emitted an invalid grade; no result is fabricated."""


@dataclass(frozen=True)
class Grade:
    correction: str | None
    meaning: int
    grammar: int
    naturalness: int
    feedback: str
    band_config_version: int
    retries: int


async def grade(
    backend: Backend,
    target: str,
    sense: SenseRef,
    text: str,
    config: BandConfig,
    *,
    retries: int = 2,
) -> Grade:
    for attempt in range(retries + 1):
        payload = await backend.grade(target, sense, text)
        result = validate_output(payload, text, config)
        if not isinstance(result, ValidationError):
            return Grade(
                cast(str | None, payload["correction"]),
                result.meaning,
                result.bands.grammar,
                result.bands.naturalness,
                result.feedback,
                config.version,
                attempt,
            )
    raise GradeUnavailable("backend output failed validation after retries")
