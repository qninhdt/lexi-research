"""One interface for every serving engine, plus capability flags.

Capability flags rather than try/except. An engine that cannot do FP8 should make
that arm *skipped and reported as skipped*, not a crash to interpret or, worse, a
silent fallback to bf16 that lands in the results as an FP8 number.

Nothing here imports an engine. Adapters import lazily inside `launch`, so a
machine with none installed can still enumerate arms, read reports, and run the
tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class EngineError(RuntimeError):
    """The engine could not be started, or was asked for something it cannot do."""


@dataclass(frozen=True)
class Capabilities:
    """What an engine can actually do on this machine.

    `reason` explains a `False` so a skipped arm carries its explanation into the
    report instead of leaving a hole a reader fills in with a guess.
    """

    quantisations: frozenset[str] = frozenset({"bf16"})
    supports_lora: bool = False
    supports_mtp: bool = False
    supports_prefix_cache: bool = False
    supports_constrained_decoding: bool = False
    notes: Mapping[str, str] = field(default_factory=dict)

    def can(self, feature: str) -> bool:
        return bool(getattr(self, f"supports_{feature}", False))

    def why_not(self, feature: str) -> str:
        return self.notes.get(feature, f"the engine reports no support for {feature}")


@dataclass(frozen=True)
class Launched:
    """A running server: where to reach it, and what it is."""

    base_url: str
    engine: str
    digest: str
    quantisation: str
    extra: Mapping[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    """Launch, wait for ready, serve an OpenAI-compatible URL, tear down."""

    name: str

    def capabilities(self) -> Capabilities: ...

    def launch(self, **options: Any) -> Launched: ...

    def shutdown(self) -> None: ...


def skip_reason(
    capabilities: Capabilities, *, quantisation: str, features: Sequence[str]
) -> str | None:
    """Why this arm cannot run here, or None if it can.

    Returned rather than raised: the caller records it in the report, and an arm
    that never ran must be visible as skipped rather than missing.
    """
    if quantisation not in capabilities.quantisations:
        return (
            f"quantisation {quantisation!r} unsupported; this engine offers "
            f"{sorted(capabilities.quantisations)}"
        )
    for feature in features:
        if not capabilities.can(feature):
            return capabilities.why_not(feature)
    return None


__all__ = ["Capabilities", "Engine", "EngineError", "Launched", "skip_reason"]
