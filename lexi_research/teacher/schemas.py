"""Wire schemas for the two teacher calls, plus run configuration and accounting.

`GraderOutput` is the load-bearing one: it is the schema the teacher fills in
call 2, the schema the student is trained to emit, and the schema the serving
shim parses. One definition, three consumers.

Bands `grammar` and `naturalness` are absent on purpose — code derives them from
the correction's tags (`lexi_research.format.bands`), so a model that emitted
them could disagree with the formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from lexi_research.format import MAX_BAND, MIN_BAND

#: A chat message in the provider-neutral format consumed by the prompt registry.
ChatMsg = dict[str, str]

#: Strategies supported by LangChain's `with_structured_output`.
STRUCTURED_METHODS: tuple[str, ...] = ("json_schema", "function_calling", "json_mode")


class GraderOutput(BaseModel):
    """Call 2's output — the ground truth a training row is built from.

    Kept structurally permissive (any `meaning` int, any `feedback` string) so a
    schema violation surfaces through `validate_output`'s typed rejection with a
    stable reason code, instead of a pydantic error that costs a retry and
    reports nothing countable. The range is still declared for the provider's
    benefit: `json_schema` mode enforces it server-side where supported.
    """

    correction: str | None = Field(
        description=(
            "The learner sentence re-emitted verbatim with edits marked inline as "
            "[original>replacement:tag]. Null when the sentence is too broken to correct."
        )
    )
    meaning: int = Field(
        description=(
            f"How well the sentence uses the target word in the given sense, {MIN_BAND}-{MAX_BAND}."
        ),
    )
    feedback: str = Field(description="Exactly one sentence of feedback, in English.")


class DiversifiedSentence(BaseModel):
    """One learner-like sentence produced by call 1, tied back to its spec."""

    spec_id: str = Field(description="The spec id this sentence was written for.")
    text: str = Field(description="The learner-written sentence, in English.")


class DiversifyBatch(BaseModel):
    """Call 1's output — K sentences for one sense, one per requested spec."""

    sentences: list[DiversifiedSentence]


@dataclass(frozen=True)
class TeacherConfig:
    """Everything needed to talk to an OpenAI-compatible endpoint.

    No default `base_url`, `api_key` or `model`: a silent default here would send
    a generation run to the wrong provider and only show up in the bill.
    """

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    #: `json_mode` emits a JSON object in the normal text channel; the other two
    #: strategies use provider-native schema or tool calling support.
    method: str = "json_mode"
    reasoning_effort: str = ""
    max_retries: int = 4
    base_delay: float = 0.5
    #: Concurrent in-flight requests. Also the knob to turn down on a 429 storm.
    concurrency: int = 8
    #: Cost per million tokens, for the run report. Zeros mean "unknown", which
    #: reports as 0.0 rather than pretending to a number nobody supplied.
    prompt_cost_per_mtok: float = 0.0
    completion_cost_per_mtok: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in STRUCTURED_METHODS:
            raise ValueError(
                f"unknown structured-output method {self.method!r}; "
                f"expected one of {STRUCTURED_METHODS}"
            )

    def cost_of(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_cost_per_mtok
            + completion_tokens * self.completion_cost_per_mtok
        ) / 1_000_000

    @classmethod
    def from_env(cls, env: dict[str, str], **overrides: Any) -> TeacherConfig:
        """Build from environment variables; secrets never live in tracked files.

        Reads `LEXI_TEACHER_BASE_URL`, `LEXI_TEACHER_API_KEY`, `LEXI_TEACHER_MODEL`,
        and optional runtime/accounting controls under `LEXI_TEACHER_*`.
        """
        missing = [
            name
            for name in ("LEXI_TEACHER_BASE_URL", "LEXI_TEACHER_API_KEY", "LEXI_TEACHER_MODEL")
            if not env.get(name)
        ]
        if missing:
            raise ValueError(f"missing required environment variables: {', '.join(missing)}")

        kwargs: dict[str, Any] = {
            "base_url": env["LEXI_TEACHER_BASE_URL"],
            "api_key": env["LEXI_TEACHER_API_KEY"],
            "model": env["LEXI_TEACHER_MODEL"],
        }
        if value := env.get("LEXI_TEACHER_TEMPERATURE"):
            kwargs["temperature"] = float(value)
        if value := env.get("LEXI_TEACHER_METHOD"):
            kwargs["method"] = value
        if value := env.get("LEXI_TEACHER_REASONING_EFFORT"):
            kwargs["reasoning_effort"] = value
        if value := env.get("LEXI_TEACHER_CONCURRENCY"):
            kwargs["concurrency"] = int(value)
        if value := env.get("LEXI_TEACHER_MAX_RETRIES"):
            kwargs["max_retries"] = int(value)
        if value := env.get("LEXI_TEACHER_BASE_DELAY"):
            kwargs["base_delay"] = float(value)
        if value := env.get("LEXI_TEACHER_PROMPT_COST_PER_MTOK"):
            kwargs["prompt_cost_per_mtok"] = float(value)
        if value := env.get("LEXI_TEACHER_COMPLETION_COST_PER_MTOK"):
            kwargs["completion_cost_per_mtok"] = float(value)
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class CallStats:
    """Token, cost, retry and cache accounting for a run.

    Mutable and accumulated in place by the client. `cache_hits` counts calls
    that never reached the network, which is what makes a resumed run's real
    spend legible.
    """

    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    @property
    def network_calls(self) -> int:
        return self.calls - self.cache_hits

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.calls if self.calls else 0.0

    def record_usage(self, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost += cost

    def as_dict(self) -> dict[str, float | int]:
        """Flat mapping for the run report."""
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "network_calls": self.network_calls,
            "hit_rate": round(self.hit_rate, 4),
            "retries": self.retries,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": round(self.cost, 6),
        }


@dataclass(frozen=True)
class SenseRef:
    """The one dictionary sense a sentence is graded against."""

    definition: str
    pos: str


@dataclass(frozen=True)
class DiversifySpec:
    """A call-1 diversity knob set. Metadata for coverage analysis, never a label.

    `meaning_req` and `error_spec` steer what call 1 writes. Call 2 then reads the
    resulting text blind and decides what is actually true of it.
    """

    spec_id: str
    profile_id: str
    meaning_req: int
    error_spec: str
    #: Tags the profile tends to produce. A hint to call 1, not a target for call 2.
    error_bias: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "CallStats",
    "ChatMsg",
    "DiversifiedSentence",
    "DiversifyBatch",
    "DiversifySpec",
    "GraderOutput",
    "SenseRef",
    "STRUCTURED_METHODS",
    "TeacherConfig",
]
