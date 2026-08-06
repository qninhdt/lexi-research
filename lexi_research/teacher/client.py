"""The teacher client: one narrow `parse` seam over any OpenAI-compatible endpoint.

Shape borrowed from `lexi-ai`'s `lexi_ai/llm.py` (read-only reference) — a
`StructuredLLM` protocol with two structured-output modes, retry with exponential
backoff, and a lazily built SDK client. It is copied rather than imported: this
repo stays standalone, and a research pipeline should not pin itself to the
product's release cadence.

What is added here, because a bulk generation run needs it:

- **Cache-first dispatch.** Every call looks in the content-addressed cache
  before reaching the network, so resume is the default path rather than a mode.
- **A concurrency cap.** One semaphore, sized by config, is what keeps a
  thousand-call fan-out from turning into a 429 storm.
- **Accounting.** Tokens, cost, retries and cache hits accumulate into
  `CallStats`, which lands in the run report.

Tests inject a fake `StructuredLLM`, so the suite never touches the network.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel, ValidationError

from .cache import ResponseCache, cache_key
from .schemas import CallStats, ChatMsg, TeacherConfig

_T = TypeVar("_T", bound=BaseModel)

#: A request's messages, or a factory that builds them fresh for each attempt.
#:
#: The factory form exists because a retry has to change something to be worth
#: paying for. Measured against this proxy on 100 rows: about 7% of gradings come
#: back with empty tool arguments, and re-asking with byte-identical messages
#: recovered 7 of 8 such rows while re-rendering the prompt recovered 8 of 8.
#: Raising the temperature instead recovered only 5 of 8, so the fix is fresh
#: bytes at temperature 0 rather than a hotter sample.
MessageSource = list[ChatMsg] | Callable[[], list[ChatMsg]]


def render_messages(source: MessageSource) -> list[ChatMsg]:
    """The message list for one attempt, calling the factory if there is one."""
    return source() if callable(source) else source


def _unwrap_arguments(payload: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    """Recover a schema payload the endpoint wrapped or renamed.

    Measured against one proxy: alongside the fields it was asked for, it
    sometimes returns them prefixed (`__correction`) or nested one level under a
    placeholder key it invented (`$PARAMETER_NAME`, `corrections`). The answer is
    present and correct in both shapes, so unwrapping it is free where a retry
    costs a call. Anything that does not match a known shape is passed through
    untouched, so a genuinely wrong payload still fails validation.
    """
    required = {name for name, field in schema.model_fields.items() if field.is_required()}
    if not required or required <= payload.keys():
        return payload

    stripped = {key.lstrip("_"): value for key, value in payload.items()}
    if required <= stripped.keys():
        return stripped

    if len(payload) == 1:
        inner = next(iter(payload.values()))
        if isinstance(inner, dict) and required <= inner.keys():
            return inner

    return payload

#: Retries wait `base_delay * 2**attempt`, jittered. Full jitter (uniform over
#: the whole window, not delay±10%) is what actually de-synchronises a fleet of
#: concurrent workers that all got rate-limited by the same upstream at the same
#: instant; a tight jitter band leaves them retrying in lockstep.
_JITTER = random.Random(0)


@runtime_checkable
class StructuredLLM(Protocol):
    """The injectable seam: turn messages into a validated schema instance."""

    async def parse(self, messages: list[ChatMsg], schema: type[_T]) -> _T: ...


class RetryExhausted(RuntimeError):
    """Every attempt failed. Carries the last cause so a run report can group them.

    Typed rather than a bare re-raise: the generation loop distinguishes "this
    row is unlucky, keep going" from "the endpoint is gone, stop the run", and it
    cannot do that against an arbitrary provider exception.
    """

    def __init__(self, attempts: int, cause: Exception) -> None:
        super().__init__(f"all {attempts} attempts failed: {cause!r}")
        self.attempts = attempts
        self.cause = cause


class EmptyToolArguments(ValueError):
    """The endpoint reported a tool call but sent no arguments to validate.

    Distinguished from a schema violation because the two mean different things:
    a schema violation is the model answering badly, while this is the transport
    losing the answer — measured at roughly 7% of calls against one proxy, with
    `finish_reason: tool_calls` and an empty `content`. Naming it keeps that
    countable in a run report instead of appearing as a puzzling "field required"
    error against an empty dict.
    """


class OpenAIStructuredLLM:
    """Real `StructuredLLM` over an OpenAI-compatible `/chat/completions`.

    Two structured-output methods:

    * `json_schema` — the SDK's native strict `chat.completions.parse`.
    * `function_calling` — the schema as one forced tool, arguments validated
      here. Some OpenAI-compatible proxies accept a strict `json_schema` and then
      return loose JSON anyway; those honour tool calls reliably, so this mode is
      the escape hatch. `probe` reports which one the configured endpoint
      actually honours.

    Usage is exposed through `last_usage` because the caller needs the token
    counts for accounting, and threading them through the `parse` return type
    would put provider bookkeeping into the seam that tests fake.
    """

    def __init__(self, config: TeacherConfig) -> None:
        from openai import AsyncOpenAI

        # TeacherClient owns the retry policy and accounting. Disabling the SDK's
        # hidden retries keeps the configured attempt count and backoff truthful.
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            max_retries=0,
        )
        self._config = config
        self.last_usage: tuple[int, int] = (0, 0)

    def _extra(self) -> dict[str, Any] | None:
        # Not a first-class kwarg on every SDK version or model, so it rides in
        # extra_body where an unsupported model simply ignores it.
        effort = self._config.reasoning_effort
        return {"reasoning_effort": effort} if effort else None

    async def parse(self, messages: list[ChatMsg], schema: type[_T]) -> _T:
        if self._config.method == "function_calling":
            return await self._parse_via_tool(messages, schema)
        return await self._parse_via_json_schema(messages, schema)

    def _record_usage(self, completion: Any) -> None:
        usage = getattr(completion, "usage", None)
        self.last_usage = (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    async def _parse_via_json_schema(self, messages: list[ChatMsg], schema: type[_T]) -> _T:
        completion = await self._client.chat.completions.parse(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            response_format=schema,
            temperature=self._config.temperature,
            extra_body=self._extra(),
        )
        self._record_usage(completion)
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("model returned no parsed structured output")
        return parsed

    async def _parse_via_tool(self, messages: list[ChatMsg], schema: type[_T]) -> _T:
        from openai import pydantic_function_tool

        tool = pydantic_function_tool(schema, name="emit")
        # The SDK's parameter types are TypedDicts; our messages are plain dicts
        # (the wire format the fake seam also speaks). Casting at this one boundary
        # keeps `ChatMsg` simple everywhere else.
        completion = await self._client.chat.completions.create(
            model=self._config.model,
            messages=cast(Any, messages),
            tools=[tool],
            tool_choice=cast(Any, {"type": "function", "function": {"name": "emit"}}),
            temperature=self._config.temperature,
            extra_body=self._extra(),
        )
        self._record_usage(completion)
        calls = completion.choices[0].message.tool_calls
        call = calls[0] if calls else None
        function = getattr(call, "function", None)
        if function is None:
            raise ValueError("model returned no function tool call for structured output")
        payload = json.loads(function.arguments)
        if not payload:
            raise EmptyToolArguments(
                f"endpoint returned a tool call with empty arguments "
                f"({function.arguments!r}); the answer was lost in transport"
            )
        return schema.model_validate(_unwrap_arguments(payload, schema))


class TeacherClient:
    """Cache-first, retrying, concurrency-capped structured calls with accounting.

    `llm` is injected so tests can pass a fake. `cache` defaults to a real
    on-disk store because resume should not need to be switched on — pass a
    `NullCache` for the parity checks that must genuinely re-ask the model.
    """

    def __init__(
        self,
        config: TeacherConfig,
        *,
        cache: ResponseCache,
        llm: StructuredLLM | None = None,
        prompt_hash: str = "",
    ) -> None:
        self.config = config
        self.cache = cache
        self.prompt_hash = prompt_hash
        self.stats = CallStats()
        self._llm = llm if llm is not None else OpenAIStructuredLLM(config)
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def _sleep(self, attempt: int) -> None:
        window = self.config.base_delay * (2**attempt)
        await asyncio.sleep(_JITTER.uniform(0, window))

    def _key(self, messages: MessageSource, cache_extra: Any) -> str:
        """Content address for one request.

        `sampling` participates because the cache stores answers, and a decoding
        setting that changes the answer must change the address. `reasoning_effort`
        is the case that matters: measured on this endpoint, `max` spends roughly
        1.9x the completion tokens of `low` and returns different gradings, so a
        run that raised it while reading an entry written at the old setting would
        report reasoning it never paid for. `temperature` rides along for the same
        reason.
        """
        request = cache_extra if cache_extra is not None else render_messages(messages)
        sampling = {
            "reasoning_effort": self.config.reasoning_effort,
            "temperature": self.config.temperature,
        }
        # Absent settings are omitted rather than sent as empty, so entries
        # written before this existed still hit at the default configuration.
        active = {name: value for name, value in sampling.items() if value}
        payload: Any = {"request": request, "sampling": active} if active else request
        return cache_key(self.config.model, self.prompt_hash, payload)

    def _record_usage(self) -> None:
        """Fold the provider's token counts for the last call into the run stats."""
        prompt_tokens, completion_tokens = getattr(self._llm, "last_usage", (0, 0))
        self.stats.record_usage(
            prompt_tokens,
            completion_tokens,
            self.config.cost_of(prompt_tokens, completion_tokens),
        )

    async def call(
        self,
        messages: MessageSource,
        schema: type[_T],
        *,
        cache_extra: Any = None,
    ) -> _T:
        """One structured call: cache, then network with retry.

        `cache_extra` joins the key for anything that changes the meaning of a
        request without appearing in `messages` — a per-request nonce makes the
        rendered prompt differ on every call, so the caller passes the stable
        identity (target, sense, texts) instead.

        `messages` may be a callable, in which case it is invoked once per
        attempt. Retrying byte-identical messages at temperature 0 asks a
        deterministic backend the same question again; a caller whose prompt
        carries a nonce should pass the renderer so each attempt differs.
        """
        key = self._key(messages, cache_extra)
        self.stats.calls += 1

        cached = self.cache.get(key)
        if cached is not None:
            try:
                validated = schema.model_validate(cached)
            except ValidationError:
                # A cache entry written under an older schema is stale, not fatal:
                # fall through to the network and let `put` overwrite it.
                pass
            else:
                self.stats.cache_hits += 1
                return validated

        last: Exception | None = None
        for attempt in range(self.config.max_retries):
            async with self._semaphore:
                try:
                    result = await self._llm.parse(render_messages(messages), schema)
                except Exception as exc:  # noqa: BLE001 - retried, then re-raised typed
                    last = exc
                else:
                    self._record_usage()
                    self.cache.put(key, result.model_dump(mode="json"))
                    return result

            self.stats.retries += 1
            if attempt < self.config.max_retries - 1:
                await self._sleep(attempt)

        self.stats.failures += 1
        assert last is not None  # the loop runs at least once: max_retries >= 1
        raise RetryExhausted(self.config.max_retries, last)

    def invalidate(self, messages: MessageSource, *, cache_extra: Any = None) -> None:
        """Forget a structurally decoded response rejected by caller validation."""
        self.cache.delete(self._key(messages, cache_extra))

    async def map(
        self,
        requests: list[tuple[MessageSource, Any]],
        schema: type[_T],
    ) -> list[_T | RetryExhausted]:
        """Run many calls concurrently, returning failures inline rather than raising.

        One unlucky row must not abort a run that has already paid for thousands
        of others, so an exhausted retry comes back as a value the caller can
        quarantine.
        """

        async def run(messages: MessageSource, extra: Any) -> _T | RetryExhausted:
            try:
                return await self.call(messages, schema, cache_extra=extra)
            except RetryExhausted as exc:
                return exc

        return await asyncio.gather(*(run(messages, extra) for messages, extra in requests))


__all__ = [
    "EmptyToolArguments",
    "MessageSource",
    "OpenAIStructuredLLM",
    "RetryExhausted",
    "StructuredLLM",
    "TeacherClient",
    "render_messages",
]
