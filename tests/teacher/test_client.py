"""Client behaviour: cache-first dispatch, retry, concurrency, accounting.

Every test here runs against a fake `StructuredLLM`, so the suite never opens a
socket. The three properties worth protecting are the ones that cost money or
correctness when they break: a cache hit must not reach the network, a transient
failure must be retried rather than dropped, and an exhausted retry must come
back typed so a bulk run can quarantine one row and keep going.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from lexi_research.teacher import (
    CallStats,
    GraderOutput,
    LangChainStructuredLLM,
    ResponseCache,
    RetryExhausted,
    TeacherClient,
)

from .conftest import FakeLLM, FlakyLLM

# No module-level asyncio mark: `asyncio_mode = "auto"` in pyproject.toml applies
# it to the async tests only, so the synchronous tests here stay unmarked.

MESSAGES = [{"role": "system", "content": "grade"}, {"role": "user", "content": "hi"}]
PAYLOAD = {"correction": "I like it.", "meaning": 4, "feedback": "Good."}


async def test_langchain_adapter_uses_json_mode_and_records_usage(monkeypatch, config) -> None:
    calls: dict[str, Any] = {}

    class FakeChain:
        async def ainvoke(self, messages):
            calls["messages"] = messages
            return {
                "raw": SimpleNamespace(
                    usage_metadata={"input_tokens": 12, "output_tokens": 8},
                    response_metadata={},
                ),
                "parsed": GraderOutput.model_validate(PAYLOAD),
                "parsing_error": None,
            }

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def with_structured_output(self, schema, **kwargs):
            calls["schema"] = schema
            calls["options"] = kwargs
            return FakeChain()

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    client = LangChainStructuredLLM(config)
    result = await client.parse(MESSAGES, GraderOutput)

    assert result.meaning == 4
    assert calls["schema"] is GraderOutput
    assert calls["options"] == {"method": "json_mode", "include_raw": True}
    assert calls["init"]["max_retries"] == 0
    assert [message.type for message in calls["messages"]] == ["system", "human"]
    assert client.last_usage == (12, 8)


async def test_langchain_adapter_recovers_wrapped_tool_arguments(monkeypatch, config) -> None:
    class FakeChain:
        async def ainvoke(self, messages):
            del messages
            wrapped = {f"__{key}": value for key, value in PAYLOAD.items()}
            return {
                "raw": SimpleNamespace(
                    usage_metadata={"input_tokens": 12, "output_tokens": 8},
                    response_metadata={},
                    tool_calls=[{"args": wrapped}],
                ),
                "parsed": None,
                "parsing_error": ValueError("wrapped tool arguments"),
            }

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            del kwargs

        def with_structured_output(self, schema, **kwargs):
            del schema, kwargs
            return FakeChain()

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    client = LangChainStructuredLLM(replace(config, method="function_calling"))
    result = await client.parse(MESSAGES, GraderOutput)

    assert result.meaning == 4
    assert client.last_usage == (12, 8)


async def test_langchain_adapter_recovers_malformed_json_mode_output(monkeypatch, config) -> None:
    class FakeChain:
        async def ainvoke(self, messages):
            del messages
            # Malformed JSON: trailing comma and missing closing brace
            broken_text = '{"correction": "I like it.", "meaning": 4, "feedback": "Good.",'
            return {
                "raw": SimpleNamespace(
                    content=broken_text,
                    usage_metadata={"input_tokens": 10, "output_tokens": 15},
                    response_metadata={},
                ),
                "parsed": None,
                "parsing_error": ValueError("Invalid JSON"),
            }

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            del kwargs

        def with_structured_output(self, schema, **kwargs):
            del schema, kwargs
            return FakeChain()

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    client = LangChainStructuredLLM(replace(config, method="json_mode"))
    result = await client.parse(MESSAGES, GraderOutput)

    assert result.meaning == 4
    assert result.correction == "I like it."
    assert result.feedback == "Good."
    assert client.last_usage == (10, 15)


async def test_langchain_adapter_recovers_malformed_tool_args(
    monkeypatch, config
) -> None:
    class FakeChain:
        async def ainvoke(self, messages):
            del messages
            # Tool call args as a broken JSON string
            broken_str = '{"correction": "I like it.", "meaning": 4, "feedback": "Good.",}'
            return {
                "raw": SimpleNamespace(
                    usage_metadata={"input_tokens": 12, "output_tokens": 8},
                    response_metadata={},
                    tool_calls=[{"args": broken_str}],
                ),
                "parsed": None,
                "parsing_error": ValueError("JSON decode error"),
            }

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            del kwargs

        def with_structured_output(self, schema, **kwargs):
            del schema, kwargs
            return FakeChain()

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    client = LangChainStructuredLLM(replace(config, method="function_calling"))
    result = await client.parse(MESSAGES, GraderOutput)

    assert result.meaning == 4
    assert result.correction == "I like it."
    assert client.last_usage == (12, 8)


async def test_call_returns_parsed_schema(client_factory, config) -> None:
    client = client_factory(FakeLLM(PAYLOAD))
    result = await client.call(MESSAGES, GraderOutput)

    assert isinstance(result, GraderOutput)
    assert result.meaning == 4
    assert client.stats.calls == 1
    assert client.stats.network_calls == 1


async def test_cache_hit_avoids_the_network(client_factory, tmp_path) -> None:
    """The resume property: a repeated request must issue zero network calls."""
    cache = ResponseCache(tmp_path / "cache")

    first_llm = FakeLLM(PAYLOAD)
    first = client_factory(first_llm, cache=cache)
    await first.call(MESSAGES, GraderOutput)
    assert first_llm.calls == 1

    # A fresh client over the same cache stands in for a restarted process.
    second_llm = FakeLLM(PAYLOAD)
    second = client_factory(second_llm, cache=cache)
    result = await second.call(MESSAGES, GraderOutput)

    assert second_llm.calls == 0
    assert result.meaning == 4
    assert second.stats.cache_hits == 1
    assert second.stats.network_calls == 0


async def test_cache_key_ignores_the_nonce_via_cache_extra(client_factory, tmp_path) -> None:
    """A fresh nonce per request must not defeat the cache.

    The rendered prompt differs on every call, so callers key on the stable
    request identity instead. Without this, resume would never hit.
    """
    cache = ResponseCache(tmp_path / "cache")
    identity = {"target": "bright", "text": "The room is bright."}

    llm = FakeLLM(PAYLOAD)
    client = client_factory(llm, cache=cache)
    await client.call(
        [{"role": "user", "content": "nonce-aaaa"}], GraderOutput, cache_extra=identity
    )
    await client.call(
        [{"role": "user", "content": "nonce-bbbb"}], GraderOutput, cache_extra=identity
    )

    assert llm.calls == 1
    assert client.stats.cache_hits == 1


async def test_stale_cache_entry_falls_through_to_the_network(client_factory, tmp_path) -> None:
    """An entry written under an older schema is stale, not fatal."""
    cache = ResponseCache(tmp_path / "cache")
    llm = FakeLLM(PAYLOAD)
    client = client_factory(llm, cache=cache)

    from lexi_research.teacher import cache_key

    key = cache_key(client.config.model, client.prompt_hash, MESSAGES)
    cache.put(key, {"legacy_field": "no longer valid"})

    result = await client.call(MESSAGES, GraderOutput)

    assert llm.calls == 1
    assert result.meaning == 4
    assert client.stats.cache_hits == 0


async def test_transient_failure_is_retried(client_factory) -> None:
    llm = FlakyLLM(PAYLOAD, failures=2)
    client = client_factory(llm)

    result = await client.call(MESSAGES, GraderOutput)

    assert result.meaning == 4
    assert llm.calls == 3
    assert client.stats.retries == 2
    assert client.stats.failures == 0


async def test_failed_attempt_usage_is_accounted(client_factory) -> None:
    llm = FlakyLLM(PAYLOAD, failures=2, usage=(1_000, 500))
    client = client_factory(
        llm,
        prompt_cost_per_mtok=3.0,
        completion_cost_per_mtok=15.0,
    )

    await client.call(MESSAGES, GraderOutput)

    assert client.stats.prompt_tokens == 3_000
    assert client.stats.completion_tokens == 1_500
    assert client.stats.cost == pytest.approx(0.0315)


async def test_retry_exhaustion_raises_typed_error(client_factory) -> None:
    """Exhaustion must be distinguishable from a provider exception."""
    llm = FlakyLLM(PAYLOAD, failures=99)
    client = client_factory(llm)

    with pytest.raises(RetryExhausted) as excinfo:
        await client.call(MESSAGES, GraderOutput)

    assert excinfo.value.attempts == client.config.max_retries
    assert isinstance(excinfo.value.cause, RuntimeError)
    assert llm.calls == client.config.max_retries
    assert client.stats.failures == 1


async def test_map_returns_failures_inline(client_factory) -> None:
    """One unlucky row must not abort a run that already paid for the others."""
    llm = FlakyLLM(PAYLOAD, failures=0, fail_on_text="poison")
    client = client_factory(llm)

    requests = [
        ([{"role": "user", "content": "fine"}], {"id": 1}),
        ([{"role": "user", "content": "poison"}], {"id": 2}),
        ([{"role": "user", "content": "also fine"}], {"id": 3}),
    ]
    results = await client.map(requests, GraderOutput)

    assert isinstance(results[0], GraderOutput)
    assert isinstance(results[1], RetryExhausted)
    assert isinstance(results[2], GraderOutput)


async def test_concurrency_cap_is_respected(client_factory, tmp_path) -> None:
    """The semaphore is what keeps a fan-out from becoming a 429 storm."""
    cap = 3
    llm = FakeLLM(PAYLOAD, delay=0.01)
    client = client_factory(llm, concurrency=cap)

    requests = [([{"role": "user", "content": f"s{i}"}], {"id": i}) for i in range(12)]
    await client.map(requests, GraderOutput)

    assert llm.max_in_flight <= cap
    # Without the cap all twelve would overlap; assert the cap actually bit.
    assert llm.max_in_flight > 1


async def test_cost_and_token_accounting(client_factory) -> None:
    llm = FakeLLM(PAYLOAD, usage=(1_000, 500))
    client = client_factory(
        llm,
        prompt_cost_per_mtok=3.0,
        completion_cost_per_mtok=15.0,
    )

    await client.call(MESSAGES, GraderOutput)

    assert client.stats.prompt_tokens == 1_000
    assert client.stats.completion_tokens == 500
    # 1000 * 3 / 1e6 + 500 * 15 / 1e6
    assert client.stats.cost == pytest.approx(0.0105)


async def test_cached_call_does_not_double_count_cost(client_factory, tmp_path) -> None:
    """A resumed run's reported spend must reflect what was actually paid."""
    cache = ResponseCache(tmp_path / "cache")
    llm = FakeLLM(PAYLOAD, usage=(1_000, 500))
    client = client_factory(
        llm, cache=cache, prompt_cost_per_mtok=3.0, completion_cost_per_mtok=15.0
    )

    await client.call(MESSAGES, GraderOutput)
    await client.call(MESSAGES, GraderOutput)

    assert client.stats.calls == 2
    assert client.stats.network_calls == 1
    assert client.stats.prompt_tokens == 1_000
    assert client.stats.cost == pytest.approx(0.0105)


def test_stats_report_shape() -> None:
    stats = CallStats(calls=10, cache_hits=4)
    stats.record_usage(100, 50, 0.5)
    report = stats.as_dict()

    assert report["network_calls"] == 6
    assert report["hit_rate"] == 0.4
    assert report["prompt_tokens"] == 100
    assert report["cost"] == 0.5


def test_empty_stats_hit_rate_is_zero_not_an_error() -> None:
    assert CallStats().hit_rate == 0.0


async def test_client_is_reusable_across_event_loop_calls(client_factory) -> None:
    """Sanity: the semaphore is created per client, not per call."""
    client = client_factory(FakeLLM(PAYLOAD))
    await asyncio.gather(
        client.call([{"role": "user", "content": "a"}], GraderOutput, cache_extra={"i": 1}),
        client.call([{"role": "user", "content": "b"}], GraderOutput, cache_extra={"i": 2}),
    )
    assert client.stats.calls == 2


def test_teacher_config_requires_credentials(monkeypatch) -> None:
    from lexi_research.teacher import TeacherConfig

    with pytest.raises(ValueError, match="LEXI_TEACHER_BASE_URL"):
        TeacherConfig.from_env({})


def test_teacher_config_from_env_reads_optionals() -> None:
    from lexi_research.teacher import TeacherConfig

    config = TeacherConfig.from_env(
        {
            "LEXI_TEACHER_BASE_URL": "https://example.invalid/v1",
            "LEXI_TEACHER_API_KEY": "sk-test",
            "LEXI_TEACHER_MODEL": "teacher-1",
            "LEXI_TEACHER_METHOD": "function_calling",
            "LEXI_TEACHER_CONCURRENCY": "4",
            "LEXI_TEACHER_MAX_RETRIES": "6",
            "LEXI_TEACHER_BASE_DELAY": "0.25",
            "LEXI_TEACHER_PROMPT_COST_PER_MTOK": "3.0",
            "LEXI_TEACHER_COMPLETION_COST_PER_MTOK": "15.0",
        }
    )

    assert config.method == "function_calling"
    assert config.concurrency == 4
    assert config.max_retries == 6
    assert config.base_delay == 0.25
    assert config.prompt_cost_per_mtok == 3.0
    assert config.completion_cost_per_mtok == 15.0
    assert config.temperature == 0.0


def test_teacher_config_rejects_unknown_structured_method() -> None:
    from lexi_research.teacher import TeacherConfig

    with pytest.raises(ValueError, match="unknown structured-output method"):
        TeacherConfig(base_url="u", api_key="k", model="m", method="text")


def test_unknown_cost_reports_zero_rather_than_guessing() -> None:
    from lexi_research.teacher import TeacherConfig

    config = TeacherConfig(base_url="u", api_key="k", model="m")
    assert config.cost_of(10_000, 10_000) == 0.0


class TestSamplingInTheCacheKey:
    """A decoding setting that changes the answer must change the address.

    Measured on the configured endpoint, `reasoning_effort: max` spends about
    1.9x the completion tokens of `low` and returns different gradings. A run
    that raised the setting and then read an entry written at the old one would
    report reasoning it never paid for.
    """

    async def test_raising_reasoning_effort_misses_the_cache(
        self, client_factory, tmp_path
    ) -> None:
        from lexi_research.teacher import ResponseCache

        cache = ResponseCache(tmp_path / "cache")
        first = client_factory(FakeLLM(PAYLOAD), cache=cache, reasoning_effort="low")
        await first.call(MESSAGES, GraderOutput, cache_extra={"row": 1})

        hotter_llm = FakeLLM(PAYLOAD)
        hotter = client_factory(hotter_llm, cache=cache, reasoning_effort="max")
        await hotter.call(MESSAGES, GraderOutput, cache_extra={"row": 1})

        assert hotter_llm.calls == 1, "a different effort must reach the provider"
        assert hotter.stats.cache_hits == 0

    async def test_the_same_effort_still_hits(self, client_factory, tmp_path) -> None:
        from lexi_research.teacher import ResponseCache

        cache = ResponseCache(tmp_path / "cache")
        first = client_factory(FakeLLM(PAYLOAD), cache=cache, reasoning_effort="max")
        await first.call(MESSAGES, GraderOutput, cache_extra={"row": 1})

        second_llm = FakeLLM(PAYLOAD)
        second = client_factory(second_llm, cache=cache, reasoning_effort="max")
        await second.call(MESSAGES, GraderOutput, cache_extra={"row": 1})

        assert second_llm.calls == 0
        assert second.stats.cache_hits == 1

    async def test_temperature_also_participates(self, client_factory, tmp_path) -> None:
        from lexi_research.teacher import ResponseCache

        cache = ResponseCache(tmp_path / "cache")
        cold = client_factory(FakeLLM(PAYLOAD), cache=cache, temperature=0.0)
        await cold.call(MESSAGES, GraderOutput, cache_extra={"row": 1})

        hot_llm = FakeLLM(PAYLOAD)
        hot = client_factory(hot_llm, cache=cache, temperature=0.7)
        await hot.call(MESSAGES, GraderOutput, cache_extra={"row": 1})

        assert hot_llm.calls == 1

    async def test_entries_written_before_this_existed_still_hit(
        self, client_factory, tmp_path
    ) -> None:
        """At the default configuration the key is the bare request, as before.

        Otherwise adding this would have silently orphaned every paid entry in
        the cache on disk.
        """
        from lexi_research.teacher import ResponseCache, cache_key

        cache = ResponseCache(tmp_path / "cache")
        legacy = cache_key("test-model", "test-prompt-hash", {"row": 1})
        cache.put(legacy, PAYLOAD)

        llm = FakeLLM(PAYLOAD)
        client = client_factory(llm, cache=cache)
        result = await client.call(MESSAGES, GraderOutput, cache_extra={"row": 1})

        assert llm.calls == 0
        assert result.meaning == 4


def test_client_teacher_client_is_exported() -> None:
    assert TeacherClient.__module__.endswith("teacher.client")


class TestMessageSource:
    """A retry must be able to change the request, not just repeat it.

    Measured against one proxy: roughly 7% of gradings return empty tool
    arguments, and re-asking with byte-identical messages left a residue of rows
    that failed every attempt, while re-rendering the prompt recovered all of
    them. That makes "each attempt renders again" a property worth pinning.
    """

    async def test_a_callable_is_rendered_once_per_attempt(self, client_factory) -> None:
        llm = FlakyLLM(PAYLOAD, failures=2)
        client = client_factory(llm)
        rendered = 0

        def messages() -> list[dict[str, str]]:
            nonlocal rendered
            rendered += 1
            return [{"role": "user", "content": f"attempt {rendered}"}]

        result = await client.call(messages, GraderOutput, cache_extra={"row": 1})

        assert result.meaning == 4
        assert llm.calls == 3
        assert rendered == 3
        # Each attempt genuinely differed, which is the whole point.
        assert [msg[0]["content"] for msg in llm.messages] == [
            "attempt 1",
            "attempt 2",
            "attempt 3",
        ]

    async def test_a_plain_list_still_works(self, client_factory) -> None:
        """The list form stays valid: most callers have no nonce to refresh."""
        llm = FakeLLM(PAYLOAD)
        client = client_factory(llm)

        result = await client.call(MESSAGES, GraderOutput)

        assert result.meaning == 4
        assert llm.messages == [MESSAGES]

    async def test_a_cache_hit_never_renders(self, client_factory, tmp_path) -> None:
        """Rendering is cheap but a nonce-bearing prompt must not key the cache."""
        from lexi_research.teacher import ResponseCache

        cache = ResponseCache(tmp_path / "cache")
        calls = 0

        def messages() -> list[dict[str, str]]:
            nonlocal calls
            calls += 1
            return [{"role": "user", "content": f"nonce {calls}"}]

        first = client_factory(FakeLLM(PAYLOAD), cache=cache)
        await first.call(messages, GraderOutput, cache_extra={"row": 7})

        second_llm = FakeLLM(PAYLOAD)
        second = client_factory(second_llm, cache=cache)
        result = await second.call(messages, GraderOutput, cache_extra={"row": 7})

        assert second_llm.calls == 0
        assert result.meaning == 4
        # One render for the paid call; the cache hit needed none.
        assert calls == 1


class TestToolArgumentRecovery:
    """Shapes this proxy actually returned, and what the client does with them."""

    def _unwrap(self, payload: dict) -> dict:
        from lexi_research.teacher.client import _unwrap_arguments

        return _unwrap_arguments(payload, GraderOutput)

    def test_a_well_formed_payload_is_untouched(self) -> None:
        assert self._unwrap(PAYLOAD) == PAYLOAD

    def test_underscore_prefixed_fields_are_stripped(self) -> None:
        """Observed: `{"__correction": ..., "__meaning": 4, "__feedback": ...}`."""
        wrapped = {f"__{key}": value for key, value in PAYLOAD.items()}
        assert self._unwrap(wrapped) == PAYLOAD

    def test_a_single_placeholder_wrapper_is_unnested(self) -> None:
        """Observed: the schema nested under `$PARAMETER_NAME` or `corrections`."""
        for key in ("$PARAMETER_NAME", "corrections"):
            assert self._unwrap({key: PAYLOAD}) == PAYLOAD

    def test_an_unrecognised_shape_is_passed_through_to_fail_validation(self) -> None:
        """A genuinely wrong payload must still be rejected, not coerced."""
        odd = {"verdict": "looks fine", "score": 9}
        assert self._unwrap(odd) == odd

    def test_two_unknown_keys_are_not_unnested(self) -> None:
        """Unnesting only one key keeps the rule from guessing at real ambiguity."""
        odd = {"a": PAYLOAD, "b": PAYLOAD}
        assert self._unwrap(odd) == odd

    def test_empty_arguments_are_named_rather_than_a_schema_error(self) -> None:
        """`{}` with `finish_reason: tool_calls` is transport loss, not a bad answer."""
        from lexi_research.teacher import EmptyToolArguments

        assert issubclass(EmptyToolArguments, ValueError)
