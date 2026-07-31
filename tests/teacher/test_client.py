"""Client behaviour: cache-first dispatch, retry, concurrency, accounting.

Every test here runs against a fake `StructuredLLM`, so the suite never opens a
socket. The three properties worth protecting are the ones that cost money or
correctness when they break: a cache hit must not reach the network, a transient
failure must be retried rather than dropped, and an exhausted retry must come
back typed so a bulk run can quarantine one row and keep going.
"""

from __future__ import annotations

import asyncio

import pytest

from lexi_research.teacher import (
    CallStats,
    GraderOutput,
    ResponseCache,
    RetryExhausted,
    TeacherClient,
)

from .conftest import FakeLLM, FlakyLLM

# No module-level asyncio mark: `asyncio_mode = "auto"` in pyproject.toml applies
# it to the async tests only, so the synchronous tests here stay unmarked.

MESSAGES = [{"role": "system", "content": "grade"}, {"role": "user", "content": "hi"}]
PAYLOAD = {"correction": "I like it.", "meaning": 4, "feedback": "Good."}


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
        }
    )

    assert config.method == "function_calling"
    assert config.concurrency == 4
    assert config.temperature == 0.0


def test_unknown_cost_reports_zero_rather_than_guessing() -> None:
    from lexi_research.teacher import TeacherConfig

    config = TeacherConfig(base_url="u", api_key="k", model="m")
    assert config.cost_of(10_000, 10_000) == 0.0


def test_client_teacher_client_is_exported() -> None:
    assert TeacherClient.__module__.endswith("teacher.client")
