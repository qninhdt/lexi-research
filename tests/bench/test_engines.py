"""Engine adapters: one interface, and capability flags instead of try/except.

An engine that cannot do FP8 must make that arm skipped *and reported as
skipped*. A crash leaves a hole someone fills in with a guess, and a silent
fallback to bf16 puts a bf16 number in the table under an FP8 heading.
"""

from __future__ import annotations

import pytest

from bench.engines import ENGINES, Capabilities, EngineError, build, skip_reason


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_every_engine_satisfies_the_interface(name) -> None:
    engine = build(name, "some/checkpoint")
    assert engine.name == name
    assert isinstance(engine.capabilities(), Capabilities)
    assert callable(engine.launch)
    assert callable(engine.shutdown)


def test_an_unknown_engine_raises() -> None:
    with pytest.raises(EngineError, match="unknown engine"):
        build("tensorrt", "some/checkpoint")


def test_the_baseline_is_always_available() -> None:
    """`hf` is what makes a vLLM number mean something: faster than what?"""
    assert "hf" in ENGINES
    assert "bf16" in build("hf", "x").capabilities().quantisations


def test_the_baseline_admits_what_it_cannot_do() -> None:
    capabilities = build("hf", "x").capabilities()
    assert not capabilities.can("prefix_cache")
    assert "prefix cache" in capabilities.why_not("prefix_cache")


def test_skip_reason_names_the_unsupported_quantisation() -> None:
    reason = skip_reason(build("hf", "x").capabilities(), quantisation="fp8", features=())
    assert reason is not None and "fp8" in reason


def test_skip_reason_names_the_unsupported_feature() -> None:
    reason = skip_reason(build("hf", "x").capabilities(), quantisation="bf16", features=("mtp",))
    assert reason is not None and "speculative" in reason


def test_a_supported_arm_has_no_skip_reason() -> None:
    reason = skip_reason(
        build("vllm", "x").capabilities(), quantisation="fp8", features=("prefix_cache", "lora")
    )
    assert reason is None


def test_launching_an_absent_engine_says_so_rather_than_crashing_obscurely() -> None:
    with pytest.raises(EngineError, match="nightly|not installed"):
        build("vllm", "x").launch()


def test_the_two_nightly_engines_are_separate_adapters() -> None:
    """They disagree on flag names and on what ready means; folding them together
    would hide exactly the differences B1 exists to measure."""
    assert ENGINES["vllm"] is not ENGINES["sglang"]
    assert build("vllm", "x").capabilities().supports_prefix_cache
    assert build("sglang", "x").capabilities().supports_prefix_cache
