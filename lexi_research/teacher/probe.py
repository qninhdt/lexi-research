"""One round trip against the configured endpoint, before spending a budget on it.

`python -m lexi_research.teacher.probe` answers the three questions that decide
whether a bulk run is even possible, and answers them for the price of one call:

1. Do the credentials work?
2. Which structured-output mode does this endpoint actually honour? A proxy that
   accepts a strict `json_schema` and then returns loose JSON fails thousands of
   calls in, so it is worth finding out now — `function_calling` is the fallback.
3. How slow is one call? Multiply by the call count to size a run.

It also runs the grader output through `validate_output`, because a teacher whose
`correction` fails check 3 on a trivial sentence cannot label a dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass

from lexi_research.format import BandConfig, ValidationError, default_config_path, validate_output

from .cache import NullCache
from .client import RetryExhausted, TeacherClient
from .registry import prompt_hash, render_grader_prompt
from .schemas import GraderOutput, SenseRef, TeacherConfig

#: A deliberately mundane probe: correct sense, one agreement error, one missing
#: article. A working teacher marks both and returns `meaning` 4.
_TARGET = "bright"
_SENSE = SenseRef(definition="full of light", pos="adjective")
_TEXT = "The room have bright light in morning."


@dataclass(frozen=True)
class ProbeResult:
    """What one probe call established."""

    method: str
    latency_s: float
    output: GraderOutput
    validation: str


async def probe_once(config: TeacherConfig) -> ProbeResult:
    """Grade the fixed probe sentence once, with the cache disabled.

    `NullCache` matters here: a cached answer would make a broken endpoint look
    healthy, which is the one thing this command exists to rule out.
    """
    client = TeacherClient(config, cache=NullCache(), prompt_hash=prompt_hash())
    messages = render_grader_prompt(_TARGET, _SENSE, _TEXT)

    started = time.perf_counter()
    output = await client.call(messages, GraderOutput)
    latency = time.perf_counter() - started

    band_config = BandConfig.from_json(default_config_path())
    result = validate_output(output.model_dump(mode="json"), _TEXT, band_config)
    verdict = (
        f"REJECTED ({result.code}: {result.detail})"
        if isinstance(result, ValidationError)
        else f"ok — grammar {result.bands.grammar}, naturalness {result.bands.naturalness}"
    )
    return ProbeResult(method=config.method, latency_s=latency, output=output, validation=verdict)


async def probe(config: TeacherConfig, *, try_fallback: bool = True) -> ProbeResult:
    """Probe with the configured method, falling back to `function_calling`.

    The fallback is the point of the command: it reports which mode *works*, not
    which mode was requested.
    """
    try:
        return await probe_once(config)
    except (RetryExhausted, ValueError) as exc:
        if not try_fallback or config.method == "function_calling":
            raise
        print(f"  {config.method} failed ({exc}); retrying with function_calling")
        from dataclasses import replace

        return await probe_once(replace(config, method="function_calling"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="override LEXI_TEACHER_MODEL")
    parser.add_argument(
        "--method",
        choices=("json_schema", "function_calling", "json_mode"),
        help="override the structured-output mode to try first",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="fail instead of retrying with function_calling",
    )
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in (("model", args.model), ("method", args.method)) if v}
    try:
        config = TeacherConfig.from_env(dict(os.environ), **overrides)
    except ValueError as exc:
        print(f"config error: {exc}")
        return 2

    print(f"endpoint : {config.base_url}")
    print(f"model    : {config.model}")
    print(f"prompt   : {prompt_hash()[:16]}")

    try:
        result = asyncio.run(probe(config, try_fallback=not args.no_fallback))
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, it does not raise them
        print(f"FAILED: {exc!r}")
        return 1

    print(f"method   : {result.method} (working)")
    print(f"latency  : {result.latency_s:.2f}s")
    print(f"meaning  : {result.output.meaning}")
    print(f"correction: {result.output.correction!r}")
    print(f"feedback : {result.output.feedback!r}")
    print(f"validate : {result.validation}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
