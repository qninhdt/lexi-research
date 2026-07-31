"""vLLM, nightly and pinned by digest.

The digest is recorded in every report rather than assumed from a version string:
these builds are nightly, the model support that matters here lands in them
before it lands in a release, and "it worked last Tuesday" is not a provenance
record. Breakage mid-sweep is expected, and debugging it is the exercise rather
than an obstacle to it.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import Capabilities, EngineError, Launched

#: Read from the environment so a pinned digest is recorded without a code edit.
DIGEST_ENV = "LEXI_VLLM_DIGEST"


class VLLMEngine:
    name = "vllm"

    def __init__(self, base_model: str, adapter: str | None = None, port: int = 8000) -> None:
        self.base_model = base_model
        self.adapter = adapter
        self.port = port
        self._process: Any = None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            quantisations=frozenset({"bf16", "fp8", "awq", "gptq"}),
            supports_lora=True,
            supports_mtp=True,
            supports_prefix_cache=True,
            supports_constrained_decoding=True,
        )

    def launch(
        self,
        *,
        quantisation: str = "bf16",
        prefix_cache: bool = True,
        speculative: str | None = None,
        timeout_s: float = 600.0,
        **_: Any,
    ) -> Launched:
        import shutil
        import subprocess

        binary = shutil.which("vllm")
        if binary is None:
            raise EngineError(
                "vllm is not installed. It is a nightly build in this project; "
                "install it and pin the digest in LEXI_VLLM_DIGEST."
            )
        command = [
            binary,
            "serve",
            self.base_model,
            "--port",
            str(self.port),
            "--served-model-name",
            "lexi",
        ]
        if quantisation != "bf16":
            command += ["--quantization", quantisation]
        if not prefix_cache:
            command += ["--no-enable-prefix-caching"]
        if speculative:
            command += ["--speculative-model", speculative]
        if self.adapter:
            command += ["--enable-lora", "--lora-modules", f"lexi={self.adapter}"]

        self._process = subprocess.Popen(command)
        base_url = f"http://127.0.0.1:{self.port}/v1"
        wait_until_ready(base_url, timeout_s=timeout_s)
        return Launched(
            base_url=base_url,
            engine=self.name,
            digest=os.environ.get(DIGEST_ENV, "unpinned"),
            quantisation=quantisation,
            extra={"prefix_cache": prefix_cache, "speculative": speculative},
        )

    def shutdown(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=60)
            self._process = None


def wait_until_ready(base_url: str, *, timeout_s: float, poll_s: float = 2.0) -> None:
    """Block until the server answers, or say how long it was given.

    A benchmark that starts before the engine finished loading measures the load,
    which is how a cold-start artifact becomes a throughput number.
    """
    deadline = time.monotonic() + timeout_s
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(poll_s)
    raise EngineError(f"engine at {base_url} was not ready within {timeout_s}s: {last}")


__all__ = ["DIGEST_ENV", "VLLMEngine", "wait_until_ready"]
