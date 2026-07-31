"""SGLang, nightly and pinned by digest. Same treatment as vLLM.

Kept as its own adapter rather than a flag on the vLLM one: the two disagree on
flag names, on which quantisations they accept, and on what "ready" means, and
folding them together would hide exactly the differences B1 exists to measure.
"""

from __future__ import annotations

import os
from typing import Any

from .base import Capabilities, EngineError, Launched
from .vllm import wait_until_ready

DIGEST_ENV = "LEXI_SGLANG_DIGEST"


class SGLangEngine:
    name = "sglang"

    def __init__(self, base_model: str, adapter: str | None = None, port: int = 30000) -> None:
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
            notes={"mtp": "depends on the checkpoint shipping an MTP head"},
        )

    def launch(
        self,
        *,
        quantisation: str = "bf16",
        prefix_cache: bool = True,
        timeout_s: float = 600.0,
        **_: Any,
    ) -> Launched:
        import shutil
        import subprocess
        import sys

        if shutil.which("python") is None:  # pragma: no cover - defensive
            raise EngineError("no python on PATH to launch sglang with")
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.base_model,
            "--port",
            str(self.port),
        ]
        if quantisation != "bf16":
            command += ["--quantization", quantisation]
        if not prefix_cache:
            command += ["--disable-radix-cache"]
        if self.adapter:
            command += ["--lora-paths", f"lexi={self.adapter}"]

        try:
            self._process = subprocess.Popen(command)
        except OSError as exc:
            raise EngineError(f"could not launch sglang: {exc}") from exc
        base_url = f"http://127.0.0.1:{self.port}/v1"
        wait_until_ready(base_url, timeout_s=timeout_s)
        return Launched(
            base_url=base_url,
            engine=self.name,
            digest=os.environ.get(DIGEST_ENV, "unpinned"),
            quantisation=quantisation,
            extra={"prefix_cache": prefix_cache},
        )

    def shutdown(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=60)
            self._process = None


__all__ = ["DIGEST_ENV", "SGLangEngine"]
