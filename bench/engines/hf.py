"""The `transformers` baseline. Slowest, simplest, and always available.

Built first on purpose: the harness gets debugged against an engine that cannot
fail for interesting reasons. It is also what makes a vLLM number mean something
— "2.3x faster than what" is otherwise an unanswered question.
"""

from __future__ import annotations

from typing import Any

from .base import Capabilities, EngineError, Launched


class HuggingFaceEngine:
    """In-process generation through `transformers`, exposed as an engine."""

    name = "hf"

    def __init__(self, base_model: str, adapter: str | None = None) -> None:
        self.base_model = base_model
        self.adapter = adapter
        self._model: Any = None
        self._tokenizer: Any = None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            quantisations=frozenset({"bf16", "int4"}),
            supports_lora=True,
            supports_mtp=False,
            supports_prefix_cache=False,
            supports_constrained_decoding=False,
            notes={
                "mtp": "the transformers baseline has no speculative-decoding path here",
                "prefix_cache": "no cross-request prefix cache; this is the point of the baseline",
                "constrained_decoding": "no grammar backend wired into the baseline",
            },
        )

    def launch(self, *, quantisation: str = "bf16", **_: Any) -> Launched:
        from lexi_research.train.trainer import load_model_and_tokenizer

        if quantisation not in self.capabilities().quantisations:
            raise EngineError(f"{self.name} cannot serve {quantisation!r}")
        self._model, self._tokenizer = load_model_and_tokenizer(
            self.base_model, load_in_4bit=quantisation == "int4"
        )
        if self.adapter:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter)
        self._model.eval()
        return Launched(
            base_url="inprocess://hf",
            engine=self.name,
            digest=_version(),
            quantisation=quantisation,
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            raise EngineError("engine has not been launched")
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise EngineError("engine has not been launched")
        return self._tokenizer

    def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"transformers=={version('transformers')}"
    except PackageNotFoundError:  # pragma: no cover - transformers is a smoke-group dep
        return "transformers==unknown"


__all__ = ["HuggingFaceEngine"]
