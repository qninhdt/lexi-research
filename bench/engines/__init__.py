"""Engine adapters behind one interface, so B1 compares engines rather than code."""

from .base import Capabilities, Engine, EngineError, Launched, skip_reason
from .hf import HuggingFaceEngine
from .sglang import SGLangEngine
from .vllm import VLLMEngine

ENGINES = {
    "hf": HuggingFaceEngine,
    "vllm": VLLMEngine,
    "sglang": SGLangEngine,
}


def build(name: str, base_model: str, adapter: str | None = None) -> Engine:
    if name not in ENGINES:
        raise EngineError(f"unknown engine {name!r}; expected one of {sorted(ENGINES)}")
    engine: Engine = ENGINES[name](base_model, adapter)
    return engine


__all__ = [
    "ENGINES",
    "Capabilities",
    "Engine",
    "EngineError",
    "HuggingFaceEngine",
    "Launched",
    "SGLangEngine",
    "VLLMEngine",
    "build",
    "skip_reason",
]
