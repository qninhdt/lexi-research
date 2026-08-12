"""Teacher-side machinery: the client, the resume cache, and the prompt registry.

The prompt registry is the important export. `render_grader_prompt` is the single
code path that builds the grading prompt, and call 2, the training-data builder,
the eval harness, and the serving shim all go through it. That identity is what
makes this distillation rather than training on data a teacher happened to emit —
if generation and inference could render the prompt differently, the student would
learn a function it is never asked to compute.
"""

from .cache import NullCache, ResponseCache, cache_key
from .client import (
    EmptyToolArguments,
    LangChainStructuredLLM,
    MessageSource,
    OpenAIStructuredLLM,
    RetryExhausted,
    StructuredLLM,
    TeacherClient,
    render_messages,
)
from .registry import (
    GRADER_TEMPLATES,
    PROMPTS_DIR,
    prompt_hash,
    render_diversify_prompt,
    render_grader_prompt,
    template_names,
)
from .schemas import (
    STRUCTURED_METHODS,
    CallStats,
    ChatMsg,
    DiversifiedSentence,
    DiversifyBatch,
    DiversifySpec,
    GraderOutput,
    SenseRef,
    TeacherConfig,
)

__all__ = [
    "GRADER_TEMPLATES",
    "PROMPTS_DIR",
    "CallStats",
    "ChatMsg",
    "DiversifiedSentence",
    "DiversifyBatch",
    "DiversifySpec",
    "EmptyToolArguments",
    "GraderOutput",
    "LangChainStructuredLLM",
    "MessageSource",
    "NullCache",
    "OpenAIStructuredLLM",
    "ResponseCache",
    "RetryExhausted",
    "SenseRef",
    "STRUCTURED_METHODS",
    "StructuredLLM",
    "TeacherClient",
    "TeacherConfig",
    "cache_key",
    "prompt_hash",
    "render_diversify_prompt",
    "render_messages",
    "render_grader_prompt",
    "template_names",
]
