"""Render a training example through the chat template and mask the loss to the answer.

Two defects in the old trainer live here. It joined `system\\n\\nuser\\n\\ncompletion`
by hand, so the student learned a format it is never served with — which breaks
the prompt-parity property the parent design calls load-bearing. And it computed
loss over the whole sequence, so most of the gradient came from reproducing a
rubric that is supplied verbatim at inference anyway.

The prompt half comes from `render_grader_prompt`, the same function the serving
shim calls, so parity holds by construction rather than by convention. The mask
is built from token positions — the template's own assistant mask where it
provides one, a token-level prefix comparison where it does not. Decoded text is
never string-matched: a tokenizer that renders one space differently would move
the boundary silently.

Nothing here knows a model. Everything model-specific lives in the tokenizer's
own chat template, which is what makes a different base model a config change.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from lexi_research.teacher import render_grader_prompt
from lexi_research.teacher.schemas import ChatMsg, SenseRef

#: The label value the loss ignores. Fixed by PyTorch's cross-entropy default.
IGNORE_INDEX = -100

#: The three fields the student emits. `grammar` and `naturalness` are derived
#: by code from the correction, so asking the model for them invites drift.
ANSWER_FIELDS = ("correction", "meaning", "feedback")

#: The three arms of ablation A2.
#:
#: `forced-empty` is the arm that makes the other two interpretable. A win for
#: `on` over `off` could mean reasoning helps, or it could mean the `<think>`
#: scaffold alone changes the distribution the answer is sampled from; training
#: the model to emit an empty block separates those.
THINKING_MODES = ("on", "off", "forced-empty")

#: What `forced-empty` supervises before the answer.
EMPTY_REASONING = "<think>\n\n</think>\n\n"


class CollationError(ValueError):
    """The template, the mask, or the length made an example unusable."""


class SequenceTooLong(CollationError):
    """The rendered example exceeds `max_seq_len`.

    Raised rather than truncated. Cutting the tail silently teaches the model to
    emit unterminated JSON; cutting the head silently drops the rubric the answer
    depends on. The caller drops the row and counts it instead.
    """


class ChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def encode(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Example:
    """One tokenised training example."""

    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.attention_mask:
            object.__setattr__(self, "attention_mask", [1] * len(self.input_ids))

    @property
    def supervised_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


def training_messages(row: Mapping[str, Any], *, nonce: str | None = None) -> list[ChatMsg]:
    """The prompt, built by the one function serving and generation also call.

    `nonce` is left unset in production so each example carries a fresh
    delimiter, matching the distribution inference draws from; it is injectable
    only so tests can assert byte-identical output.
    """
    return render_grader_prompt(
        str(row["target"]),
        SenseRef(definition=str(row["definition"]), pos=str(row["pos"])),
        str(row["text"]),
        nonce=nonce,
    )


def completion_text(row: Mapping[str, Any]) -> str:
    """The gold answer, serialised exactly as the student must emit it."""
    return json.dumps(
        {key: row[key] for key in ANSWER_FIELDS},
        ensure_ascii=False,
        sort_keys=True,
    )


def _ids(result: Any) -> list[int]:
    """`apply_chat_template` returns a list, or a mapping when asked for one."""
    if isinstance(result, Mapping):
        result = result["input_ids"]
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise CollationError(f"chat template produced {type(result).__name__}, not token ids")
    return [int(token) for token in result]


#: The marker a template needs to report which tokens the assistant produced.
#: Matched exactly as `transformers.utils.chat_template_utils` matches it: asked
#: for an assistant mask without this block, it logs a warning and returns a mask
#: of all zeros rather than raising, so absence has to be detected up front.
#: Most published templates — Qwen, Llama, Mistral, Gemma — do not carry it.
_GENERATION_BLOCK = re.compile(r"\{\%-?\s*generation\s*-?\%\}")


def _marks_assistant_tokens(tokenizer: ChatTokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return bool(template) and bool(_GENERATION_BLOCK.search(str(template)))


def _mask_from_template(
    tokenizer: ChatTokenizer,
    conversation: list[ChatMsg],
    *,
    enable_thinking: bool,
) -> tuple[list[int], list[int]] | None:
    """Token ids and assistant mask, or None if the template did not supply one."""
    try:
        encoded = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            enable_thinking=enable_thinking,
        )
    except (TypeError, ValueError, KeyError):
        return None
    if not isinstance(encoded, Mapping) or "assistant_masks" not in encoded:
        return None

    ids = _ids(encoded)
    mask = [int(flag) for flag in encoded["assistant_masks"]]
    if len(mask) != len(ids):
        raise CollationError(
            f"assistant mask covers {len(mask)} tokens but the render is {len(ids)}"
        )
    if not any(mask):
        # The template carries a generation block and still marked nothing, so
        # the block is misplaced. Falling back would hide a broken template.
        raise CollationError(
            "the chat template has a generation block but marked no assistant "
            "tokens; the block does not cover the assistant turn"
        )
    return ids, mask


def _encode_plain(tokenizer: ChatTokenizer, text: str) -> list[int]:
    """Tokenise raw text, adding none of the tokenizer's own framing."""
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def _supervised_text(row: Mapping[str, Any], thinking: str) -> str:
    """What the assistant turn must contain, for this arm of A2."""
    if thinking not in THINKING_MODES:
        raise CollationError(f"thinking={thinking!r}; expected one of {list(THINKING_MODES)}")
    if thinking == "forced-empty":
        return EMPTY_REASONING + completion_text(row)
    return completion_text(row)


def _mask_from_concatenation(
    tokenizer: ChatTokenizer,
    messages: list[ChatMsg],
    row: Mapping[str, Any],
    *,
    enable_thinking: bool,
    thinking: str,
) -> tuple[list[int], list[int]]:
    """Fallback: build the sequence as prompt + answer, so the boundary is exact.

    Comparing a prompt render against a full-conversation render would be the
    other option, and it is unsound: a template is free to render an assistant
    turn differently from the generation prompt it emits for the same turn. Qwen's
    does exactly that under `enable_thinking=false` — it inserts an empty reasoning
    block into the generation prompt only — so a prefix comparison would fail on
    every row of that ablation arm.

    Rendering only the prompt half through the template keeps parity with what
    serving sends, which is the property that matters; the answer is appended and
    terminated with the tokenizer's own end-of-turn token.
    """
    prompt = _ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    )
    terminator = str(getattr(tokenizer, "eos_token", "") or "")
    if not terminator:
        raise CollationError(
            "the tokenizer declares no eos_token, so the answer cannot be terminated"
        )
    completion = _encode_plain(tokenizer, _supervised_text(row, thinking) + terminator)
    if not completion:
        raise CollationError("the answer tokenised to nothing")
    return prompt + completion, [0] * len(prompt) + [1] * len(completion)


def build_example(
    tokenizer: ChatTokenizer,
    row: Mapping[str, Any],
    *,
    thinking: str = "on",
    completion_only: bool = True,
    max_seq_len: int | None = None,
    nonce: str | None = None,
) -> Example:
    """Tokenise one row into `input_ids` plus completion-masked `labels`.

    Two paths, chosen by what the tokenizer's own template can do: its assistant
    mask where it provides one, otherwise prompt-plus-answer concatenation. Both
    locate the boundary by token position; decoded text is never string-matched,
    because a tokenizer that renders one space differently would move the
    boundary silently.

    `thinking` is ablation A2. `on` and `forced-empty` both open the template's
    reasoning path — the difference is that `forced-empty` supervises an empty
    block — while `off` asks the template not to open one at all.

    `completion_only=False` supervises the whole sequence — the old behaviour,
    kept as the other arm of the loss-mask ablation rather than as a fallback.
    """
    if thinking not in THINKING_MODES:
        raise CollationError(f"thinking={thinking!r}; expected one of {list(THINKING_MODES)}")
    enable_thinking = thinking != "off"
    messages = training_messages(row, nonce=nonce)

    marked = None
    if _marks_assistant_tokens(tokenizer):
        conversation: list[ChatMsg] = [
            *messages,
            {"role": "assistant", "content": _supervised_text(row, thinking)},
        ]
        marked = _mask_from_template(tokenizer, conversation, enable_thinking=enable_thinking)
    if marked is None:
        ids, mask = _mask_from_concatenation(
            tokenizer, messages, row, enable_thinking=enable_thinking, thinking=thinking
        )
    else:
        ids, mask = marked

    if max_seq_len is not None and len(ids) > max_seq_len:
        raise SequenceTooLong(f"example renders to {len(ids)} tokens, over the {max_seq_len} limit")

    if completion_only:
        labels = [token if flag else IGNORE_INDEX for token, flag in zip(ids, mask, strict=True)]
    else:
        labels = list(ids)
    return Example(input_ids=ids, labels=labels)


__all__ = [
    "ANSWER_FIELDS",
    "EMPTY_REASONING",
    "IGNORE_INDEX",
    "THINKING_MODES",
    "ChatTokenizer",
    "CollationError",
    "Example",
    "SequenceTooLong",
    "build_example",
    "completion_text",
    "training_messages",
]
