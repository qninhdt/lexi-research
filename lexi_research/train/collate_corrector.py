"""Tokenise a stage-A row: the corrector prompt, and the correction as the answer.

Separate from `collate.py` rather than a flag on it. That module renders the
grader prompt through `render_grader_prompt` and supervises a three-field JSON
object, and both of those are the inference contract — the reason a test asserts
only the registry loads the grader templates. Threading a "no sense, one field"
mode through it would put a training-only branch inside the one code path whose
job is to be identical to serving.

What is shared is the part that must not diverge: the mask is still built from
token positions via `_mask_from_template` and `_mask_from_concatenation`, so a
tokenizer that renders a space differently cannot move the boundary here while
leaving it correct there.

The answer is the correction string itself, not JSON. Stage A has one field, and
wrapping it in an object would teach the model a frame it must then unlearn in
stage B — where the object has three fields in `sort_keys=True` order. A bare
string keeps the two stages differing in what they contain rather than in how
they are shaped.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lexi_research.teacher.schemas import ChatMsg

from .collate import (
    EMPTY_REASONING,
    IGNORE_INDEX,
    THINKING_MODES,
    ChatTokenizer,
    CollationError,
    Example,
    SequenceTooLong,
    _encode_plain,
    _marks_assistant_tokens,
    _mask_from_template,
)
from .corrector_prompt import render_corrector_prompt


def corrector_messages(
    row: Mapping[str, Any], *, nonce: str | None = None, rubric: str = "full"
) -> list[ChatMsg]:
    """The stage-A prompt for one row."""
    return render_corrector_prompt(str(row["text"]), nonce=nonce, rubric=rubric)


def corrector_answer(row: Mapping[str, Any]) -> str:
    """The gold answer: the correction string, or `null` for an unreadable sentence.

    The literal `null` matches what the grader emits in the same situation, so
    the two stages agree on how "this cannot be corrected" is spelled.
    """
    correction = row.get("correction")
    if correction is None:
        return "null"
    if not isinstance(correction, str):
        raise CollationError(f"correction is {type(correction).__name__}, not a string")
    return correction


def _supervised_text(row: Mapping[str, Any], thinking: str) -> str:
    if thinking not in THINKING_MODES:
        raise CollationError(f"thinking={thinking!r}; expected one of {list(THINKING_MODES)}")
    if thinking == "forced-empty":
        return EMPTY_REASONING + corrector_answer(row)
    return corrector_answer(row)


def build_corrector_example(
    tokenizer: ChatTokenizer,
    row: Mapping[str, Any],
    *,
    thinking: str = "off",
    max_seq_len: int | None = None,
    nonce: str | None = None,
    rubric: str = "full",
) -> Example:
    """Tokenise one stage-A row into `input_ids` plus a completion-masked `labels`.

    `thinking` defaults to `off` here, unlike stage B. Stage A has no reasoning to
    learn: the answer is a mechanical re-emission of the input with markup, and a
    reasoning block would be tokens spent on every one of ~20k rows for a step the
    task does not contain.

    There is no `completion_only=False` arm. That arm exists in stage B to measure
    what supervising the prompt costs; the answer here is already the majority of
    a much shorter sequence, so the same ablation would be measuring something
    else under the same name.
    """
    messages = corrector_messages(row, nonce=nonce, rubric=rubric)
    enable_thinking = thinking != "off"
    answer = _supervised_text(row, thinking)

    marked = None
    if _marks_assistant_tokens(tokenizer):
        conversation: list[ChatMsg] = [*messages, {"role": "assistant", "content": answer}]
        marked = _mask_from_template(tokenizer, conversation, enable_thinking=enable_thinking)

    if marked is None:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if isinstance(prompt_ids, Mapping):
            prompt_ids = prompt_ids["input_ids"]
        prompt = [int(token) for token in prompt_ids]

        terminator = str(getattr(tokenizer, "eos_token", "") or "")
        if not terminator:
            raise CollationError(
                "the tokenizer declares no eos_token, so the answer cannot be terminated"
            )
        completion = _encode_plain(tokenizer, answer + terminator)
        if not completion:
            raise CollationError("the answer tokenised to nothing")
        ids = prompt + completion
        mask = [0] * len(prompt) + [1] * len(completion)
    else:
        ids, mask = marked

    if max_seq_len is not None and len(ids) > max_seq_len:
        raise SequenceTooLong(f"example renders to {len(ids)} tokens, over the {max_seq_len} limit")

    labels = [token if flag else IGNORE_INDEX for token, flag in zip(ids, mask, strict=True)]
    return Example(input_ids=ids, labels=labels)


__all__ = [
    "build_corrector_example",
    "corrector_answer",
    "corrector_messages",
]
