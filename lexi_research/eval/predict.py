"""Generate predictions from a checkpoint. The only part of eval that needs a GPU.

Split from scoring so a metric fix never costs a re-run of the model. What lands
on disk is a JSONL the scorer reads; the two meet at that schema rather than in a
shared process.

Prompts come from `render_grader_prompt` through the tokenizer's chat template —
the same path training used and the same path serving uses. Evaluating with a
different prompt than the student was trained on measures the wrong function, and
would do so invisibly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lexi_research.cli.config import Config
from lexi_research.format import BandConfig, ValidationError, validate_output
from lexi_research.train.collate import training_messages


class PredictionError(RuntimeError):
    """The model could not be loaded, or produced nothing usable."""


def _extract_json(text: str) -> dict[str, Any] | None:
    """The first balanced JSON object in a completion.

    A model that opens a reasoning block and then answers puts prose before the
    object, and a raw `json.loads` on the whole completion would reject a
    perfectly good answer.
    """
    depth = 0
    start = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def load_for_inference(config: Config, adapter: str | Path | None) -> tuple[Any, Any]:
    """The base model named in config, with an adapter applied when given."""
    from lexi_research.train.trainer import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        config.get_str("train.base_model"),
        load_in_4bit=config.get_bool("train.load_in_4bit"),
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def predict_rows(
    config: Config,
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    band_config: BandConfig,
    max_retries: int = 1,
) -> list[dict[str, Any]]:
    """One prediction per row, retrying only what failed validation.

    The retry loop is part of what is measured, not a way to hide failures: the
    retry count lands in the report, because a student that needs two attempts is
    a different product than one that needs none.
    """
    import torch

    max_new_tokens = config.get_int("eval.max_new_tokens")
    temperature = config.get_float("eval.temperature")
    # `forced-empty` still opens the template's reasoning path at inference;
    # only `off` asks it not to.
    enable_thinking = config.get_str("train.thinking") != "off"

    out: list[dict[str, Any]] = []
    for row in rows:
        messages = training_messages(row)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
        )
        prompt = prompt.to(model.device)

        prediction: dict[str, Any] | None = None
        completion = ""
        attempts = 0
        while attempts <= max_retries:
            with torch.no_grad():
                generated = model.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(
                generated[0][prompt.shape[-1] :], skip_special_tokens=True
            )
            candidate = _extract_json(completion)
            if candidate is not None:
                checked = validate_output(candidate, str(row["text"]), band_config)
                if not isinstance(checked, ValidationError):
                    prediction = candidate
                    break
                prediction = candidate
            attempts += 1

        out.append(
            {
                "req_uid": str(row.get("req_uid", len(out))),
                "text": str(row["text"]),
                "target": str(row["target"]),
                "definition": str(row["definition"]),
                "pos": str(row["pos"]),
                "gold": {key: row.get(key) for key in ("correction", "meaning", "feedback")},
                "prediction": prediction,
                "raw": completion,
                "retries": attempts,
            }
        )
    return out


__all__ = ["PredictionError", "load_for_inference", "predict_rows"]
