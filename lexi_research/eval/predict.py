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
        attn_implementation=config.get_str("train.attn_implementation"),
        text_only=config.get_bool("train.text_only"),
        bnb_4bit_use_double_quant=config.get_bool("train.bnb_4bit_use_double_quant"),
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def _generation_inputs(encoded: Any, device: Any) -> tuple[Any, dict[str, Any]]:
    """Normalise tensor and BatchEncoding chat-template results for ``generate``."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise PredictionError("chat template result has no input_ids")
        inputs = {
            str(key): value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
            if value is not None
        }
        prompt = inputs["input_ids"]
    else:
        prompt = encoded.to(device) if hasattr(encoded, "to") else encoded
        inputs = {"input_ids": prompt}
    return prompt, inputs


def _left_pad(
    prompts: Sequence[Any],
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any]:
    """Stack prompts of differing length into one batch, padding on the left.

    Left rather than right: every sequence's final prompt token must sit at the
    same index, because that is the position ``generate`` continues from. A
    right-padded batch would ask the model to continue from padding for every
    row except the longest, and the shorter rows would decode from the wrong
    place — a silently wrong answer rather than a slow one.
    """
    import torch

    width = max(int(row.shape[-1]) for row in prompts)
    ids = []
    masks = []
    for row in prompts:
        flat = row.reshape(-1)
        padding = width - int(flat.shape[-1])
        pad = torch.full((padding,), pad_token_id, dtype=flat.dtype, device=flat.device)
        ids.append(torch.cat((pad, flat)))
        masks.append(
            torch.cat(
                (
                    torch.zeros(padding, dtype=torch.long, device=flat.device),
                    torch.ones(int(flat.shape[-1]), dtype=torch.long, device=flat.device),
                )
            )
        )
    return torch.stack(ids).to(device), torch.stack(masks).to(device)


def _decode_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Any],
    *,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate continuations for several prompts in one ``generate`` call.

    The whole batch decodes for as many steps as the slowest row needs, which is
    still far cheaper than paying a separate kernel launch and prefill per row.
    Measured on an L4 over 16 rows of the smoke fixture: 208 s one row at a time
    against 39 s at a batch of eight.

    Batching is not bit-identical to decoding one row at a time, and that is
    worth stating precisely rather than implying it is exact. Batch size changes
    the reduction order inside the bf16 matmuls, so a greedy argmax at a
    near-tie can land on a different token. Measured on the same L4 with greedy
    decoding and 128 new tokens: decoding a row alone twice agreed 8/8 times, so
    the model itself is deterministic; against a batch of eight, 5/8 rows were
    token-identical and the three that drifted did so from token 13 onward. All
    8 still parsed to identical answer JSON, which is what the report scores.

    The consequence to keep in mind: `raw` completions from a batched run may
    differ token-for-token from a serial one even at temperature 0. If an
    investigation needs byte-stable completions, set `eval.batch_size` to 1.
    """
    import torch

    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    input_ids, attention_mask = _left_pad(prompts, pad_token_id, model.device)

    eos_ids_set: set[int] = set()
    raw_eos = getattr(tokenizer, "eos_token_id", None)
    if raw_eos is not None:
        if isinstance(raw_eos, (list, tuple)):
            eos_ids_set.update(raw_eos)
        else:
            eos_ids_set.add(int(raw_eos))
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for sp in ("<|im_end|>", "<|endoftext|>"):
            sp_id = tokenizer.convert_tokens_to_ids(sp)
            if isinstance(sp_id, int) and sp_id != getattr(tokenizer, "unk_token_id", None):
                eos_ids_set.add(sp_id)

    eos_arg = list(eos_ids_set) if eos_ids_set else None

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            eos_token_id=eos_arg,
            pad_token_id=pad_token_id,
        )
    # Left padding makes the prompt width identical for every row, so the
    # completion starts at the same offset in all of them.
    width = int(input_ids.shape[-1])
    results = []
    for row in generated:
        gen_tensor = row[width:]
        if eos_ids_set:
            gen_list = gen_tensor.tolist()
            first_eos = None
            for idx, tok in enumerate(gen_list):
                if tok in eos_ids_set:
                    first_eos = idx
                    break
            if first_eos is not None:
                gen_tensor = gen_tensor[:first_eos]
        decoded = tokenizer.decode(gen_tensor, skip_special_tokens=True).strip()
        for leak_marker in ("\nuser", "\nassistant", "\nSentence:", "\n<|im_start|>"):
            if leak_marker in decoded:
                decoded = decoded.split(leak_marker)[0].strip()
        results.append(decoded)
    return results


def predict_rows(
    config: Config,
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    band_config: BandConfig,
    max_retries: int = 1,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """One prediction per row, retrying only what failed validation.

    The retry loop is part of what is measured, not a way to hide failures: the
    retry count lands in the report, because a student that needs two attempts is
    a different product than one that needs none.

    Rows are generated in batches of `batch_size` (defaulting to
    `eval.batch_size`) rather than one at a time. Only the rows that failed
    validation enter the next attempt, so a retry costs a batch of the failures
    rather than a re-run of everything.
    """
    max_new_tokens = config.get_int("eval.max_new_tokens")
    temperature = config.get_float("eval.temperature")
    # `forced-empty` still opens the template's reasoning path at inference;
    # only `off` asks it not to.
    enable_thinking = config.get_str("train.thinking") != "off"
    width = batch_size if batch_size is not None else config.get_int("eval.batch_size")
    if width < 1:
        raise PredictionError(f"eval.batch_size must be at least 1, got {width}")

    prompts: list[Any] = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            training_messages(row),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
        )
        prompt, _ = _generation_inputs(encoded, model.device)
        prompts.append(prompt)

    predictions: list[dict[str, Any] | None] = [None] * len(rows)
    completions: list[str] = [""] * len(rows)
    attempts: list[int] = [0] * len(rows)
    # Every row starts unresolved; each attempt re-queues only what is still
    # unresolved, which is what keeps a retry proportional to the failures.
    pending = list(range(len(rows)))

    for _ in range(max_retries + 1):
        if not pending:
            break
        still_pending: list[int] = []
        for start in range(0, len(pending), width):
            chunk = pending[start : start + width]
            decoded = _decode_batch(
                model,
                tokenizer,
                [prompts[index] for index in chunk],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            for index, completion in zip(chunk, decoded, strict=True):
                completions[index] = completion
                candidate = _extract_json(completion)
                if candidate is None:
                    # `retries` counts attempts that failed, so a row answered
                    # correctly first time reports zero.
                    attempts[index] += 1
                    still_pending.append(index)
                    continue
                predictions[index] = candidate
                checked = validate_output(candidate, str(rows[index]["text"]), band_config)
                if isinstance(checked, ValidationError):
                    attempts[index] += 1
                    still_pending.append(index)
        pending = still_pending

    return [
        {
            "req_uid": str(row.get("req_uid", index)),
            "text": str(row["text"]),
            "target": str(row["target"]),
            "definition": str(row["definition"]),
            "pos": str(row["pos"]),
            "gold": {key: row.get(key) for key in ("correction", "meaning", "feedback")},
            "prediction": predictions[index],
            "raw": completions[index],
            "retries": attempts[index],
        }
        for index, row in enumerate(rows)
    ]


__all__ = ["PredictionError", "load_for_inference", "predict_rows"]
