"""LoRA adapter merge utility for creating standalone starting checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def merge_lora_adapter(
    peft_model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    seed: int | None = None,
) -> Any:
    """Merges LoRA adapter weights into base model and saves standalone model artifact."""
    if seed is not None:
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:
            pass

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    return merged_model
