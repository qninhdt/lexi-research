"""W&B logging callbacks and GPU memory tracking for tau-research."""

import time
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class TauWandbCallback(TrainerCallback):
    """Custom W&B Callback logging step times, token throughput, and GPU VRAM usage."""

    def __init__(self) -> None:
        super().__init__()
        self.step_start_time = time.time()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.step_start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step_duration = time.time() - self.step_start_time

        metrics: dict[str, Any] = {
            "system/step_time": step_duration,
        }

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)

            metrics.update(
                {
                    "gpu/memory_allocated_gb": allocated_gb,
                    "gpu/memory_reserved_gb": reserved_gb,
                    "gpu/max_memory_allocated_gb": max_allocated_gb,
                }
            )

        if state.is_world_process_zero and "wandb" in args.report_to:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
            except Exception:
                pass
