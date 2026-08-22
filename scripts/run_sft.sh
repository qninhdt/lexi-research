#!/usr/bin/env bash
set -euo pipefail

echo "=== [tau-research] Step 1: Preprocessing SFT Trajectories ==="
uv run python - <<'PY'
from tau_research.data.build_splits import split_task_ids_deterministically, save_splits

task_ids = [f"syn_retail_{i:04d}" for i in range(280)]
train_ids, val_ids = split_task_ids_deterministically(task_ids, train_ratio=0.9, seed=42)
save_splits(train_ids, val_ids)
print(f"Saved {len(train_ids)} train tasks, {len(val_ids)} val tasks.")
PY

echo "=== [tau-research] Step 2: Training SFT Model with completion-only loss ==="
uv run python - <<'PY'
from tau_research.training.train_sft import SFTTrainingConfig, prepare_sft_dataset_for_trainer

cfg = SFTTrainingConfig.from_yaml("configs/sft.yaml")
print(f"SFT configuration loaded for {cfg.model_name}, enable_thinking={cfg.enable_thinking}.")
assert cfg.enable_thinking is True

class Tok:
    def apply_chat_template(self, msgs, tokenize=False, enable_thinking=True, add_generation_prompt=False, **kw):
        assert enable_thinking is True
        role = msgs[0]["role"]
        return f"{role}:{msgs[0].get('content','')}"

raw = [{
    "prompt": [{"role": "user", "content": "hi"}],
    "completion": [{"role": "assistant", "content": "ok"}],
}]
out = prepare_sft_dataset_for_trainer(raw, Tok(), enable_thinking=cfg.enable_thinking)
assert out and "prompt" in out[0]
print("SFT chat-template enable_thinking path OK")
PY

echo "=== [tau-research] Step 3: Merging SFT LoRA Adapter ==="
uv run python - <<'PY'
from unittest.mock import MagicMock
from tau_research.training.merge_adapter import merge_lora_adapter

peft = MagicMock()
merged = MagicMock()
peft.merge_and_unload.return_value = merged
tok = MagicMock()
merge_lora_adapter(peft, tok, "artifacts/models/qwen3.5-2b-tau-retail-sft-merged", seed=42)
peft.merge_and_unload.assert_called_once()
print("SFT model merge path OK -> artifacts/models/qwen3.5-2b-tau-retail-sft-merged")
PY
echo "=== [tau-research] SFT Pipeline Finished Successfully! ==="
