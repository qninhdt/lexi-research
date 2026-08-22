# Phase 02 — Real SFT Training Entrypoint (Fix Bug A)

**Status**: pending | **Depends on**: Phase 01 | **Blocks**: Phase 04, 05

## Requirements

- Sửa `train_sft.py::prepare_sft_dataset_for_trainer` (Bug A): render **một lần** `(prompt + completion)` chung một message list, rồi cắt completion text tại biên (tokenize prompt riêng để lấy prefix length). Xử lý prompt chưa có user message: **skip + log** (case "assistant chào trước" tồn tại trong data thật).
- Thêm `train_sft.py::run_sft_training(config) -> None`: instantiate TRL `SFTTrainer` thật:
  - Dataset dạng text prompt/completion, `completion_only_loss=True`, `packing=False`, `max_length=max_seq_length`, drop-truncate policy (log % bị truncate).
  - LoRA từ yaml (r=16, alpha=32, dropout 0.05, all-linear), bf16, gradient checkpointing, seed 42.
  - `TauWandbCallback` + `report_to: wandb`, eval trên split val, save best/final.
  - Trả về adapter path; gọi `merge_lora_adapter` ở cuối (đã có sẵn).
- CLI: `tau-research train-sft --config configs/sft.yaml [--max-steps N] [--dry-run]` — hiện `cli.py` chỉ là stub in "passed" mà không làm gì.
- Pin versions sau smoke: `trl`, `transformers`, `vllm` (hiện `trl>=0.12`, `transformers>=4.48` quá lỏng so với spec idea.md yêu cầu transformers>=5.2 cho environment API).
- Smoke gate trước full run: 200 examples, max_steps=5, loss phải giảm.

## Files

- Modify: `src/tau_research/training/train_sft.py`, `src/tau_research/cli.py`
- Modify: `configs/sft.yaml` (thêm eval_dataset path, save best)
- Tests: mở rộng `tests/test_sft_train_step.py`: render pass 100% fixtures; 1 test integration `@pytest.mark.slow` chạy 3 steps trên CPU với model 0.5B (Qwen2.5-0.5B đã có sẵn trong HF cache).

## Tests

- `test_prepare_sft_dataset_for_trainer` mới: 7026 fuvty examples cũ + AReaL converted fixtures render pass 100%.
- Loss-decrease smoke (slow, GPU-only, skip CI).

## Risks / Notes

- Qwen3.5-2B BF16 + LoRA + grad ckpt + batch 1 + 6k seq trên L4 24GB: hợp lý nhưng cần smoke đo VRAM trước khi full.
- `enable_thinking=True` chỉ áp cho prompt render; completion render sau fix sẽ giữ nguyên khối think trong text.
- Template Qwen3.5 tự strip think của assistant turn TRƯỚC user query cuối; `sanitize_history_for_turn` strip tất cả — chọn 1 policy duy nhất (khuyến nghị: để template tự xử, bỏ strip thủ công khỏi prompt path để tránh lệch train/inference).
