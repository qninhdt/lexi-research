# Phase 01 — Data Pipeline Rebuild (AReaL Converter) + Tokenizer Fixes

**Status**: pending | **Depends on**: none | **Blocks**: Phase 02, 03, 04

## Requirements

- Module mới `src/tau_research/data/load_areal_sft.py` stream `tau2_sft_train.jsonl` (874MB, không load hết vào RAM):
  - Filter: `source_dialog_id` prefix `retail_` + `metadata.correct == 1` + `metadata.reward == 1.0` + `answer.thinking.strip() != ""`.
  - Normalize 3 dạng `answer.tool_calls` quan sát được: `{name, arguments: dict}`; Anthropic `{id, type, function: {name, arguments: JSON-string}}` (json.loads); `arguments: null` → bỏ call hoặc drop example (log count).
  - Build example: `prompt = messages` (history đã strip sẵn thinking — verify lại trên full data), `completion = <think>...</think> + action` render bằng `format_functional_tool_call` (canonical `name(k=v)`).
  - Ghi `artifacts/data/areal_sft_{train,val}.json` + split theo `source_dialog_id` (không phải turn), seed 42, 90/10.
  - Log stats: tổng rows, sau filter, số example bị drop vì args null, phân bố turn_index.
- Sửa `prepare_sft.py::format_assistant_message_content`: `json.loads` khi `arguments` là string (nguồn gốc Bug B).
- Sửa `action_parser.py`: `parse_tool_string` fallback json_repair khi part không chứa `=` nhưng parse được JSON (phòng thủ cho model output lệch format).
- Script `src/tau_research/data/profile_lengths.py` chạy trên data đã convert bằng tokenizer thật; quyết định `max_seq_length` (kỳ vọng 6144; hiện 4096 che được 81.7%).
- Decontamination audit: script so scenario text (n-gram 8-gram Jaccard) giữa AReaL retail scenarios và 40 official test tasks; lưu report vào `artifacts/evaluation/decontamination_report.json`.

## Files

- Create: `src/tau_research/data/load_areal_sft.py`
- Modify: `src/tau_research/data/prepare_sft.py`, `src/tau_research/tau/action_parser.py`
- Create: `src/tau_research/data/audit_decontamination.py`
- Modify: `configs/sft.yaml` (dataset.name → areal path, max_seq_length theo profile)
- Tests: `tests/test_areal_converter.py` (mới), `tests/test_action_parser.py` (thêm JSON-args case), `tests/test_no_test_leakage.py` (dùng ID thật)

## Tests

- Round-trip property test: với mỗi completion đã convert, `parse_model_output` → `to_env_action` → `parse_tool_string` phải khớp tool name + args gốc (assert trên toàn bộ dataset converted, sample ≥1000).
- Render test: 100% converted examples pass `_apply_chat_template` (sau khi Phase 02 fix render).
- Converter test dùng fixture = 5 rows AReaL thật (tải sẵn vào `tests/fixtures/areal_sample.jsonl`).

## Risks / Notes

- File 874MB — stream line-by-line; không dùng `datasets` lib (server HF không render được do schema cast lỗi).
- Nếu sau filter số sample < 5k → cân nhắc giữ thêm domain Airline làm augmentation (quyết định lại lúc đó, báo user).
