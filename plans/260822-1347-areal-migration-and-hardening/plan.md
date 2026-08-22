---
title: "Migrate SFT to AReaL-tau2-data, Fix Blocking Bugs, Harden Pipeline for Training"
status: planning
created: 2026-08-22
mode: tdd
blockedBy: []
blocks: []
---

# Plan: AReaL Migration + Bug Fixes + Training Readiness

## Executive Summary

Scout toàn repo phát hiện pipeline hiện tại là bộ khung TDD thuần túy: **chưa từng tồn tại code path huấn luyện thật** (không có Trainer instantiation, không có dataset loader, eval chạy trên MockEnv + DummyPolicy). Kèm 2 bug blocking về tokenizer/format. Plan này rebuild data pipeline sang `inclusionAI/AReaL-tau2-data` (Retail), sửa bug, thay mock bằng tích hợp thật với `AgentGymEnv`, và harden GRPO + eval để sẵn sàng chạy thí nghiệm trên Colab L4.

## Decisions Locked (từ phiên làm việc 2026-08-22)

| Quyết định | Giá trị |
|---|---|
| SFT dataset | `inclusionAI/AReaL-tau2-data` split `sft`, Retail only, filter `correct==1 & reward==1.0 & thinking!=""` (~11,395 trước filter, kỳ vọng ~7–8k sau filter) |
| RL dataset | Official τ²-bench Retail **train** split (74 tasks trong `split_tasks.json`), qua `AgentGymEnv` |
| Final eval | Official Retail **test** split (40 tasks), 4 trials/task |
| User simulator | External API (`gpt-4.1-mini` mặc định), freeze duy nhất 1 model cho cả RL rollout và mọi eval |
| Metric | Pass^1 primary; Pass^2/Pass^4 secondary; paired bootstrap 95% CI cho ΔSFT / ΔRL |
| fuvty | Giữ artifacts hiện tại làm ablation execution-grounded, bỏ khỏi main path |

## Phases Overview

| Phase | Title | Status | Dependencies |
|---|---|---|---|
| [Phase 01](phase-01-data-pipeline-and-tokenizer-fixes.md) | Data pipeline rebuild (AReaL converter) + Bug B fix + length profile | pending | [] |
| [Phase 02](phase-02-sft-trainer-entrypoint.md) | Real SFT training entrypoint (Bug A fix ở đây) | pending | [01] |
| [Phase 03](phase-03-real-env-integration.md) | AgentGymEnv thật: env factory, reward extraction, difficulty profiler | pending | [01] |
| [Phase 04](phase-04-eval-harness-hardening.md) | Eval harness: policy loaders, decoding plumbing, pass^k, per-checkpoint results | pending | [02, 03] |
| [Phase 05](phase-05-grpo-pipeline.md) | Real TRL rollout_func + memory plan + zero-variance wiring | pending | [03, 04] |
| [Phase 06](phase-06-scripts-cli-docs-polish.md) | Scripts/CLI/docs polish + commit baseline | pending | [01..05] |

## Global Acceptance Criteria

1. Converter xuất N examples; **100% examples render được bằng chat template Qwen3.5 thật** (hiện 0%).
2. **Round-trip tool-call giữ nguyên 100% arguments**: completion → `parse_model_output` → `to_env_action()` → parse ngược ra cùng tên hàm + args (hiện mất 2816/2816).
3. `uv run python -m tau_research.cli train-sft --config configs/sft.yaml --max-steps 5` giảm loss thật trên GPU/CPU.
4. Eval 1 task thật của official test split qua `AgentGymEnv` thật trả reward có `reward_info` parse được DB/COMMUNICATE.
5. Không còn artifact giả: difficulty profile regenerate từ rollout thật trên 74 train tasks; scripts không ghi đè split thật bằng ID bịa.
6. `make check && make test` pass; mọi test mới dùng dữ liệu format thật (AReaL sample), không chỉ mock tự chế.

## Key Risks

- **Thống kê**: test split chỉ 40 tasks → ΔRL (+2 đến +5pp kỳ vọng) khó đạt significance; bắt buộc paired delta + cân nhắc 8 trials cho final pair.
- **TRL API churn**: `rollout_func` là API experimental; pin đúng version sau smoke (deps đang khai báo `trl>=0.12` quá lỏng).
- **Contamination**: AReaL card không cam kết không trùng official test; Phase 01 chứa audit step.
- **flash-attn wheel** trong `[project.optional-dependencies].colab` trỏ tới release fork bên ngoài (`lesj0610`) — cần xác minh hoặc thay bằng wheel chính thức trước khi dùng trên Colab.
