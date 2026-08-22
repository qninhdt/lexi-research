# Phase 06 — Scripts / CLI / Docs Polish + Baseline Commit

**Status**: pending | **Depends on**: Phase 01..05

## Requirements

- **Commit baseline NGAY đầu phase này** (repo đang có 243 thay đổi chưa commit — rủi ro mất code): tách commits: (1) legacy lexi deletion, (2) tau-research scaffolding hiện trạng, (3) plan files. Trước commit: quét secrets (.env không commit).
- Rewrite scripts gọi CLI thật:
  - `run_sft.sh` — xóa bước fake split IDs (`syn_retail_0000..0279` ghi đè artifacts thật); gọi `train-sft` thật.
  - `run_grpo.sh` — xóa difficulty profile bịa (`retail_easy_0...`); gọi profiler thật + `train-grpo` CLI.
  - `run_final_eval.sh` — bỏ MockTauGymEnv/DummyPolicy khỏi main path (chỉ còn trong smoke), gọi eval thật 3 checkpoints.
  - `smoke_test.sh` — CPU smoke thật (converter + render + parser round-trip), không in "passed" ảo.
- CLI đầy đủ: `train-sft`, `train-grpo`, `profile-difficulty`, `evaluate`, `convert-areal`, `audit-decontamination`.
- Docs: README (AReaL provenance, protocol eval, kết quả), `docs/tau-research-architecture-and-migration.md` update, model card template cho HF upload.
- CI: giữ ruff/mypy/pytest; cân nhắc tách extras nặng (vllm wheel fork `lesj0610` — xác minh checksum hoặc thay wheel chính thức).

## Files

- Modify: `scripts/*.sh`, `src/tau_research/cli.py`, `README.md`, `docs/`
- No new src module.

## Tests

- `bash scripts/smoke_test.sh` pass end-to-end trên máy local (CPU).
- CI green sau khi tách extras.

## Risks / Notes

- 243 thay đổi gồm cả deletion legacy `lexi_research/` — commit tách bạch để git history sạch cho CV.
- `.env` phải nằm trong .gitignore (đã có); không commit key.
