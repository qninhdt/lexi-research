# Reproduction

1. Install dependencies with `uv sync`.
2. Set `LEXI_SOURCE_DB` to the local, read-only Cambridge SQLite file.
3. Run `uv run dvc repro export`; this creates private artifacts under `data/pool/`.
4. Configure a teacher endpoint through `LEXI_TEACHER_BASE_URL`,
   `LEXI_TEACHER_API_KEY`, and `LEXI_TEACHER_MODEL`; run the probe before any bulk
   spend.
5. Run the pilot, inspect and record all Phase 4 gates, calibrate bands, then freeze
   the processed dataset before training.

Every result must record: git SHA, source DB SHA-256, DVC hash, split version,
params hash, seed, teacher model, prompt hash, and base-model revision.

Cambridge-derived parquet data is private. Reproduction requires legitimate local
access to the source database and configured private DVC/R2 storage.
