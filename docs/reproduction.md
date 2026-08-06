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

The sense pool is assembled and modified from several public datasets, on a schema
modelled after the Cambridge dictionary site; "Cambridge" in this repo names that
schema, not a redistributed Cambridge product. Reproduction still requires local
access to the source database, because the pool itself is not shipped here.

Two artifacts have different distribution rules, and the difference is a licence
rather than a preference:

- **Stage B** (`data/raw/`, teacher-generated) is publishable. `lexi data publish`
  pushes it to the Hugging Face Hub with a card generated from the run reports.
- **Stage A** (`data/gec/`, converted from W&I+LOCNESS) is **not**. The LOCNESS
  licence forbids distributing any part of the corpus to a third party, so it
  stays on the machine that built it and is absent from the upload allowlist.
