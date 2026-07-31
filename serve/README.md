# Serving shim

The shim exposes `POST /v1/chat/completions`, plus `/healthz` and `/readyz`, on a
private network. It forwards the shared grader prompt to any OpenAI-compatible
backend, validates the three model fields, and computes `grammar` and
`naturalness` from the bundled `band_config.json`.

```bash
export LEXI_BACKEND_URL=http://vllm:8000/v1
export LEXI_BACKEND_MODEL=lexi-grader
docker compose up --build
```

It intentionally has **no authentication**. The Compose mapping binds only to
localhost; add authentication before exposing it beyond a trusted network.
