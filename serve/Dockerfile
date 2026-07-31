FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY lexi_research ./lexi_research
COPY serve ./serve
COPY band_config.json ./band_config.json

RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

EXPOSE 8000
CMD [".venv/bin/uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
