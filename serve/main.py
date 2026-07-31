"""Environment-driven ASGI application entrypoint for private deployments."""

from __future__ import annotations

import os

from fastapi import FastAPI

from lexi_research.format import BandConfig, default_config_path

from .app import create_app
from .backend import OpenAIBackend


def configured_app() -> FastAPI:
    base_url = os.environ.get("LEXI_BACKEND_URL")
    model = os.environ.get("LEXI_BACKEND_MODEL")
    if not base_url or not model:
        raise RuntimeError("LEXI_BACKEND_URL and LEXI_BACKEND_MODEL are required")
    config_path = os.environ.get("LEXI_BAND_CONFIG", str(default_config_path()))
    return create_app(
        OpenAIBackend(base_url, model, os.environ.get("LEXI_BACKEND_API_KEY", "")),
        BandConfig.from_json(config_path),
        adapter_revision=os.environ.get("LEXI_ADAPTER_REVISION", "unknown"),
    )


app = configured_app()
