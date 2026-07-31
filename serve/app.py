"""FastAPI factory for the private sentence-grading shim."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lexi_research.format import BandConfig
from lexi_research.teacher.schemas import SenseRef

from .service import Backend, GradeUnavailable, grade


class GradeRequest(BaseModel):
    target: str
    definition: str
    pos: str
    text: str


def create_app(
    backend: Backend, config: BandConfig, *, adapter_revision: str = "unknown"
) -> FastAPI:
    app = FastAPI(title="lexi grader shim")

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> dict[str, object]:
        return {
            "ok": True,
            "adapter_revision": adapter_revision,
            "band_config_version": config.version,
        }

    @app.post("/v1/chat/completions")
    async def completion(request: GradeRequest) -> dict[str, object]:
        try:
            result = await grade(
                backend,
                request.target,
                SenseRef(definition=request.definition, pos=request.pos),
                request.text,
                config,
            )
        except GradeUnavailable as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "correction": result.correction,
            "meaning": result.meaning,
            "grammar": result.grammar,
            "naturalness": result.naturalness,
            "feedback": result.feedback,
            "band_config_version": result.band_config_version,
        }

    return app
