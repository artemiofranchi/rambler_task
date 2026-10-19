"""REST API сервиса модерации (FastAPI)."""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, Field

from moderation import __version__
from moderation.batcher import MicroBatcher
from moderation.config import load_config
from moderation.moderator import ModerationResult, Moderator


class ModerateRequest(BaseModel):
    text: str = Field(..., description="Текст сообщения", examples=["Отличная статья, спасибо!"])


class ModerateBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)


class ModerateResponse(BaseModel):
    decision: Literal["ALLOW", "REVIEW", "BLOCK"]
    categories: list[str]
    confidence: float = Field(..., ge=0, le=1)
    evidence: list[str] | None = Field(None, description="Фрагменты, на которых сработали правила (explain=true)")
    scores: dict[str, float] | None = Field(None, description="Оценки компонентов ансамбля (explain=true)")


class ModerateBatchResponse(BaseModel):
    results: list[ModerateResponse]


def _to_response(r: ModerationResult, explain: bool) -> ModerateResponse:
    return ModerateResponse(decision=r.decision, categories=r.categories, confidence=r.confidence,
                            evidence=r.evidence if explain else None, scores=r.scores if explain else None)


def create_app(moderator: Moderator | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mod = moderator or Moderator(load_config())
        svc = mod.cfg["service"]
        batcher = MicroBatcher(mod.moderate_batch, svc["batch_max_size"], svc["batch_max_wait_ms"])
        await batcher.start()
        mod.moderate_batch(["прогрев"])  # первый вызов модели медленный — делаем его до приёма трафика
        app.state.moderator, app.state.batcher = mod, batcher
        yield
        await batcher.stop()

    app = FastAPI(title="UGC Moderation", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__, "components": app.state.moderator.components}

    @app.post("/moderate", response_model=ModerateResponse, response_model_exclude_none=True)
    async def moderate(req: ModerateRequest, explain: bool = Query(False)):
        return _to_response(await app.state.batcher.submit(req.text), explain)

    @app.post("/moderate/batch", response_model=ModerateBatchResponse, response_model_exclude_none=True)
    async def moderate_batch(req: ModerateBatchRequest, explain: bool = Query(False)):
        results = await asyncio.to_thread(app.state.moderator.moderate_batch, req.texts)
        return ModerateBatchResponse(results=[_to_response(r, explain) for r in results])

    return app


app = create_app()
