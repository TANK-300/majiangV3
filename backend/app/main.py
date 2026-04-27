"""FastAPI entry point.

Routes exposed:
  * GET  /health                  -- simple liveness check (V3-native path)
  * GET  /health/ping             -- legacy path the uni-app client calls
  * GET  /debug/engine_status     -- tells ops which engine actually answers
  * POST /ai/recommend            -- discard recommendation (app-compatible)
  * POST /ai/recommend-response   -- chi/peng/gang/hu/pass (NEW)
  * POST /ai/recommend-all        -- score every candidate discard
  * POST /ai/reset                -- engine reset (legacy no-op)
  * GET  /ai/health               -- AI self-check

CORS is permissive ("*") to match the legacy backend's config -- the
uni-app client runs on a separate origin / port and needs cross-origin.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import ai as ai_router
from .routers import health as health_router
from .services.orchestrator import get_orchestrator

logger = logging.getLogger("linhai.v3")

app = FastAPI(title="Linhai Mahjong V3", version="3.0.0")


@app.exception_handler(RequestValidationError)
async def _log_validation_error(request: Request, exc: RequestValidationError):
    """When the uni-app client posts a payload that doesn't match our
    schema, log the full payload + the specific errors so we can diagnose
    the shape mismatch from the server log instead of guessing. Returns
    the same 422 body FastAPI would have returned."""
    try:
        body = await request.body()
        body_text = body.decode("utf-8", errors="replace")[:4000]
    except Exception:  # noqa: BLE001
        body_text = "<unreadable>"
    logger.warning(
        "422 on %s %s -- errors=%s body=%s",
        request.method,
        request.url.path,
        exc.errors(),
        body_text,
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body_echo": body_text[:1000]},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _log_ai_recommend(request: Request, call_next):
    """Debug-only: echo request body + response body for /ai/recommend*
    so we can compare what the app actually sent vs. what the server
    returned. Logs truncated to avoid blowing up backend.log."""
    if request.url.path.startswith("/ai/recommend"):
        try:
            body = await request.body()
            body_text = body.decode("utf-8", errors="replace")[:1500]
        except Exception:  # noqa: BLE001
            body_text = "<unreadable>"
        logger.warning("AI_REQ %s from=%s body=%s", request.url.path,
                       request.client.host if request.client else "?",
                       body_text)
        response = await call_next(request)
        try:
            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk
            logger.warning("AI_RES %s body=%s", request.url.path,
                           resp_body.decode("utf-8", errors="replace")[:1500])
            return JSONResponse(
                content=__import__("json").loads(resp_body.decode("utf-8")),
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("AI_RES decode_fail err=%s", e)
            return response
    return await call_next(request)

app.include_router(health_router.router)
app.include_router(ai_router.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "linhai-majiang-v3"}


@app.get("/debug/engine_status")
def engine_status() -> dict:
    """Tells you which engine (`v3` / `v2` / `heuristic`) is actually
    answering AI requests. If this returns `active_engine: "heuristic"`,
    the C++ extension failed to load -- fix that before shipping.
    """
    return get_orchestrator().engine_status()
