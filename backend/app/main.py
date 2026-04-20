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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import ai as ai_router
from .routers import health as health_router
from .services.orchestrator import get_orchestrator

app = FastAPI(title="Linhai Mahjong V3", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
