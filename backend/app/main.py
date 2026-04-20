from __future__ import annotations

from fastapi import FastAPI

from .services.orchestrator import get_orchestrator

app = FastAPI(title="Linhai Mahjong V3")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "linhai-majiang-v3"}


@app.get("/debug/engine_status")
def engine_status() -> dict:
    """Expose which engine the orchestrator is currently routing through.

    Ops/clients can hit this to tell whether V3, V2, or the pure-Python heuristic
    fallback is live. If "active_engine" is "heuristic", the service is running
    in a degraded mode (most commonly because the C++ extension failed to load).
    """
    return get_orchestrator().engine_status()

