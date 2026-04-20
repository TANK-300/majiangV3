"""Legacy-compatible health router.

The uni-app frontend hits `GET /health/ping` (see
`majiang/frontend/utils/api.js`). We expose both /health/ping and the
existing top-level /health so that every existing client keeps working.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/ping")
def ping() -> dict:
    return {"status": "ok"}
