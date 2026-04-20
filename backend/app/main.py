from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Linhai Mahjong V3")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "linhai-majiang-v3"}

