"""HTTP routers for the V3 backend.

Kept explicit so imports `from .routers import ai, health` work and
`app.include_router(...)` doesn't accidentally double-register.
"""

from . import ai, health

__all__ = ["ai", "health"]
