"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from app.config import Settings, get_settings
from app.db.connection import check_connection, close_db, init_db
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

# Step 2 routes will be mounted under /api/v1/


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down shared resources."""
    settings: Settings = app.state.settings
    logging.basicConfig(level=settings.log_level.upper())
    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded; database_url=%s", settings.database_url_redacted)

    init_db(settings.database_url)
    yield
    await close_db()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    resolved_settings = settings or get_settings()

    app = FastAPI(title="Aion Memory Service", lifespan=lifespan)
    app.state.settings = resolved_settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe — no database dependency."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(response: Response) -> dict[str, Any]:
        """Readiness probe — verifies database connectivity."""
        if await check_connection():
            return {"status": "ready"}
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "detail": "database unavailable"}

    return app


app = create_app()
