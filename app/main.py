"""FastAPI application factory and lifespan manager."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db
from app.security.token_filter import TokenRedactFilter


# ── Logging setup ─────────────────────────────────────────────────────────────
def _configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    # Install the token redaction filter on the root logger so ALL log records
    # (including those from uvicorn, FastAPI, and MegatronBridge) are scrubbed.
    root_logger = logging.getLogger()
    root_logger.addFilter(TokenRedactFilter())


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Application lifespan: startup → yield → shutdown."""
    _configure_logging()

    logger = logging.getLogger(__name__)
    logger.info("MegatronBridge API v%s starting up …", settings.version)

    # ── Startup ───────────────────────────────────────────────────────────────
    # Ensure runtime directories exist
    for directory in (
        settings.data_dir,
        settings.checkpoints_root,
        settings.logs_root,
        settings.hf_cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # Initialise database (creates tables, enables WAL, recovers stuck jobs)
    await init_db()

    # Start the job dispatcher background task
    import asyncio

    from app.worker.dispatcher import dispatcher_loop

    dispatcher_task = asyncio.create_task(dispatcher_loop())

    logger.info("Startup complete. Listening on %s:%s", settings.api_host, settings.api_port)
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    dispatcher_task.cancel()
    with suppress(asyncio.CancelledError):
        await dispatcher_task
    logger.info("MegatronBridge API shutting down.")


# ── Application factory ───────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Separated from module-level instantiation so tests can call this
    with custom environment overrides.
    """
    from app.api.router import router  # local import to allow env override in tests

    app = FastAPI(
        title="MegatronBridge API",
        description=(
            "Open-source REST API wrapper around Nvidia's MegatronBridge Python SDK. "
            "Exposes checkpoint conversion, distributed training, and PEFT fine-tuning "
            "over HTTP and WebSocket — making MegatronBridge accessible from any "
            "language, tool, or LLM agent."
        ),
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── Rate limiting ────────────────────────────────────────────────────────
    from app.security.rate_limit import RateLimitMiddleware  # noqa: PLC0415
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_requests,
    )

    # ── Auth ─────────────────────────────────────────────────────────────────
    # Middleware is evaluated last-added first, so add auth before CORS so that
    # unauthenticated requests are rejected before CORS headers are sent.
    if settings.api_key:
        from app.security.auth import ApiKeyMiddleware  # noqa: PLC0415
        app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)

    # ── CORS ──────────────────────────────────────────────────────────────────
    if settings.cors_origins_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Exception handlers ────────────────────────────────────────────────────
    from app.services.job_service import QueueFullError  # noqa: PLC0415

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(_req: Request, exc: QueueFullError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    return app


# Module-level app instance used by uvicorn
app = create_app()


def main() -> None:
    """Entry point for the ``megatronbridge-api`` CLI command."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
