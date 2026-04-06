"""API key authentication middleware.

Activated when ``API_KEY`` is set in the environment / .env file.
When disabled (API_KEY is empty), all requests pass through unchanged —
preserving backward-compatibility for local/VPN-only deployments.

Protocol: clients must include ``Authorization: Bearer <key>`` on every
request. WebSocket clients must pass it as the ``api_key`` query parameter
because the browser WebSocket API does not support custom headers:
  ws://host/v1/ws/jobs/{id}/logs?api_key=<key>

Exempt paths (always allowed regardless of API_KEY setting):
  GET /health
  GET /health/ready
  GET /docs
  GET /redoc
  GET /openapi.json
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that must always be reachable (health checks, docs)
_EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/health/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
})


def _constant_time_equal(a: str, b: str) -> bool:
    """Timing-safe string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces Bearer token auth on all non-exempt paths.

    Instantiated in ``app.main.create_app()`` only when ``settings.api_key``
    is non-empty, so zero overhead when auth is disabled.
    """

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        # Always allow exempt paths
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # Extract token: prefer Authorization header, fall back to query param
        # (query param needed for WebSocket clients)
        token: str | None = None

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[len("Bearer "):]

        if token is None:
            token = request.query_params.get("api_key")

        if token is None or not _constant_time_equal(token, self._api_key):
            logger.warning(
                "Unauthorized request: %s %s from %s",
                request.method,
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
