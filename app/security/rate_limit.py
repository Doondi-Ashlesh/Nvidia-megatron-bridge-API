"""Simple in-memory sliding-window rate limiter middleware.

Limits write requests (POST, PUT, PATCH, DELETE) to ``settings.rate_limit_requests``
per IP per 60-second window. GET/HEAD/OPTIONS are never rate-limited.

No external dependency — uses a dict of deques in process memory.
Note: this resets on restart and is per-process only. For multi-process
deployments, move state to Redis or use a proper rate-limiting proxy.
"""

from __future__ import annotations

import collections
import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, requests_per_minute: int) -> None:
        super().__init__(app)
        self._limit = requests_per_minute
        # {ip: deque of timestamps}
        self._windows: dict[str, collections.deque[float]] = collections.defaultdict(
            collections.deque
        )

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.method not in _WRITE_METHODS:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._windows[ip]

        # Evict timestamps older than the window
        while window and window[0] < now - _WINDOW_SECONDS:
            window.popleft()

        if len(window) >= self._limit:
            logger.warning("Rate limit exceeded for IP %s", ip)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: max {self._limit} write "
                        f"requests per {_WINDOW_SECONDS}s"
                    )
                },
                headers={"Retry-After": str(_WINDOW_SECONDS)},
            )

        window.append(now)
        return await call_next(request)
