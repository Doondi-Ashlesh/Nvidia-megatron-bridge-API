"""Tests for API key auth middleware and rate limiter."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    async def test_no_auth_configured_allows_all(self, client: AsyncClient):
        """When API_KEY is empty (default), all requests pass through."""
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_exempt_paths_always_allowed(self, client: AsyncClient):
        """Health endpoints are always reachable even if auth were configured."""
        for path in ["/health", "/health/ready"]:
            resp = await client.get(path)
            assert resp.status_code in (200, 503)  # 503 if DB not ready

    async def test_auth_middleware_rejects_missing_key(self, test_app):
        """When API_KEY is set, requests without a key get 401."""
        import app.config as cfg

        original = cfg.settings.api_key
        cfg.settings.api_key = "supersecret"
        try:
            from app.security.auth import ApiKeyMiddleware, _EXEMPT_PATHS

            # Manually verify that our middleware would reject a missing token
            # (middleware unit test — no need to rebuild the full app)
            from unittest.mock import AsyncMock, MagicMock
            from fastapi.responses import JSONResponse

            middleware = ApiKeyMiddleware(app=AsyncMock(), api_key="supersecret")

            mock_request = MagicMock()
            mock_request.url.path = "/v1/jobs"
            mock_request.headers.get.return_value = ""
            mock_request.query_params.get.return_value = None
            mock_request.client.host = "127.0.0.1"

            call_next = AsyncMock()
            response = await middleware.dispatch(mock_request, call_next)

            assert isinstance(response, JSONResponse)
            assert response.status_code == 401
        finally:
            cfg.settings.api_key = original

    async def test_auth_middleware_accepts_valid_key(self, test_app):
        """Valid Bearer token passes through to the next handler."""
        from unittest.mock import AsyncMock, MagicMock
        from fastapi import Request
        from app.security.auth import ApiKeyMiddleware

        mock_response = MagicMock()
        call_next = AsyncMock(return_value=mock_response)

        middleware = ApiKeyMiddleware(app=AsyncMock(), api_key="mykey")

        mock_request = MagicMock()
        mock_request.url.path = "/v1/jobs"
        mock_request.headers.get.return_value = "Bearer mykey"
        mock_request.query_params.get.return_value = None

        response = await middleware.dispatch(mock_request, call_next)
        call_next.assert_called_once()
        assert response is mock_response

    async def test_auth_middleware_accepts_query_param_key(self, test_app):
        """api_key query param is accepted (needed for WebSocket clients)."""
        from unittest.mock import AsyncMock, MagicMock
        from app.security.auth import ApiKeyMiddleware

        mock_response = MagicMock()
        call_next = AsyncMock(return_value=mock_response)

        middleware = ApiKeyMiddleware(app=AsyncMock(), api_key="mykey")

        mock_request = MagicMock()
        mock_request.url.path = "/v1/ws/jobs/abc/logs"
        mock_request.headers.get.return_value = ""
        mock_request.query_params.get.return_value = "mykey"

        response = await middleware.dispatch(mock_request, call_next)
        call_next.assert_called_once()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    async def test_get_requests_not_rate_limited(self, client: AsyncClient):
        """GET requests are never counted against the rate limit."""
        for _ in range(10):
            resp = await client.get("/health")
            assert resp.status_code == 200

    async def test_rate_limit_middleware_unit(self):
        """Verify the sliding-window logic rejects after limit is reached."""
        from unittest.mock import AsyncMock, MagicMock
        from app.security.rate_limit import RateLimitMiddleware

        mock_response = MagicMock()
        call_next = AsyncMock(return_value=mock_response)

        middleware = RateLimitMiddleware(app=AsyncMock(), requests_per_minute=3)

        def _make_post_request():
            req = MagicMock()
            req.method = "POST"
            req.client.host = "10.0.0.1"
            req.url.path = "/v1/jobs"
            return req

        # First 3 requests should pass
        for _ in range(3):
            resp = await middleware.dispatch(_make_post_request(), call_next)
            assert resp is mock_response

        # 4th request should be rate-limited
        from fastapi.responses import JSONResponse
        resp = await middleware.dispatch(_make_post_request(), call_next)
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Queue depth limit
# ---------------------------------------------------------------------------


class TestQueueDepth:
    async def test_queue_full_error_returns_429(self, client: AsyncClient, test_app):
        """When the queue is full, job creation returns 429."""
        import app.config as cfg
        from unittest.mock import patch, AsyncMock
        from app.services.job_service import QueueFullError

        # Simulate a full queue by patching create_job to raise QueueFullError
        with patch(
            "app.services.job_service.create_job",
            new_callable=AsyncMock,
            side_effect=QueueFullError("Queue is full (100/100 queued)"),
        ):
            resp = await client.post(
                "/v1/checkpoints/import",
                json={
                    "source_path": "meta-llama/Llama-3-8B",
                    "target_name": "llama3-megatron",
                },
            )
        assert resp.status_code == 429
        assert "queue" in resp.json()["detail"].lower()
