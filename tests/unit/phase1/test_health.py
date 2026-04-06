"""Phase 1 unit tests — /health and /health/ready endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    async def test_returns_200_when_db_healthy(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=True):
            response = await client.get("/health")
        assert response.status_code == 200

    async def test_response_schema(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=True):
            response = await client.get("/health")
        body = response.json()
        assert "status" in body
        assert "version" in body
        assert "db_status" in body

    async def test_db_status_connected_when_healthy(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=True):
            response = await client.get("/health")
        assert response.json()["db_status"] == "connected"

    async def test_returns_503_when_db_unreachable(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=False):
            response = await client.get("/health")
        assert response.status_code == 503

    async def test_db_status_error_when_unhealthy(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=False):
            response = await client.get("/health")
        assert response.json()["db_status"] == "error"

    async def test_status_degraded_when_db_unreachable(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=False):
            response = await client.get("/health")
        assert response.json()["status"] == "degraded"

    async def test_version_is_string(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=True):
            response = await client.get("/health")
        assert isinstance(response.json()["version"], str)


class TestReadyEndpoint:
    async def test_returns_200_when_ready(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=True):
            response = await client.get("/health/ready")
        assert response.status_code == 200

    async def test_returns_503_when_not_ready(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=False):
            response = await client.get("/health/ready")
        assert response.status_code == 503

    async def test_ready_response_schema(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=True):
            response = await client.get("/health/ready")
        assert response.json()["status"] == "ready"

    async def test_not_ready_includes_reason(self, client: AsyncClient):
        with patch("app.api.health.check_db_health", new_callable=AsyncMock, return_value=False):
            response = await client.get("/health/ready")
        body = response.json()
        assert "reason" in body
