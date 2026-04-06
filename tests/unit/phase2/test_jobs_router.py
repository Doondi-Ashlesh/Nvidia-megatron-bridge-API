"""Phase 2 unit tests — jobs REST router (/v1/jobs).

Uses the test_app / client fixtures from conftest.py (fresh DB, mocked megatron.bridge).
"""

from __future__ import annotations

import importlib
import json
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Helper: create a job directly in the DB
# ---------------------------------------------------------------------------


async def _seed_job(client: AsyncClient, job_type: str = "pretrain", num_gpus: int = 1) -> str:
    """Insert a job via the DB helper and return its job_id."""
    import aiosqlite
    from datetime import datetime, timezone

    import app.config as cfg

    db_url = cfg.settings.database_url
    prefix = "sqlite+aiosqlite:///"
    db_path = db_url[len(prefix):]

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO jobs (id, type, status, payload, num_gpus, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, job_type, "queued", json.dumps({}), num_gpus, now),
        )
        await db.commit()

    return job_id


# ---------------------------------------------------------------------------
# GET /v1/jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    async def test_empty_list(self, client: AsyncClient):
        resp = await client.get("/v1/jobs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["total"] == 0

    async def test_returns_created_jobs(self, client: AsyncClient):
        await _seed_job(client)
        await _seed_job(client)
        resp = await client.get("/v1/jobs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["items"]) == 2

    async def test_filter_by_status(self, client: AsyncClient):
        await _seed_job(client)
        resp = await client.get("/v1/jobs?status=queued")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1

    async def test_filter_by_invalid_status_returns_422(self, client: AsyncClient):
        resp = await client.get("/v1/jobs?status=bogus")
        assert resp.status_code == 422

    async def test_limit_and_offset(self, client: AsyncClient):
        for _ in range(5):
            await _seed_job(client)
        resp = await client.get("/v1/jobs?limit=2&offset=0")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 2


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id}
# ---------------------------------------------------------------------------


class TestGetJob:
    async def test_returns_job(self, client: AsyncClient):
        job_id = await _seed_job(client)
        resp = await client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == job_id
        assert body["status"] == "queued"

    async def test_not_found_returns_404(self, client: AsyncClient):
        resp = await client.get(f"/v1/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_non_uuid_returns_400(self, client: AsyncClient):
        resp = await client.get("/v1/jobs/not-a-uuid")
        assert resp.status_code == 400

    async def test_response_contains_required_fields(self, client: AsyncClient):
        job_id = await _seed_job(client)
        body = (await client.get(f"/v1/jobs/{job_id}")).json()
        for field in ("id", "type", "status", "created_at", "num_gpus"):
            assert field in body


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id}/logs
# ---------------------------------------------------------------------------


class TestGetJobLogs:
    async def test_queued_job_returns_empty_log(self, client: AsyncClient):
        job_id = await _seed_job(client)
        resp = await client.get(f"/v1/jobs/{job_id}/logs")
        assert resp.status_code == 200
        assert resp.text == ""

    async def test_non_uuid_returns_400(self, client: AsyncClient):
        resp = await client.get("/v1/jobs/bad-id/logs")
        assert resp.status_code == 400

    async def test_missing_job_returns_404(self, client: AsyncClient):
        resp = await client.get(f"/v1/jobs/{uuid.uuid4()}/logs")
        assert resp.status_code == 404

    async def test_returns_log_file_contents(self, client: AsyncClient, tmp_path):
        """When log_path is set and file exists, returns its content."""
        import aiosqlite
        import app.config as cfg

        job_id = await _seed_job(client)
        log_dir = cfg.settings.logs_root
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{job_id}.log"
        log_file.write_text("step 1 loss 2.5\n")

        db_url = cfg.settings.database_url
        db_path = db_url[len("sqlite+aiosqlite:///"):]
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE jobs SET log_path=?, status='running', pid=1 WHERE id=?",
                (str(log_file), job_id),
            )
            await db.commit()

        resp = await client.get(f"/v1/jobs/{job_id}/logs")
        assert resp.status_code == 200
        assert "step 1 loss 2.5" in resp.text


# ---------------------------------------------------------------------------
# DELETE /v1/jobs/{job_id}
# ---------------------------------------------------------------------------


class TestCancelJob:
    async def test_cancel_queued_job(self, client: AsyncClient):
        job_id = await _seed_job(client)
        resp = await client.delete(f"/v1/jobs/{job_id}")
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "cancelled"

    async def test_cancel_running_job_sends_sigterm(self, client: AsyncClient):
        import aiosqlite
        import app.config as cfg

        job_id = await _seed_job(client)
        db_path = cfg.settings.database_url[len("sqlite+aiosqlite:///"):]

        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE jobs SET status='running', pid=9999 WHERE id=?", (job_id,)
            )
            await db.commit()

        with patch("app.api.jobs.terminate_worker") as mock_term:
            resp = await client.delete(f"/v1/jobs/{job_id}")

        assert resp.status_code == 202
        assert resp.json()["status"] == "cancelling"
        mock_term.assert_called_once_with(9999)

    async def test_cancel_completed_job_returns_409(self, client: AsyncClient):
        import aiosqlite
        import app.config as cfg

        job_id = await _seed_job(client)
        db_path = cfg.settings.database_url[len("sqlite+aiosqlite:///"):]

        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE jobs SET status='completed' WHERE id=?", (job_id,)
            )
            await db.commit()

        resp = await client.delete(f"/v1/jobs/{job_id}")
        assert resp.status_code == 409

    async def test_cancel_failed_job_returns_409(self, client: AsyncClient):
        import aiosqlite
        import app.config as cfg

        job_id = await _seed_job(client)
        db_path = cfg.settings.database_url[len("sqlite+aiosqlite:///"):]

        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE jobs SET status='failed', error='oom' WHERE id=?", (job_id,)
            )
            await db.commit()

        resp = await client.delete(f"/v1/jobs/{job_id}")
        assert resp.status_code == 409

    async def test_cancel_missing_job_returns_404(self, client: AsyncClient):
        resp = await client.delete(f"/v1/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_cancel_non_uuid_returns_400(self, client: AsyncClient):
        resp = await client.delete("/v1/jobs/not-a-uuid")
        assert resp.status_code == 400

    async def test_cancel_running_job_without_pid(self, client: AsyncClient):
        """Running job with no PID stored — should still transition to cancelling."""
        import aiosqlite
        import app.config as cfg

        job_id = await _seed_job(client)
        db_path = cfg.settings.database_url[len("sqlite+aiosqlite:///"):]

        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE jobs SET status='running', pid=NULL WHERE id=?", (job_id,)
            )
            await db.commit()

        with patch("app.api.jobs.terminate_worker") as mock_term:
            resp = await client.delete(f"/v1/jobs/{job_id}")

        assert resp.status_code == 202
        mock_term.assert_not_called()
