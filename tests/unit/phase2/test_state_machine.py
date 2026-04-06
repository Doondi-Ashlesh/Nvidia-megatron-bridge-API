"""Phase 2 unit tests — job state machine (job_service.py)."""

from __future__ import annotations

import importlib
import os
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest

from app.schemas.common import JobStatus, JobType
from app.services.job_service import InvalidStateTransitionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fresh_db(tmp_path: Path):
    """Helper: reload config + database with a fresh SQLite DB."""
    db_path = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["DATA_DIR"] = str(tmp_path)

    import app.config as cfg
    importlib.reload(cfg)
    import app.database as db_mod
    importlib.reload(db_mod)
    await db_mod.init_db()

    # job_service has no direct settings dependency — do NOT reload it
    # (reloading would create new class objects that break isinstance checks)
    import app.services.job_service as js

    return db_path, db_mod, js


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------


class TestCreateJob:
    async def test_creates_queued_job(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(
                db,
                job_type=JobType.PRETRAIN,
                payload={"dataset": "wiki"},
                num_gpus=2,
            )
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID4

    async def test_job_appears_in_db(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(
                db,
                job_type=JobType.LORA,
                payload={"model": "llama3"},
                num_gpus=1,
            )
            row = await js.get_job(db, job_id)
        assert row is not None
        assert row["status"] == JobStatus.QUEUED.value
        assert row["type"] == JobType.LORA.value
        assert row["payload"]["model"] == "llama3"
        assert row["num_gpus"] == 1


# ---------------------------------------------------------------------------
# get_job / list_jobs
# ---------------------------------------------------------------------------


class TestGetJob:
    async def test_returns_none_for_missing_job(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            row = await js.get_job(db, "00000000-0000-4000-8000-000000000000")
        assert row is None

    async def test_returns_job_dict(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.FINETUNE, payload={})
            row = await js.get_job(db, job_id)
        assert row["id"] == job_id


class TestListJobs:
    async def test_empty_list(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            items, total = await js.list_jobs(db)
        assert items == []
        assert total == 0

    async def test_filter_by_status(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            jid1 = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            jid2 = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            # Manually mark one as running
            await db.execute(
                "UPDATE jobs SET status='running' WHERE id=?", (jid2,)
            )
            await db.commit()
            items, total = await js.list_jobs(db, status=JobStatus.QUEUED)
        assert total == 1
        assert items[0]["id"] == jid1

    async def test_filter_by_type(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.create_job(db, job_type=JobType.LORA, payload={})
            items, total = await js.list_jobs(db, job_type=JobType.LORA)
        assert total == 1
        assert items[0]["type"] == JobType.LORA.value

    async def test_limit_and_offset(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            for _ in range(5):
                await js.create_job(db, job_type=JobType.FINETUNE, payload={})
            items, total = await js.list_jobs(db, limit=2, offset=1)
        assert len(items) == 2
        assert total == 5


# ---------------------------------------------------------------------------
# State transitions — happy paths
# ---------------------------------------------------------------------------


class TestTransitions:
    async def test_queued_to_running(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.transition_to_running(db, job_id, pid=12345, log_path="/logs/x.log")
            row = await js.get_job(db, job_id)
        assert row["status"] == JobStatus.RUNNING.value
        assert row["pid"] == 12345
        assert row["log_path"] == "/logs/x.log"
        assert row["started_at"] is not None

    async def test_running_to_completed(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.transition_to_running(db, job_id, pid=1, log_path="/logs/x.log")
            await js.transition_to_completed(db, job_id, progress={"step": 1000})
            row = await js.get_job(db, job_id)
        assert row["status"] == JobStatus.COMPLETED.value
        assert row["progress"]["step"] == 1000
        assert row["completed_at"] is not None

    async def test_running_to_failed(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.transition_to_running(db, job_id, pid=1, log_path="/logs/x.log")
            await js.transition_to_failed(db, job_id, error="OOM error")
            row = await js.get_job(db, job_id)
        assert row["status"] == JobStatus.FAILED.value
        assert row["error"] == "OOM error"

    async def test_running_to_cancelling_to_cancelled(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.transition_to_running(db, job_id, pid=99, log_path="/logs/x.log")
            pid = await js.transition_to_cancelling(db, job_id)
            assert pid == 99
            row = await js.get_job(db, job_id)
            assert row["status"] == JobStatus.CANCELLING.value

            await js.transition_to_cancelled(db, job_id)
            row = await js.get_job(db, job_id)
        assert row["status"] == JobStatus.CANCELLED.value

    async def test_queued_to_cancelled_directly(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.transition_to_cancelled(db, job_id)
            row = await js.get_job(db, job_id)
        assert row["status"] == JobStatus.CANCELLED.value

    async def test_completed_with_no_progress(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.transition_to_running(db, job_id, pid=1, log_path="/logs/x.log")
            await js.transition_to_completed(db, job_id)
            row = await js.get_job(db, job_id)
        assert row["status"] == JobStatus.COMPLETED.value

    async def test_update_progress(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.transition_to_running(db, job_id, pid=1, log_path="/logs/x.log")
            await js.update_progress(db, job_id, {"step": 50, "loss": 2.1})
            row = await js.get_job(db, job_id)
        assert row["progress"]["step"] == 50


# ---------------------------------------------------------------------------
# State transitions — invalid / rejected
# ---------------------------------------------------------------------------


class TestInvalidTransitions:
    async def test_completed_to_failed_raises(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.transition_to_running(db, job_id, pid=1, log_path="/logs/x.log")
            await js.transition_to_completed(db, job_id)
            with pytest.raises(InvalidStateTransitionError):
                await js.transition_to_failed(db, job_id, error="too late")

    async def test_cancelled_to_running_raises(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            job_id = await js.create_job(db, job_type=JobType.PRETRAIN, payload={})
            await js.transition_to_cancelled(db, job_id)
            with pytest.raises(InvalidStateTransitionError):
                await js.transition_to_running(db, job_id, pid=1, log_path="/x.log")

    async def test_transition_on_missing_job_raises(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            with pytest.raises(ValueError, match="not found"):
                await js.transition_to_running(
                    db, "00000000-0000-4000-8000-000000000000", pid=1, log_path="/x.log"
                )

    async def test_cancelling_missing_job_raises(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            with pytest.raises(ValueError, match="not found"):
                await js.transition_to_cancelling(db, "00000000-0000-4000-8000-000000000000")

    async def test_cancel_missing_job_raises(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            with pytest.raises(ValueError, match="not found"):
                await js.transition_to_cancelled(db, "00000000-0000-4000-8000-000000000000")

    async def test_failed_missing_job_raises(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            with pytest.raises(ValueError, match="not found"):
                await js.transition_to_failed(db, "00000000-0000-4000-8000-000000000000", error="x")

    async def test_completed_missing_job_raises(self, tmp_path):
        db_path, _, js = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            with pytest.raises(ValueError, match="not found"):
                await js.transition_to_completed(db, "00000000-0000-4000-8000-000000000000")
