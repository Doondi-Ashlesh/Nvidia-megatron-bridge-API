"""Phase 1 unit tests — database.py."""

from __future__ import annotations

import importlib
import os
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest


async def _fresh_db(tmp_path: Path):
    """Helper: reload config with a fresh DB path and call init_db."""
    db_path = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["DATA_DIR"] = str(tmp_path)

    import app.config as cfg
    importlib.reload(cfg)

    # Also reload database so _settings() picks up fresh config
    import app.database as db_mod
    importlib.reload(db_mod)

    await db_mod.init_db()
    return db_path, db_mod


class TestInitDb:
    async def test_creates_jobs_table(self, tmp_path: Path):
        db_path, _ = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
            )
            row = await cursor.fetchone()
        assert row is not None

    async def test_creates_checkpoints_table(self, tmp_path: Path):
        db_path, _ = await _fresh_db(tmp_path)
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            )
            row = await cursor.fetchone()
        assert row is not None

    async def test_idempotent_multiple_calls(self, tmp_path: Path):
        _, db_mod = await _fresh_db(tmp_path)
        # Second call must not raise
        await db_mod.init_db()

    async def test_recovers_stuck_running_jobs(self, tmp_path: Path):
        db_path, db_mod = await _fresh_db(tmp_path)

        # Insert a stuck 'running' job directly into the DB
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO jobs (id, type, status, payload, num_gpus, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "test-id-001", "pretrain", "running", "{}", 1,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()

        # Re-run init_db — should recover the stuck job
        await db_mod.init_db()

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT status, error FROM jobs WHERE id='test-id-001'"
            )
            row = await cursor.fetchone()

        assert row["status"] == "failed"
        assert "restarted" in row["error"].lower()


class TestCheckDbHealth:
    async def test_returns_true_when_healthy(self, tmp_path: Path):
        _, db_mod = await _fresh_db(tmp_path)
        assert await db_mod.check_db_health() is True

    async def test_returns_false_for_uninitialised_db(self, tmp_path: Path):
        """An SQLite file with no tables (no init_db called) → False."""
        db_path = tmp_path / "empty.db"
        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
        os.environ["DATA_DIR"] = str(tmp_path)

        import app.config as cfg
        importlib.reload(cfg)
        import app.database as db_mod
        importlib.reload(db_mod)

        # Create the file WITHOUT initialising tables
        async with aiosqlite.connect(db_path):
            pass  # just creates the empty file

        result = await db_mod.check_db_health()
        assert result is False

    async def test_returns_false_when_exception_raised(self, tmp_path: Path, monkeypatch):
        """Lines 112-114: exception handler in check_db_health returns False."""
        _, db_mod = await _fresh_db(tmp_path)

        # Make aiosqlite.connect raise so the except branch is taken
        def _raise(*_a, **_kw):
            raise OSError("simulated disk error")

        monkeypatch.setattr(aiosqlite, "connect", _raise)
        result = await db_mod.check_db_health()
        assert result is False


class TestGetDb:
    async def test_yields_working_connection(self, tmp_path: Path):
        """Lines 94-100: get_db() generator yields a functional connection."""
        _, db_mod = await _fresh_db(tmp_path)

        async for db in db_mod.get_db():
            cursor = await db.execute("SELECT 1")
            row = await cursor.fetchone()
            assert row[0] == 1
            break  # consume exactly one connection
