"""Database initialisation, connection management, and startup recovery.

Connection strategy
-------------------
* One ``aiosqlite`` connection is opened per request (or background task) via
  ``get_db()``.  This keeps the code simple and avoids connection-pool
  complexity that SQLite does not need.
* WAL mode is enabled once at startup so that the API server (reader) and the
  worker subprocess (writer) can coexist without lock contention.
* On startup, any job stuck in ``running`` status is reset to ``failed`` with
  an explanatory error message — this prevents permanently orphaned jobs after
  an unclean shutdown.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import aiosqlite

# Import the module (not the object) so that after test-time reloads of
# app.config the database functions always read the *current* settings instance.
from app import config as _config_module


def _settings():
    return _config_module.settings

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


async def _read_migration(filename: str) -> str:
    migration_path = _MIGRATIONS_DIR / filename
    return migration_path.read_text(encoding="utf-8")


async def init_db() -> None:
    """Create tables (idempotent) and enable WAL mode.

    Called once during application lifespan startup.
    """
    db_path = _settings().sqlite_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        # Enable WAL so worker subprocess writes don't block API reads.
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")

        sql = await _read_migration("001_initial.sql")
        await db.executescript(sql)
        await db.commit()

    logger.info("Database initialised at %s", db_path)
    await _recover_stuck_jobs()


async def _recover_stuck_jobs() -> None:
    """Reset jobs left in 'running' state from a previous crashed session."""
    async with aiosqlite.connect(_settings().sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id FROM jobs WHERE status = 'running'"
        )
        rows = await cursor.fetchall()
        if rows:
            ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(ids))
            await db.execute(
                f"UPDATE jobs SET status='failed', error='Server restarted while job was running' "  # noqa: S608  # nosec B608 — placeholders are only '?' characters, not user input
                f"WHERE id IN ({placeholders})",
                ids,
            )
            await db.commit()
            logger.warning(
                "Recovered %d stuck job(s) from previous session: %s",
                len(ids),
                ids,
            )


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Yield an open ``aiosqlite`` connection for use in a request or task.

    Usage::

        async def my_endpoint(db: aiosqlite.Connection = Depends(get_db)):
            ...
    """
    async with aiosqlite.connect(_settings().sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            pass  # context manager handles close


async def check_db_health() -> bool:
    """Return True if the database is reachable and the jobs table exists."""
    try:
        async with aiosqlite.connect(_settings().sqlite_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
            )
            row = await cursor.fetchone()
            return row is not None
    except Exception:  # noqa: BLE001
        logger.exception("Database health check failed")
        return False
