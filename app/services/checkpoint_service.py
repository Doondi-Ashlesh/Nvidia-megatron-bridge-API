"""Checkpoint CRUD operations and metadata utilities."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


async def register_checkpoint(
    db: aiosqlite.Connection,
    *,
    name: str,
    fmt: str,
    path: str,
    size_bytes: int | None = None,
    model_arch: str | None = None,
    created_from_job: str | None = None,
) -> str:
    """Insert a new checkpoint record. Returns the new checkpoint ID."""
    checkpoint_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO checkpoints "
        "(id, name, format, path, size_bytes, model_arch, created_from_job, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (checkpoint_id, name, fmt, path, size_bytes, model_arch, created_from_job, now),
    )
    await db.commit()
    logger.info("Checkpoint %s registered (name=%s, format=%s)", checkpoint_id, name, fmt)
    return checkpoint_id


async def get_checkpoint(
    db: aiosqlite.Connection,
    checkpoint_id: str,
) -> dict[str, Any] | None:
    """Fetch a single checkpoint by ID. Returns None if not found."""
    db.row_factory = aiosqlite.Row
    cursor = await db.execute(
        "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return dict(row)


async def list_checkpoints(
    db: aiosqlite.Connection,
    *,
    fmt: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return (items, total_count) with optional format filter."""
    db.row_factory = aiosqlite.Row

    conditions: list[str] = []
    params: list[Any] = []

    if fmt is not None:
        conditions.append("format = ?")
        params.append(fmt)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    count_cursor = await db.execute(
        f"SELECT COUNT(*) FROM checkpoints {where}",  # noqa: S608  # nosec B608
        params,
    )
    count_row = await count_cursor.fetchone()
    total: int = count_row[0] if count_row else 0

    cursor = await db.execute(
        f"SELECT * FROM checkpoints {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # noqa: S608  # nosec B608
        [*params, limit, offset],
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows], total


async def delete_checkpoint(
    db: aiosqlite.Connection,
    checkpoint_id: str,
) -> bool:
    """Delete a checkpoint record by ID. Returns True if deleted, False if not found."""
    cursor = await db.execute(
        "DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,)
    )
    await db.commit()
    deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Checkpoint %s deleted from DB", checkpoint_id)
    return deleted


async def get_checkpoint_size(path: Path) -> int | None:
    """Walk a directory and sum file sizes. Returns None on OSError or if path doesn't exist."""
    try:
        if not path.exists():
            return None
        if path.is_file():
            return path.stat().st_size
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total
    except OSError:
        return None
