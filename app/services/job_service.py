"""Job CRUD operations and state machine transitions.

All writes go through the explicit ``transition_*`` functions so that invalid
state changes are detected before they touch the database.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from app.schemas.common import JobStatus, JobType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State machine: allowed transitions
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLING,
    },
    JobStatus.CANCELLING: {JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}


class InvalidStateTransitionError(Exception):
    """Raised when a state transition is not permitted."""


def _assert_transition(current: JobStatus, target: JobStatus) -> None:
    allowed = _ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        msg = f"Cannot transition job from {current!r} to {target!r}"
        raise InvalidStateTransitionError(msg)


# ---------------------------------------------------------------------------
# Row → dict helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    """Convert an aiosqlite Row (with row_factory) to a plain dict."""
    return dict(row)


def _parse_json_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Deserialize JSON-encoded columns (payload, progress) in-place."""
    for col in ("payload", "progress"):
        raw = d.get(col)
        if raw and isinstance(raw, str):
            with suppress(json.JSONDecodeError):
                d[col] = json.loads(raw)
    return d


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


async def get_job(db: aiosqlite.Connection, job_id: str) -> dict[str, Any] | None:
    """Fetch a single job by ID. Returns None if not found."""
    db.row_factory = aiosqlite.Row
    cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    return _parse_json_fields(_row_to_dict(row))


async def list_jobs(
    db: aiosqlite.Connection,
    *,
    status: JobStatus | None = None,
    job_type: JobType | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return (items, total_count) with optional filters."""
    db.row_factory = aiosqlite.Row

    conditions: list[str] = []
    params: list[Any] = []

    if status is not None:
        conditions.append("status = ?")
        params.append(status.value)
    if job_type is not None:
        conditions.append("type = ?")
        params.append(job_type.value)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    count_cursor = await db.execute(
        f"SELECT COUNT(*) FROM jobs {where}",  # noqa: S608  # nosec B608 — conditions built from enum values only
        params,
    )
    count_row = await count_cursor.fetchone()
    total: int = count_row[0] if count_row else 0

    cursor = await db.execute(
        f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # noqa: S608  # nosec B608
        [*params, limit, offset],
    )
    rows = await cursor.fetchall()
    return [_parse_json_fields(_row_to_dict(r)) for r in rows], total


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


async def create_job(
    db: aiosqlite.Connection,
    *,
    job_type: JobType,
    payload: dict[str, Any],
    num_gpus: int = 1,
) -> str:
    """Insert a new QUEUED job. Returns the new job ID."""
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO jobs (id, type, status, payload, num_gpus, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, job_type.value, JobStatus.QUEUED.value, json.dumps(payload), num_gpus, now),
    )
    await db.commit()
    logger.info("Job %s created (type=%s, gpus=%d)", job_id, job_type.value, num_gpus)
    return job_id


async def transition_to_running(
    db: aiosqlite.Connection,
    job_id: str,
    *,
    pid: int,
    log_path: str,
) -> None:
    """Mark a job as RUNNING with its worker PID and log path."""
    row = await get_job(db, job_id)
    if row is None:
        msg = f"Job {job_id!r} not found"
        raise ValueError(msg)
    current = JobStatus(row["status"])
    _assert_transition(current, JobStatus.RUNNING)

    now = datetime.now(UTC).isoformat()
    await db.execute(
        "UPDATE jobs SET status=?, pid=?, log_path=?, started_at=? WHERE id=?",
        (JobStatus.RUNNING.value, pid, log_path, now, job_id),
    )
    await db.commit()
    logger.info("Job %s → running (pid=%d)", job_id, pid)


async def transition_to_completed(
    db: aiosqlite.Connection,
    job_id: str,
    *,
    progress: dict[str, Any] | None = None,
) -> None:
    """Mark a job as COMPLETED."""
    row = await get_job(db, job_id)
    if row is None:
        msg = f"Job {job_id!r} not found"
        raise ValueError(msg)
    current = JobStatus(row["status"])
    _assert_transition(current, JobStatus.COMPLETED)

    now = datetime.now(UTC).isoformat()
    await db.execute(
        "UPDATE jobs SET status=?, progress=?, completed_at=? WHERE id=?",
        (
            JobStatus.COMPLETED.value,
            json.dumps(progress) if progress is not None else None,
            now,
            job_id,
        ),
    )
    await db.commit()
    logger.info("Job %s → completed", job_id)


async def transition_to_failed(
    db: aiosqlite.Connection,
    job_id: str,
    *,
    error: str,
) -> None:
    """Mark a job as FAILED with an error message."""
    row = await get_job(db, job_id)
    if row is None:
        msg = f"Job {job_id!r} not found"
        raise ValueError(msg)
    current = JobStatus(row["status"])
    _assert_transition(current, JobStatus.FAILED)

    now = datetime.now(UTC).isoformat()
    await db.execute(
        "UPDATE jobs SET status=?, error=?, completed_at=? WHERE id=?",
        (JobStatus.FAILED.value, error, now, job_id),
    )
    await db.commit()
    logger.warning("Job %s → failed: %s", job_id, error)


async def transition_to_cancelling(
    db: aiosqlite.Connection,
    job_id: str,
) -> int | None:
    """Mark a running job as CANCELLING. Returns the worker PID for SIGTERM."""
    row = await get_job(db, job_id)
    if row is None:
        msg = f"Job {job_id!r} not found"
        raise ValueError(msg)
    current = JobStatus(row["status"])
    _assert_transition(current, JobStatus.CANCELLING)

    await db.execute(
        "UPDATE jobs SET status=? WHERE id=?",
        (JobStatus.CANCELLING.value, job_id),
    )
    await db.commit()
    logger.info("Job %s → cancelling", job_id)
    return row.get("pid")


async def transition_to_cancelled(
    db: aiosqlite.Connection,
    job_id: str,
) -> None:
    """Mark a QUEUED or CANCELLING job as CANCELLED."""
    row = await get_job(db, job_id)
    if row is None:
        msg = f"Job {job_id!r} not found"
        raise ValueError(msg)
    current = JobStatus(row["status"])
    _assert_transition(current, JobStatus.CANCELLED)

    now = datetime.now(UTC).isoformat()
    await db.execute(
        "UPDATE jobs SET status=?, completed_at=? WHERE id=?",
        (JobStatus.CANCELLED.value, now, job_id),
    )
    await db.commit()
    logger.info("Job %s → cancelled", job_id)


async def update_progress(
    db: aiosqlite.Connection,
    job_id: str,
    progress: dict[str, Any],
) -> None:
    """Update the progress JSON blob for a running job."""
    await db.execute(
        "UPDATE jobs SET progress=? WHERE id=?",
        (json.dumps(progress), job_id),
    )
    await db.commit()
