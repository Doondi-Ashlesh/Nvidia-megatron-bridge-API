"""REST endpoints for job management: GET/DELETE /v1/jobs."""

from __future__ import annotations

import logging

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from app.database import get_db
from app.schemas.common import (
    ErrorDetail,
    JobListResponse,
    JobResponse,
    JobStatus,
    JobType,
)
from app.services import job_service
from app.worker.launcher import is_valid_uuid, terminate_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_row_to_response(row: dict) -> JobResponse:  # type: ignore[type-arg]
    """Convert a raw DB row dict to a JobResponse Pydantic model."""
    return JobResponse(
        id=row["id"],
        type=JobType(row["type"]),
        status=JobStatus(row["status"]),
        payload=row.get("payload") or {},
        error=row.get("error"),
        progress=row.get("progress"),
        log_path=row.get("log_path"),
        pid=row.get("pid"),
        num_gpus=row.get("num_gpus", 1),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
    )


def _assert_job_exists(row: dict | None, job_id: str) -> dict:  # type: ignore[type-arg]
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return row


# ---------------------------------------------------------------------------
# GET /v1/jobs
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=JobListResponse,
    summary="List all jobs",
)
async def list_jobs(
    status: JobStatus | None = Query(default=None, description="Filter by status"),
    job_type: JobType | None = Query(default=None, description="Filter by type"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: aiosqlite.Connection = Depends(get_db),
) -> JobListResponse:
    items, total = await job_service.list_jobs(
        db,
        status=status,
        job_type=job_type,
        limit=limit,
        offset=offset,
    )
    return JobListResponse(
        items=[_job_row_to_response(r) for r in items],
        total=total,
    )


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get a single job",
    responses={404: {"model": ErrorDetail}},
)
async def get_job(
    job_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> JobResponse:
    if not is_valid_uuid(job_id):
        raise HTTPException(status_code=400, detail="job_id must be a valid UUID")
    row = _assert_job_exists(await job_service.get_job(db, job_id), job_id)
    return _job_row_to_response(row)


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id}/logs
# ---------------------------------------------------------------------------


@router.get(
    "/{job_id}/logs",
    response_model=None,
    summary="Fetch job log file (HTTP fallback)",
    responses={404: {"model": ErrorDetail}},
)
async def get_job_logs(
    job_id: str,
    line_offset: int = Query(default=0, ge=0, description="Skip first N lines"),
    line_limit: int = Query(default=2000, ge=1, le=10000, description="Max lines to return"),
    db: aiosqlite.Connection = Depends(get_db),
):  # type: ignore[return]
    """Return log file lines as plain text, with optional line-based pagination."""
    import aiofiles
    from fastapi.responses import PlainTextResponse

    from app import config as _cfg
    from app.utils.paths import PathTraversalError, safe_log_path

    if not is_valid_uuid(job_id):
        raise HTTPException(status_code=400, detail="job_id must be a valid UUID")

    row = _assert_job_exists(await job_service.get_job(db, job_id), job_id)
    log_path_str = row.get("log_path")

    if not log_path_str:
        return PlainTextResponse("")

    try:
        log_path = safe_log_path(_cfg.settings.logs_root, job_id)
    except PathTraversalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not log_path.exists():
        return PlainTextResponse("")

    async with aiofiles.open(log_path, encoding="utf-8", errors="replace") as f:
        content = await f.read()

    lines = content.splitlines()
    page = lines[line_offset: line_offset + line_limit]
    return PlainTextResponse("\n".join(page))


# ---------------------------------------------------------------------------
# DELETE /v1/jobs/{job_id}  (cancel)
# ---------------------------------------------------------------------------


_TERMINAL_STATUSES = frozenset({
    JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED,
})


@router.delete(
    "/{job_id}",
    status_code=202,
    summary="Cancel an active job, or permanently delete a terminal job and its log file",
    responses={404: {"model": ErrorDetail}, 409: {"model": ErrorDetail}},
)
async def cancel_or_delete_job(
    job_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:  # type: ignore[type-arg]
    """
    - **QUEUED / RUNNING**: cancels the job (returns 202 with new status).
    - **COMPLETED / FAILED / CANCELLED**: permanently deletes the DB record
      and removes the associated log file from disk (returns 202 with `deleted: true`).
    """
    import aiofiles.os  # noqa: PLC0415

    from app import config as _cfg  # noqa: PLC0415
    from app.utils.paths import PathTraversalError, safe_log_path  # noqa: PLC0415

    if not is_valid_uuid(job_id):
        raise HTTPException(status_code=400, detail="job_id must be a valid UUID")

    row = _assert_job_exists(await job_service.get_job(db, job_id), job_id)
    current_status = JobStatus(row["status"])

    # Terminal jobs: delete record + log file
    if current_status in _TERMINAL_STATUSES:
        await job_service.delete_job(db, job_id)

        # Best-effort log file removal — don't 500 if file is already gone
        try:
            log_path = safe_log_path(_cfg.settings.logs_root, job_id)
            if log_path.exists():
                await aiofiles.os.remove(log_path)
        except (PathTraversalError, OSError):
            pass

        return {"job_id": job_id, "deleted": True}

    try:
        if current_status == JobStatus.QUEUED:
            await job_service.transition_to_cancelled(db, job_id)
            return {"job_id": job_id, "status": JobStatus.CANCELLED}

        if current_status == JobStatus.RUNNING:
            pid = await job_service.transition_to_cancelling(db, job_id)
            if pid:
                terminate_worker(pid)
            return {"job_id": job_id, "status": JobStatus.CANCELLING}

        # CANCELLING — already on the way out
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job in status {current_status.value!r}",
        )

    except job_service.InvalidStateTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
