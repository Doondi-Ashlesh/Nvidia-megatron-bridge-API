"""WebSocket endpoints for real-time job log streaming and progress tracking."""

from __future__ import annotations

import asyncio
import json
import logging

import aiofiles
import aiosqlite
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import config as _cfg
from app.schemas.common import JobStatus
from app.services import gpu_service, job_service
from app.utils.paths import PathTraversalError, safe_log_path
from app.worker.launcher import is_valid_uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ws", tags=["WebSocket"])

# Terminal states: no more transitions possible after these
TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)

_LOG_POLL_INTERVAL = 0.2  # seconds between log file polls
_PROGRESS_POLL_INTERVAL = 2.0  # seconds between progress polls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_job_ws(websocket: WebSocket, job_id: str) -> dict | None:  # type: ignore[type-arg]
    """Validate job_id and fetch the job row.

    Returns the job dict on success, or closes the WebSocket with 1008 and
    returns None on validation / lookup failure.
    """
    if not is_valid_uuid(job_id):
        await websocket.close(code=1008)
        return None

    async with aiosqlite.connect(_cfg.settings.sqlite_path) as db:
        row = await job_service.get_job(db, job_id)

    if row is None:
        await websocket.close(code=1008)
        return None

    return row


# ---------------------------------------------------------------------------
# WS /v1/ws/jobs/{job_id}/logs
# ---------------------------------------------------------------------------


@router.websocket("/jobs/{job_id}/logs")
async def ws_job_logs(websocket: WebSocket, job_id: str) -> None:
    """Stream log file lines to the client as they are written.

    Protocol:
    - Validate job_id (UUID) — close 1008 if invalid
    - Look up job in DB — close 1008 if not found
    - Send all existing log lines (catch-up)
    - Poll every 200 ms for new lines
    - When job reaches terminal state and no more new lines, send
      ``{"type": "stream_end", "status": "<status>"}`` and close
    """
    if not is_valid_uuid(job_id):
        await websocket.accept()
        await websocket.close(code=1008)
        return

    async with aiosqlite.connect(_cfg.settings.sqlite_path) as db:
        row = await job_service.get_job(db, job_id)

    if row is None:
        await websocket.accept()
        await websocket.close(code=1008)
        return

    await websocket.accept()

    try:
        # Determine log file path (defence-in-depth: validate even though UUID is checked)
        try:
            log_path = safe_log_path(_cfg.settings.logs_root, job_id)
        except PathTraversalError:
            await websocket.close(code=1008)
            return

        lines_sent = 0

        while True:
            # Re-fetch job status each iteration
            async with aiosqlite.connect(_cfg.settings.sqlite_path) as db:
                current_row = await job_service.get_job(db, job_id)

            if current_row is None:
                break

            current_status = JobStatus(current_row["status"])

            # Read whatever lines are available now
            new_lines: list[str] = []
            if log_path.exists():
                try:
                    async with aiofiles.open(log_path, encoding="utf-8", errors="replace") as f:
                        all_lines = await f.readlines()
                    new_lines = all_lines[lines_sent:]
                    lines_sent += len(new_lines)
                except OSError:
                    pass

            for line in new_lines:
                try:
                    await websocket.send_text(line.rstrip("\n"))
                except WebSocketDisconnect:
                    return

            # If terminal and nothing new to send, close
            if current_status in TERMINAL_STATUSES and not new_lines:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "stream_end", "status": current_status.value})
                    )
                except WebSocketDisconnect:
                    return
                await websocket.close()
                return

            await asyncio.sleep(_LOG_POLL_INTERVAL)

    except WebSocketDisconnect:
        logger.debug("Client disconnected from log stream for job %s", job_id)


# ---------------------------------------------------------------------------
# WS /v1/ws/jobs/{job_id}/progress
# ---------------------------------------------------------------------------


@router.websocket("/jobs/{job_id}/progress")
async def ws_job_progress(websocket: WebSocket, job_id: str) -> None:
    """Stream job progress as JSON frames every 2 seconds.

    Protocol:
    - Validate job_id (UUID) — close 1008 if invalid
    - Look up job in DB — close 1008 if not found
    - Poll DB every 2 s, send progress frame with GPU telemetry
    - When job reaches terminal state, send
      ``{"type": "job_complete", "status": "<status>"}`` and close
    """
    if not is_valid_uuid(job_id):
        await websocket.accept()
        await websocket.close(code=1008)
        return

    async with aiosqlite.connect(_cfg.settings.sqlite_path) as db:
        row = await job_service.get_job(db, job_id)

    if row is None:
        await websocket.accept()
        await websocket.close(code=1008)
        return

    await websocket.accept()

    try:
        while True:
            async with aiosqlite.connect(_cfg.settings.sqlite_path) as db:
                current_row = await job_service.get_job(db, job_id)

            if current_row is None:
                break

            current_status = JobStatus(current_row["status"])

            if current_status in TERMINAL_STATUSES:
                try:
                    await websocket.send_text(
                        json.dumps(
                            {"type": "job_complete", "status": current_status.value}
                        )
                    )
                except WebSocketDisconnect:
                    return
                await websocket.close()
                return

            # Build progress frame
            gpus = gpu_service.get_all_gpu_info()
            progress_value = current_row.get("progress")

            frame = {
                "type": "progress",
                "job_id": job_id,
                "status": current_status.value,
                "progress": progress_value,
                "gpus": gpus,
            }

            try:
                await websocket.send_text(json.dumps(frame))
            except WebSocketDisconnect:
                return

            await asyncio.sleep(_PROGRESS_POLL_INTERVAL)

    except WebSocketDisconnect:
        logger.debug("Client disconnected from progress stream for job %s", job_id)
