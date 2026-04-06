"""Asyncio dispatcher loop: polls SQLite for queued jobs and launches workers.

The dispatcher runs as a background asyncio Task started during the FastAPI
lifespan.  It:
  1. Polls the ``jobs`` table every ``dispatcher_poll_interval_s`` seconds.
  2. If the number of RUNNING jobs is below ``max_concurrent_jobs``, it
     picks the oldest QUEUED job and launches a torchrun worker for it.
  3. Monitors running workers; marks them COMPLETED or FAILED when they exit.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app import config as _config_module
from app.services import job_service
from app.worker import launcher

logger = logging.getLogger(__name__)


def _settings():  # type: ignore[return]
    return _config_module.settings


# ---------------------------------------------------------------------------
# In-memory registry of active workers
# ---------------------------------------------------------------------------

#: Maps job_id → subprocess.Popen
_active_workers: dict[str, Any] = {}


def get_active_workers() -> dict[str, Any]:
    """Return reference to the active workers registry (for testing)."""
    return _active_workers


# ---------------------------------------------------------------------------
# Core dispatcher logic (separated for unit-testing without sleeping)
# ---------------------------------------------------------------------------


async def _poll_once(db_getter) -> None:  # type: ignore[type-arg]
    """Single tick of the dispatcher loop.

    Parameters
    ----------
    db_getter:
        Async callable (no args) that returns an aiosqlite connection
        as an async context manager, e.g. ``aiosqlite.connect(path)``.
    """
    settings = _settings()

    async with db_getter() as db:
        # ── 1. Collect exit status of finished workers ────────────────────
        finished_ids = [
            job_id
            for job_id, proc in list(_active_workers.items())
            if proc.poll() is not None
        ]
        for job_id in finished_ids:
            proc = _active_workers.pop(job_id)
            returncode: int = proc.returncode
            if returncode == 0:
                await job_service.transition_to_completed(db, job_id)
            else:
                await job_service.transition_to_failed(
                    db,
                    job_id,
                    error=f"Worker exited with return code {returncode}",
                )

        # ── 2. Launch new workers if capacity allows ──────────────────────
        running_count = len(_active_workers)
        capacity = settings.max_concurrent_jobs - running_count
        if capacity <= 0:
            return

        queued_items, _ = await job_service.list_jobs(
            db,
            status=job_service.JobStatus.QUEUED,
            limit=capacity,
        )

        for job_row in queued_items:
            job_id = job_row["id"]
            num_gpus = int(job_row.get("num_gpus", 1))
            log_path = Path(settings.logs_root) / f"{job_id}.log"

            try:
                proc = launcher.launch_worker(
                    job_row,
                    log_path,
                    cuda_visible_devices=settings.cuda_visible_devices,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to launch worker for job %s: %s", job_id, exc)
                await job_service.transition_to_failed(
                    db, job_id, error=f"Launcher error: {exc}"
                )
                continue

            _active_workers[job_id] = proc
            await job_service.transition_to_running(
                db,
                job_id,
                pid=proc.pid,
                log_path=str(log_path),
            )
            logger.info(
                "Dispatched job %s (gpus=%d, pid=%d)", job_id, num_gpus, proc.pid
            )


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------


async def dispatcher_loop(db_getter=None) -> None:  # type: ignore[assignment]
    """Infinite asyncio loop; runs until cancelled.

    Parameters
    ----------
    db_getter:
        Optional override for the database connection factory (used in tests).
        Defaults to a factory that opens the configured SQLite file.
    """
    import aiosqlite

    if db_getter is None:

        def db_getter():  # type: ignore[misc]
            return aiosqlite.connect(_settings().sqlite_path)

    settings = _settings()
    logger.info(
        "Dispatcher started (poll_interval=%.1fs, max_concurrent=%d)",
        settings.dispatcher_poll_interval_s,
        settings.max_concurrent_jobs,
    )

    while True:
        try:
            await _poll_once(db_getter)
        except asyncio.CancelledError:
            logger.info("Dispatcher loop cancelled.")
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Dispatcher encountered an unexpected error; continuing.")

        await asyncio.sleep(settings.dispatcher_poll_interval_s)
