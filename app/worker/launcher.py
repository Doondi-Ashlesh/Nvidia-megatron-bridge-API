"""Torchrun process launcher with strict argument validation.

Security invariants (verified by bandit + unit tests):
- shell=False on every Popen call (no shell injection possible)
- All torchrun flags come from an explicit whitelist; unknown flags raise LauncherValidationError
- CUDA_VISIBLE_DEVICES is validated to contain only digits and commas
- The job-id argument is validated as a UUID before being passed to the worker
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

#: Regex matching a strict UUID v4 (hyphenated lowercase hex)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

#: Regex matching a safe CUDA_VISIBLE_DEVICES value (digits and commas only)
_CUDA_RE = re.compile(r"^[0-9]+(,[0-9]+)*$|^$")

#: Allowed torchrun long-option names (no others are ever forwarded to the subprocess)
_ALLOWED_TORCHRUN_FLAGS: frozenset[str] = frozenset(
    {
        "--nproc-per-node",
        "--nnodes",
        "--master-addr",
        "--master-port",
        "--rdzv-backend",
        "--rdzv-endpoint",
        "--rdzv-id",
    }
)


class LauncherValidationError(ValueError):
    """Raised when launcher arguments fail validation."""


def _validate_job_id(job_id: str) -> None:
    """Raise LauncherValidationError if job_id is not a valid UUID4."""
    if not _UUID_RE.match(job_id.lower()):
        msg = f"job_id must be a UUID4, got {job_id!r}"
        raise LauncherValidationError(msg)


def _validate_cuda_devices(value: str) -> None:
    """Raise LauncherValidationError if CUDA_VISIBLE_DEVICES is unsafe."""
    if not _CUDA_RE.match(value):
        msg = f"CUDA_VISIBLE_DEVICES contains invalid characters: {value!r}"
        raise LauncherValidationError(msg)


def _validate_num_gpus(num_gpus: int) -> None:
    if num_gpus < 1 or num_gpus > 64:
        msg = f"num_gpus must be between 1 and 64, got {num_gpus}"
        raise LauncherValidationError(msg)


def _validate_torchrun_flags(extra_flags: dict[str, str]) -> None:
    """Raise LauncherValidationError if any key is not in the whitelist."""
    for flag in extra_flags:
        if flag not in _ALLOWED_TORCHRUN_FLAGS:
            msg = f"Torchrun flag {flag!r} is not in the allowed whitelist"
            raise LauncherValidationError(msg)


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------


def find_free_port() -> int:
    """Bind to port 0 and return the OS-assigned ephemeral port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def launch_worker(
    job: dict[str, Any],
    log_path: Path,
    *,
    cuda_visible_devices: str = "0",
    extra_torchrun_flags: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch a torchrun worker subprocess for *job*.

    Parameters
    ----------
    job:
        Job dict from the database (must contain ``id`` and ``num_gpus``).
    log_path:
        File where stdout + stderr of the worker are redirected.
    cuda_visible_devices:
        Comma-separated GPU indices (e.g. ``"0,1"``).  Validated to only
        contain digits and commas — no shell-escape sequences.
    extra_torchrun_flags:
        Optional ``{flag: value}`` map.  Every key must be in
        ``_ALLOWED_TORCHRUN_FLAGS``; any unknown flag raises
        ``LauncherValidationError``.

    Returns
    -------
    subprocess.Popen
        The running torchrun process (group leader).  Use ``os.killpg``
        with ``os.getpgid(proc.pid)`` to SIGTERM all worker ranks.
    """
    job_id: str = job["id"]
    num_gpus: int = int(job.get("num_gpus", 1))
    extra_flags: dict[str, str] = extra_torchrun_flags or {}

    # ── Validate all inputs before touching subprocess ────────────────────
    _validate_job_id(job_id)
    _validate_num_gpus(num_gpus)
    _validate_cuda_devices(cuda_visible_devices)
    _validate_torchrun_flags(extra_flags)

    master_port = str(find_free_port())

    # ── Build command from validated, typed arguments ─────────────────────
    cmd: list[str] = [
        "torchrun",
        f"--nproc-per-node={num_gpus}",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
    ]

    # Append whitelisted extra flags
    for flag, value in extra_flags.items():
        cmd.append(f"{flag}={value}")

    cmd += ["-m", "app.worker.executor", "--job-id", job_id]

    # ── Build environment ─────────────────────────────────────────────────
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_visible_devices}

    # ── Ensure log directory exists ───────────────────────────────────────
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")

    logger.info(
        "Launching worker for job %s: nproc=%d gpus=%s port=%s",
        job_id,
        num_gpus,
        cuda_visible_devices,
        master_port,
    )

    proc = subprocess.Popen(  # noqa: S603  # nosec B603 — shell=False, cmd fully validated
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        shell=False,  # explicit: never shell=True
        start_new_session=True,  # creates new process group for SIGTERM-all-ranks
    )
    log_file.close()  # parent doesn't need the fd after fork
    logger.info("Worker PID %d launched for job %s", proc.pid, job_id)
    return proc


# ---------------------------------------------------------------------------
# Cancellation helper
# ---------------------------------------------------------------------------


def terminate_worker(pid: int) -> None:
    """Send SIGTERM to the entire process group led by *pid*.

    On Windows, falls back to ``proc.terminate()`` because process groups
    work differently.  The test suite mocks this function entirely.
    """
    import signal
    import sys

    if sys.platform == "win32":
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            logger.warning("Could not terminate PID %d: %s", pid, exc)
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            logger.info("Sent SIGTERM to process group %d (leader %d)", pgid, pid)
        except OSError as exc:
            logger.warning("Could not terminate process group for PID %d: %s", pid, exc)


# ---------------------------------------------------------------------------
# UUID helper (re-exported for tests)
# ---------------------------------------------------------------------------


def is_valid_uuid(value: str) -> bool:
    """Return True if *value* is a well-formed UUID (any version)."""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False
