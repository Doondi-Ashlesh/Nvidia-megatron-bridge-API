"""Log parsing service — extracts structured progress from Megatron training logs.

Megatron-LM and MegatronBridge emit training progress in a well-known format:

  [2024-01-15 12:34:56] iteration      450/   1000 | consumed samples:      14400 |
      elapsed time per iteration (ms): 1250.3 | learning rate: 3.000E-05 |
      global batch size:    32 | lm loss: 1.3421E+00 | loss scale: 1.0 |
      grad norm: 0.812 | num zeros: 0.0 | number of skipped iterations:   0 |
      number of nan iterations:   0 | samples per second: 25.6 | TFLOPs: 42.3

This module provides:
  - parse_progress_line(): extract a ProgressFrame from a single log line
  - tail_log_for_progress(): scan the last N lines of a log file for the latest frame
  - update_job_progress_sync(): write a parsed frame to the jobs.progress column
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Megatron log patterns
# ---------------------------------------------------------------------------

# Step/total line — must be present for a valid training iteration
_ITER_STEP_RE = re.compile(
    r"iteration\s+(?P<step>\d+)\s*/\s*(?P<total_steps>\d+)"
)

# Individual field patterns — searched independently (no assumed ordering)
_LM_LOSS_RE = re.compile(r"lm loss:\s*(?P<loss>[\d.eE+\-]+)")
_LR_RE = re.compile(r"learning rate:\s*(?P<lr>[\d.eE+\-]+)")
_SPS_RE = re.compile(r"samples per second:\s*(?P<sps>[\d.]+)")
_TFLOPS_RE = re.compile(r"TFLOPs:\s*(?P<tflops>[\d.]+)")
_GRAD_NORM_RE = re.compile(r"grad norm:\s*(?P<grad_norm>[\d.eE+\-]+)")

# Validation loss line:
# "validation loss at iteration 500 | lm loss value: 1.2345E+00"
_VAL_RE = re.compile(
    r"validation loss.*?iteration\s+(?P<step>\d+)"
    r".*?lm loss value:\s*(?P<val_loss>[\d.eE+\-]+)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

ProgressFrame = dict[str, Any]


def parse_progress_line(line: str) -> ProgressFrame | None:
    """Parse a single log line and return a progress dict, or None if no match.

    Returned dict schema (all fields optional except step):
        step: int
        total_steps: int | None
        loss: float | None
        val_loss: float | None
        lr: float | None
        grad_norm: float | None
        samples_per_sec: float | None
        tflops: float | None
        tokens_per_sec: float | None   # derived from samples_per_sec * seq_length if available
    """
    # Check for validation line first
    val_match = _VAL_RE.search(line)
    if val_match:
        return {
            "step": int(val_match.group("step")),
            "val_loss": float(val_match.group("val_loss")),
        }

    step_m = _ITER_STEP_RE.search(line)
    if not step_m:
        return None

    frame: ProgressFrame = {
        "step": int(step_m.group("step")),
        "total_steps": int(step_m.group("total_steps")),
    }

    # Search for each field independently — no assumed ordering
    if m := _LM_LOSS_RE.search(line):
        frame["loss"] = float(m.group("loss"))
    if m := _LR_RE.search(line):
        frame["lr"] = float(m.group("lr"))
    if m := _SPS_RE.search(line):
        frame["samples_per_sec"] = float(m.group("sps"))
    if m := _TFLOPS_RE.search(line):
        frame["tflops"] = float(m.group("tflops"))
    if m := _GRAD_NORM_RE.search(line):
        frame["grad_norm"] = float(m.group("grad_norm"))

    return frame


def tail_log_for_progress(log_path: Path, tail_lines: int = 200) -> ProgressFrame | None:
    """Scan the last `tail_lines` of a log file and return the latest progress frame.

    Reads from the end of the file to find the most recent training iteration line.
    Returns None if the file doesn't exist or no progress lines are found.
    """
    if not log_path.exists():
        return None

    try:
        with log_path.open(encoding="utf-8", errors="replace") as f:
            # Efficient tail — read last chunk rather than entire file
            f.seek(0, 2)  # seek to end
            file_size = f.tell()
            chunk_size = min(file_size, tail_lines * 300)  # ~300 chars/line estimate
            f.seek(max(0, file_size - chunk_size))
            lines = f.read().splitlines()
    except OSError:
        return None

    latest: ProgressFrame | None = None
    for line in reversed(lines[-tail_lines:]):
        frame = parse_progress_line(line)
        if frame and "step" in frame and "loss" in frame:
            latest = frame
            break

    return latest


# ---------------------------------------------------------------------------
# DB writer (synchronous — called from executor subprocess)
# ---------------------------------------------------------------------------


def _get_db_path() -> str:
    import os  # noqa: PLC0415
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///megatronbridge.db")
    prefix = "sqlite+aiosqlite:///"
    return url[len(prefix):] if url.startswith(prefix) else url


def update_job_progress_sync(job_id: str, frame: ProgressFrame) -> None:
    """Write a progress frame to jobs.progress (JSON) using synchronous sqlite3.

    Called periodically from the executor during a training run so the
    WebSocket progress stream always has fresh data.
    """
    db_path = _get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE jobs SET progress = ? WHERE id = ?",
            (json.dumps(frame), job_id),
        )
        conn.commit()
    except (sqlite3.Error, OSError) as exc:
        logger.warning("Failed to update progress for job %s: %s", job_id, exc)
    finally:
        if conn is not None:
            conn.close()
