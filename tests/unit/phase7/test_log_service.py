"""Unit tests for app.services.log_service — Megatron log parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.log_service import (
    parse_progress_line,
    tail_log_for_progress,
    update_job_progress_sync,
)


# ---------------------------------------------------------------------------
# parse_progress_line
# ---------------------------------------------------------------------------


class TestParseProgressLine:
    def test_full_iteration_line(self):
        line = (
            "[2024-01-15 12:34:56] iteration      450/   1000 | consumed samples:      14400 |"
            " elapsed time per iteration (ms): 1250.3 | learning rate: 3.000E-05 |"
            " global batch size:    32 | lm loss: 1.3421E+00 | loss scale: 1.0 |"
            " grad norm: 0.812 | samples per second: 25.6 | TFLOPs: 42.3"
        )
        frame = parse_progress_line(line)

        assert frame is not None
        assert frame["step"] == 450
        assert frame["total_steps"] == 1000
        assert abs(frame["loss"] - 1.3421) < 0.001
        assert abs(frame["lr"] - 3e-5) < 1e-7
        assert abs(frame["grad_norm"] - 0.812) < 0.001
        assert abs(frame["samples_per_sec"] - 25.6) < 0.01
        assert abs(frame["tflops"] - 42.3) < 0.01

    def test_minimal_iteration_line(self):
        line = "iteration       1/  100 | lm loss: 2.5000E+00"
        frame = parse_progress_line(line)

        assert frame is not None
        assert frame["step"] == 1
        assert frame["total_steps"] == 100
        assert abs(frame["loss"] - 2.5) < 0.001

    def test_validation_loss_line(self):
        line = "validation loss at iteration 500 | lm loss value: 1.2345E+00"
        frame = parse_progress_line(line)

        assert frame is not None
        assert frame["step"] == 500
        assert abs(frame["val_loss"] - 1.2345) < 0.001

    def test_non_matching_line_returns_none(self):
        assert parse_progress_line("Starting training...") is None
        assert parse_progress_line("") is None
        assert parse_progress_line("Loading checkpoint from /data/ckpt") is None

    def test_step_one_correctly_parsed(self):
        line = "iteration        1/10000 | lm loss: 8.1200E+00"
        frame = parse_progress_line(line)
        assert frame["step"] == 1
        assert frame["total_steps"] == 10000


# ---------------------------------------------------------------------------
# tail_log_for_progress
# ---------------------------------------------------------------------------


class TestTailLogForProgress:
    def test_returns_none_for_missing_file(self, tmp_path):
        result = tail_log_for_progress(tmp_path / "nonexistent.log")
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text("")
        result = tail_log_for_progress(log)
        assert result is None

    def test_extracts_latest_iteration(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text(
            "iteration        1/  100 | lm loss: 8.0000E+00\n"
            "iteration       10/  100 | lm loss: 5.0000E+00\n"
            "iteration       50/  100 | lm loss: 2.5000E+00\n"
        )
        frame = tail_log_for_progress(log)

        assert frame is not None
        assert frame["step"] == 50
        assert abs(frame["loss"] - 2.5) < 0.001

    def test_ignores_non_progress_lines(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text(
            "Loading model weights...\n"
            "iteration       42/  200 | lm loss: 3.1400E+00\n"
            "Saving checkpoint...\n"
        )
        frame = tail_log_for_progress(log)

        assert frame is not None
        assert frame["step"] == 42


# ---------------------------------------------------------------------------
# update_job_progress_sync
# ---------------------------------------------------------------------------


class TestUpdateJobProgressSync:
    def test_writes_progress_to_db(self, tmp_path, monkeypatch):
        import sqlite3

        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, progress TEXT)"
        )
        conn.execute("INSERT INTO jobs (id, progress) VALUES ('job-1', NULL)")
        conn.commit()
        conn.close()

        monkeypatch.setenv(
            "DATABASE_URL", f"sqlite+aiosqlite:///{db}"
        )

        frame = {"step": 100, "loss": 1.5, "gpus": []}
        update_job_progress_sync("job-1", frame)

        conn2 = sqlite3.connect(str(db))
        row = conn2.execute(
            "SELECT progress FROM jobs WHERE id='job-1'"
        ).fetchone()
        conn2.close()

        assert row is not None
        stored = json.loads(row[0])
        assert stored["step"] == 100
        assert abs(stored["loss"] - 1.5) < 0.001

    def test_handles_missing_job_gracefully(self, tmp_path, monkeypatch):
        """update_job_progress_sync does not raise if job_id doesn't exist."""
        import sqlite3

        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, progress TEXT)")
        conn.commit()
        conn.close()

        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")

        # Should not raise
        update_job_progress_sync("nonexistent-id", {"step": 1})

    def test_handles_bad_db_path_gracefully(self, monkeypatch):
        """update_job_progress_sync logs warning but does not raise on bad DB path."""
        monkeypatch.setenv(
            "DATABASE_URL", "sqlite+aiosqlite:////nonexistent/path/db.sqlite"
        )
        # Should not raise
        update_job_progress_sync("job-1", {"step": 1})
