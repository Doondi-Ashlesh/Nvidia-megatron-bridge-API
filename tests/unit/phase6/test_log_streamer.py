"""Phase 6 unit tests — WebSocket log streaming (/v1/ws/jobs/{id}/logs)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import aiosqlite
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_client(test_app):
    """Synchronous TestClient for WebSocket testing."""
    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c


async def _seed_job(db_path: str, status: str = "queued") -> str:
    """Insert a job directly into the DB and return its job_id."""
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO jobs (id, type, status, payload, num_gpus, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, "pretrain", status, "{}", 1, now),
        )
        await db.commit()
    return job_id


def _db_path(test_app) -> str:
    import app.config as cfg
    url = cfg.settings.database_url
    return url[len("sqlite+aiosqlite:///"):]


# ---------------------------------------------------------------------------
# UUID validation
# ---------------------------------------------------------------------------


class TestLogStreamUuidValidation:
    def test_non_uuid_closes_with_1008(self, sync_client):
        with sync_client.websocket_connect("/v1/ws/jobs/not-a-uuid/logs") as ws:
            # After sending the close the server won't send data — just check close code
            data = ws.receive()
            assert data["type"] == "websocket.close"
            assert data.get("code") == 1008

    def test_traversal_path_closes_with_1008(self, sync_client):
        # %2F decodes to "/" breaking the path template — ASGI rejects before our
        # handler, raising WebSocketDisconnect.  Both outcomes (close OR exception)
        # confirm the traversal attempt is rejected without leaking content.
        try:
            with sync_client.websocket_connect(
                "/v1/ws/jobs/..%2F..%2Fetc%2Fpasswd/logs"
            ) as ws:
                data = ws.receive()
                assert data["type"] == "websocket.close"
        except WebSocketDisconnect:
            pass  # Routing rejection — traversal denied

    def test_long_string_closes_gracefully(self, sync_client):
        long_id = "a" * 200
        with sync_client.websocket_connect(f"/v1/ws/jobs/{long_id}/logs") as ws:
            data = ws.receive()
            assert data["type"] == "websocket.close"


# ---------------------------------------------------------------------------
# Missing job
# ---------------------------------------------------------------------------


class TestLogStreamMissingJob:
    def test_unknown_uuid_closes_with_1008(self, sync_client):
        fake_id = str(uuid.uuid4())
        with sync_client.websocket_connect(f"/v1/ws/jobs/{fake_id}/logs") as ws:
            data = ws.receive()
            assert data["type"] == "websocket.close"
            assert data.get("code") == 1008


# ---------------------------------------------------------------------------
# Streaming existing log lines (catch-up)
# ---------------------------------------------------------------------------


class TestLogStreamCatchUp:
    async def test_existing_log_lines_delivered(self, sync_client, test_app, tmp_path):
        """Lines already in the log file are sent immediately on connect."""
        import app.config as cfg

        db_path = _db_path(test_app)
        job_id = await _seed_job(db_path, status="running")

        # Write log file
        log_dir = cfg.settings.logs_root
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{job_id}.log"
        log_file.write_text("step 1\nstep 2\n")

        # Mark job completed so the stream terminates
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE jobs SET status='completed', completed_at=? WHERE id=?",
                (datetime.now(UTC).isoformat(), job_id),
            )
            await db.commit()

        received_texts = []
        with sync_client.websocket_connect(f"/v1/ws/jobs/{job_id}/logs") as ws:
            while True:
                msg = ws.receive()
                if msg["type"] == "websocket.close":
                    break
                if msg["type"] == "websocket.send":
                    text = msg.get("text", "")
                    received_texts.append(text)
                    # Stop after getting the stream_end message
                    try:
                        parsed = json.loads(text)
                        if parsed.get("type") == "stream_end":
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass

        assert "step 1" in received_texts
        assert "step 2" in received_texts

    async def test_stream_end_message_sent_on_terminal_state(self, sync_client, test_app):
        """stream_end JSON message is sent when job reaches completed state."""
        import app.config as cfg

        db_path = _db_path(test_app)
        job_id = await _seed_job(db_path, status="completed")

        # Ensure no log file so the loop hits terminal immediately
        log_dir = cfg.settings.logs_root
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{job_id}.log"
        if log_file.exists():
            log_file.unlink()

        received_texts = []
        with sync_client.websocket_connect(f"/v1/ws/jobs/{job_id}/logs") as ws:
            for _ in range(20):  # cap iterations
                msg = ws.receive()
                if msg["type"] == "websocket.close":
                    break
                if msg["type"] == "websocket.send":
                    text = msg.get("text", "")
                    received_texts.append(text)
                    try:
                        parsed = json.loads(text)
                        if parsed.get("type") == "stream_end":
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass

        stream_end_msgs = [
            json.loads(t) for t in received_texts
            if _is_json(t) and json.loads(t).get("type") == "stream_end"
        ]
        assert len(stream_end_msgs) >= 1
        assert stream_end_msgs[0]["status"] == "completed"

    async def test_stream_end_on_failed_job(self, sync_client, test_app):
        """stream_end is also sent for failed jobs."""
        import app.config as cfg

        db_path = _db_path(test_app)
        job_id = await _seed_job(db_path, status="failed")

        log_dir = cfg.settings.logs_root
        log_dir.mkdir(parents=True, exist_ok=True)

        received_texts = []
        with sync_client.websocket_connect(f"/v1/ws/jobs/{job_id}/logs") as ws:
            for _ in range(20):
                msg = ws.receive()
                if msg["type"] == "websocket.close":
                    break
                if msg["type"] == "websocket.send":
                    text = msg.get("text", "")
                    received_texts.append(text)
                    try:
                        parsed = json.loads(text)
                        if parsed.get("type") == "stream_end":
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass

        stream_end_msgs = [
            json.loads(t) for t in received_texts
            if _is_json(t) and json.loads(t).get("type") == "stream_end"
        ]
        assert len(stream_end_msgs) >= 1
        assert stream_end_msgs[0]["status"] == "failed"


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, TypeError):
        return False
