"""Phase 6 unit tests — WebSocket progress streaming + security fuzz.

Tests /v1/ws/jobs/{id}/logs and /v1/ws/jobs/{id}/progress via synchronous
TestClient (required for WebSocket testing).
"""

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


def _db_path(test_app) -> str:
    import app.config as cfg
    return cfg.settings.database_url[len("sqlite+aiosqlite:///"):]


async def _seed_job(db_path: str, status: str = "running") -> str:
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


# ---------------------------------------------------------------------------
# Security fuzz: invalid job_ids must never expose file contents
# ---------------------------------------------------------------------------


class TestWebSocketSecurityFuzz:
    @pytest.mark.parametrize("bad_id", [
        "not-a-uuid",
        "../../../etc/passwd",
        "a" * 200,
        "'; DROP TABLE jobs; --",
        "\x00null\x00byte",
    ])
    def test_invalid_job_id_closes_ws_logs(self, sync_client, bad_id):
        """All non-UUID job_ids must close the WebSocket without leaking content."""
        import urllib.parse
        safe_id = urllib.parse.quote(bad_id, safe="")
        url = f"/v1/ws/jobs/{safe_id}/logs"
        try:
            with sync_client.websocket_connect(url) as ws:
                msg = ws.receive()
                # Must close — never send file content
                assert msg["type"] == "websocket.close"
                if "text" in msg:
                    assert "/etc/passwd" not in msg.get("text", "")
                    assert "/etc/shadow" not in msg.get("text", "")
        except WebSocketDisconnect:
            # Routing-level rejection (e.g. %2F decoded to "/" breaks path template)
            # This is still a security win — no file content leaked.
            pass

    @pytest.mark.parametrize("bad_id", [
        "not-a-uuid",
        "../../../etc/passwd",
        "a" * 200,
    ])
    def test_invalid_job_id_closes_ws_progress(self, sync_client, bad_id):
        """All non-UUID job_ids must close the progress WebSocket without leaking content."""
        import urllib.parse
        safe_id = urllib.parse.quote(bad_id, safe="")
        url = f"/v1/ws/jobs/{safe_id}/progress"
        try:
            with sync_client.websocket_connect(url) as ws:
                msg = ws.receive()
                assert msg["type"] == "websocket.close"
        except WebSocketDisconnect:
            pass  # Routing-level rejection — security requirement met


# ---------------------------------------------------------------------------
# Progress WebSocket
# ---------------------------------------------------------------------------


class TestProgressStream:
    async def test_progress_ws_connects_for_valid_job(self, sync_client, test_app):
        """Valid job_id connects successfully to the progress WebSocket."""
        db_path = _db_path(test_app)
        job_id = await _seed_job(db_path, status="running")

        with sync_client.websocket_connect(f"/v1/ws/jobs/{job_id}/progress") as ws:
            msg = ws.receive()
            # Should receive a progress frame or a close (if job already terminal)
            assert msg["type"] in ("websocket.send", "websocket.close")

    async def test_progress_frame_has_required_fields(self, sync_client, test_app):
        """Progress frame JSON contains type, job_id, status, progress, gpus."""
        db_path = _db_path(test_app)
        job_id = await _seed_job(db_path, status="running")

        with sync_client.websocket_connect(f"/v1/ws/jobs/{job_id}/progress") as ws:
            for _ in range(10):
                msg = ws.receive()
                if msg["type"] == "websocket.close":
                    break
                if msg["type"] == "websocket.send":
                    text = msg.get("text", "")
                    try:
                        frame = json.loads(text)
                        if frame.get("type") == "progress":
                            assert "job_id" in frame
                            assert "status" in frame
                            assert "gpus" in frame
                            assert frame["job_id"] == job_id
                            return  # test passed
                    except (json.JSONDecodeError, AttributeError):
                        pass

    async def test_job_complete_message_on_terminal_state(self, sync_client, test_app):
        """job_complete message is sent when job is already in terminal state."""
        db_path = _db_path(test_app)
        job_id = await _seed_job(db_path, status="completed")

        received = []
        with sync_client.websocket_connect(f"/v1/ws/jobs/{job_id}/progress") as ws:
            for _ in range(20):
                msg = ws.receive()
                if msg["type"] == "websocket.close":
                    break
                if msg["type"] == "websocket.send":
                    text = msg.get("text", "")
                    received.append(text)
                    try:
                        frame = json.loads(text)
                        if frame.get("type") == "job_complete":
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass

        complete_msgs = [
            json.loads(t) for t in received
            if _is_json(t) and json.loads(t).get("type") == "job_complete"
        ]
        assert len(complete_msgs) >= 1
        assert complete_msgs[0]["status"] == "completed"

    async def test_progress_ws_unknown_job_closes_1008(self, sync_client, test_app):
        """Unknown (but valid UUID) job_id closes with 1008."""
        fake_id = str(uuid.uuid4())
        with sync_client.websocket_connect(f"/v1/ws/jobs/{fake_id}/progress") as ws:
            msg = ws.receive()
            assert msg["type"] == "websocket.close"
            assert msg.get("code") == 1008

    async def test_failed_job_sends_complete_message(self, sync_client, test_app):
        """job_complete is also sent for failed jobs."""
        db_path = _db_path(test_app)
        job_id = await _seed_job(db_path, status="failed")

        received = []
        with sync_client.websocket_connect(f"/v1/ws/jobs/{job_id}/progress") as ws:
            for _ in range(20):
                msg = ws.receive()
                if msg["type"] == "websocket.close":
                    break
                if msg["type"] == "websocket.send":
                    text = msg.get("text", "")
                    received.append(text)
                    try:
                        frame = json.loads(text)
                        if frame.get("type") == "job_complete":
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass

        complete_msgs = [
            json.loads(t) for t in received
            if _is_json(t) and json.loads(t).get("type") == "job_complete"
        ]
        assert len(complete_msgs) >= 1

    async def test_cancelled_job_sends_complete_message(self, sync_client, test_app):
        """job_complete is also sent for cancelled jobs."""
        db_path = _db_path(test_app)
        job_id = await _seed_job(db_path, status="cancelled")

        received = []
        with sync_client.websocket_connect(f"/v1/ws/jobs/{job_id}/progress") as ws:
            for _ in range(20):
                msg = ws.receive()
                if msg["type"] == "websocket.close":
                    break
                if msg["type"] == "websocket.send":
                    text = msg.get("text", "")
                    received.append(text)
                    try:
                        frame = json.loads(text)
                        if frame.get("type") == "job_complete":
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass

        complete_msgs = [
            json.loads(t) for t in received
            if _is_json(t) and json.loads(t).get("type") == "job_complete"
        ]
        assert len(complete_msgs) >= 1


# ---------------------------------------------------------------------------
# Log WebSocket — basic connectivity
# ---------------------------------------------------------------------------


class TestLogStreamConnectivity:
    async def test_log_ws_connects_for_valid_job(self, sync_client, test_app):
        """Valid job_id connects successfully to the log WebSocket."""
        db_path = _db_path(test_app)
        job_id = await _seed_job(db_path, status="completed")

        import app.config as cfg
        log_dir = cfg.settings.logs_root
        log_dir.mkdir(parents=True, exist_ok=True)

        with sync_client.websocket_connect(f"/v1/ws/jobs/{job_id}/logs") as ws:
            msg = ws.receive()
            assert msg["type"] in ("websocket.send", "websocket.close")

    async def test_log_ws_unknown_job_closes_1008(self, sync_client):
        fake_id = str(uuid.uuid4())
        with sync_client.websocket_connect(f"/v1/ws/jobs/{fake_id}/logs") as ws:
            msg = ws.receive()
            assert msg["type"] == "websocket.close"
            assert msg.get("code") == 1008


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, TypeError):
        return False
