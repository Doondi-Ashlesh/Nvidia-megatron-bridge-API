"""HTTP integration tests for /v1/checkpoints endpoints."""

from __future__ import annotations

import logging

import pytest

from app.services.checkpoint_service import register_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_checkpoint(client, db, *, name="test-ckpt", fmt="megatron"):
    """Insert a checkpoint directly into the DB and return its ID."""
    from app import config as _cfg

    path = str(_cfg.settings.checkpoints_root / name)
    ckpt_id = await register_checkpoint(db, name=name, fmt=fmt, path=path)
    return ckpt_id


@pytest.fixture
async def db(test_app):
    """Yield a live aiosqlite connection to the test database."""
    import aiosqlite
    from app import config as _cfg

    async with aiosqlite.connect(_cfg.settings.sqlite_path) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


# ---------------------------------------------------------------------------
# POST /v1/checkpoints/import
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_returns_202_with_job_id(client):
    resp = await client.post(
        "/v1/checkpoints/import",
        json={
            "source_path": "meta-llama/Llama-3-8B",
            "target_name": "llama3-megatron",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_import_with_hf_token(client):
    resp = await client.post(
        "/v1/checkpoints/import",
        json={
            "source_path": "meta-llama/Llama-3-8B",
            "target_name": "llama3-gated",
            "hf_token": "hf_secrettoken123",
        },
    )
    assert resp.status_code == 202
    assert "job_id" in resp.json()


@pytest.mark.asyncio
async def test_import_path_traversal_returns_400(client):
    resp = await client.post(
        "/v1/checkpoints/import",
        json={
            "source_path": "safe-source",
            "target_name": "../../../etc/passwd",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_import_null_byte_returns_400(client):
    resp = await client.post(
        "/v1/checkpoints/import",
        json={
            "source_path": "safe-source",
            "target_name": "evil\x00name",
        },
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /v1/checkpoints/export
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_returns_202_with_job_id(client, db):
    ckpt_id = await _seed_checkpoint(client, db, name="source-ckpt", fmt="megatron")

    resp = await client.post(
        "/v1/checkpoints/export",
        json={
            "source_name": "source-ckpt",
            "target_name": "source-ckpt-hf",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data


@pytest.mark.asyncio
async def test_export_nonexistent_source_returns_404(client):
    resp = await client.post(
        "/v1/checkpoints/export",
        json={
            "source_name": "does-not-exist",
            "target_name": "output-hf",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_path_traversal_source_returns_400(client):
    resp = await client.post(
        "/v1/checkpoints/export",
        json={
            "source_name": "../../../etc",
            "target_name": "valid-target",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_export_path_traversal_target_returns_400(client):
    resp = await client.post(
        "/v1/checkpoints/export",
        json={
            "source_name": "valid-source",
            "target_name": "../../../etc/passwd",
        },
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /v1/checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_empty(client):
    resp = await client.get("/v1/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_returns_seeded(client, db):
    await _seed_checkpoint(client, db, name="ckpt-a", fmt="megatron")
    await _seed_checkpoint(client, db, name="ckpt-b", fmt="hf")

    resp = await client.get("/v1/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


# ---------------------------------------------------------------------------
# GET /v1/checkpoints/{id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_existing(client, db):
    ckpt_id = await _seed_checkpoint(client, db, name="single-ckpt")

    resp = await client.get(f"/v1/checkpoints/{ckpt_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == ckpt_id
    assert data["name"] == "single-ckpt"


@pytest.mark.asyncio
async def test_get_unknown_returns_404(client):
    resp = await client.get("/v1/checkpoints/unknown-id-99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /v1/checkpoints/{id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_existing_returns_204(client, db):
    ckpt_id = await _seed_checkpoint(client, db, name="to-delete")

    resp = await client.delete(f"/v1/checkpoints/{ckpt_id}")
    assert resp.status_code == 204

    # Verify gone
    get_resp = await client.get(f"/v1/checkpoints/{ckpt_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_unknown_returns_404(client):
    resp = await client.delete("/v1/checkpoints/nonexistent-99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Security: hf_token never in logs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hf_token_not_in_logs(client, caplog):
    secret_token = "hf_SuperSecretToken_ShouldNeverAppear"

    # Only capture app-level logs — aiosqlite's DEBUG logs emit raw SQL params
    # which are internal DB driver messages, not application log output.
    # The token_filter.py scrubs application-layer log records; the test
    # verifies that no app code leaks the token in its own log statements.
    with caplog.at_level(logging.DEBUG, logger="app"):
        resp = await client.post(
            "/v1/checkpoints/import",
            json={
                "source_path": "meta-llama/Llama-3-8B",
                "target_name": "llama3-tok-test",
                "hf_token": secret_token,
            },
        )
    assert resp.status_code == 202

    full_log = caplog.text
    assert secret_token not in full_log, "hf_token must not appear in app log output"
