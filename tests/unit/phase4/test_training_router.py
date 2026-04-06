"""HTTP integration tests for /v1/training endpoints."""

from __future__ import annotations

import pytest

from app.services.checkpoint_service import register_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pretrain_payload(**overrides):  # type: ignore[return]
    base = {
        "config": {
            "data": {"dataset_path": "datasets/train"},
            "precision": {},
            "optimizer": {},
            "model": {},
            "training": {},
        },
        "num_gpus": 1,
    }
    base.update(overrides)
    return base


def _make_finetune_payload(checkpoint_name: str, **overrides):  # type: ignore[return]
    base = {
        "config": {
            "data": {"dataset_path": "datasets/train"},
            "precision": {},
            "optimizer": {},
            "model": {},
            "training": {},
        },
        "num_gpus": 1,
        "checkpoint_name": checkpoint_name,
    }
    base.update(overrides)
    return base


@pytest.fixture
async def db(test_app):
    import aiosqlite
    from app import config as _cfg

    async with aiosqlite.connect(_cfg.settings.sqlite_path) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


@pytest.fixture
async def seeded_checkpoint(db, test_app):
    """Create a checkpoint in the DB and return its name."""
    from app import config as _cfg

    name = "base-ckpt"
    path = str(_cfg.settings.checkpoints_root / name)
    await register_checkpoint(db, name=name, fmt="megatron", path=path)
    return name


# ---------------------------------------------------------------------------
# POST /v1/training/pretrain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pretrain_returns_202(client):
    resp = await client.post("/v1/training/pretrain", json=_make_pretrain_payload())
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_pretrain_without_checkpoint_name_succeeds(client):
    payload = _make_pretrain_payload()
    resp = await client.post("/v1/training/pretrain", json=payload)
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_pretrain_with_valid_checkpoint_succeeds(client, seeded_checkpoint):
    payload = _make_pretrain_payload(checkpoint_name=seeded_checkpoint)
    resp = await client.post("/v1/training/pretrain", json=payload)
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_pretrain_with_missing_checkpoint_returns_404(client):
    payload = _make_pretrain_payload(checkpoint_name="nonexistent-ckpt")
    resp = await client.post("/v1/training/pretrain", json=payload)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_pretrain_invalid_config_unknown_field_returns_422(client):
    payload = _make_pretrain_payload()
    payload["config"]["unknown_field"] = "bad"
    resp = await client.post("/v1/training/pretrain", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_pretrain_path_traversal_dataset_returns_400(client):
    payload = _make_pretrain_payload()
    payload["config"]["data"]["dataset_path"] = "../../../etc/passwd"
    resp = await client.post("/v1/training/pretrain", json=payload)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /v1/training/finetune
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finetune_returns_202(client, seeded_checkpoint):
    payload = _make_finetune_payload(seeded_checkpoint)
    resp = await client.post("/v1/training/finetune", json=payload)
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data


@pytest.mark.asyncio
async def test_finetune_missing_checkpoint_returns_404(client):
    payload = _make_finetune_payload("does-not-exist")
    resp = await client.post("/v1/training/finetune", json=payload)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_finetune_path_traversal_returns_400(client, seeded_checkpoint):
    payload = _make_finetune_payload(seeded_checkpoint)
    payload["config"]["data"]["dataset_path"] = "../../etc"
    resp = await client.post("/v1/training/finetune", json=payload)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /v1/training
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_training_returns_200_empty(client):
    resp = await client.get("/v1/training")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_list_training_returns_submitted_jobs(client):
    # Submit one pretrain job first
    await client.post("/v1/training/pretrain", json=_make_pretrain_payload())

    resp = await client.get("/v1/training")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
