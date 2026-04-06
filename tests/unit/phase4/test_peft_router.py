"""HTTP integration tests for /v1/peft endpoints."""

from __future__ import annotations

import pytest

from app.services.checkpoint_service import register_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config() -> dict:  # type: ignore[return]
    return {
        "data": {"dataset_path": "datasets/train"},
        "precision": {},
        "optimizer": {},
        "model": {},
        "training": {},
    }


def _lora_payload(checkpoint_name: str, include_lora: bool = True) -> dict:  # type: ignore[return]
    config = _base_config()
    if include_lora:
        config["lora"] = {"rank": 8, "alpha": 16.0}
    return {
        "config": config,
        "num_gpus": 1,
        "checkpoint_name": checkpoint_name,
    }


def _dora_payload(checkpoint_name: str, include_dora: bool = True) -> dict:  # type: ignore[return]
    config = _base_config()
    if include_dora:
        config["dora"] = {"rank": 4, "alpha": 8.0}
    return {
        "config": config,
        "num_gpus": 1,
        "checkpoint_name": checkpoint_name,
    }


@pytest.fixture
async def db(test_app):
    import aiosqlite
    from app import config as _cfg

    async with aiosqlite.connect(_cfg.settings.sqlite_path) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


@pytest.fixture
async def seeded_checkpoint(db, test_app):
    from app import config as _cfg

    name = "base-model-ckpt"
    path = str(_cfg.settings.checkpoints_root / name)
    await register_checkpoint(db, name=name, fmt="megatron", path=path)
    return name


# ---------------------------------------------------------------------------
# POST /v1/peft/lora
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lora_returns_202(client, seeded_checkpoint):
    resp = await client.post(
        "/v1/peft/lora", json=_lora_payload(seeded_checkpoint)
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_lora_without_lora_config_returns_422(client, seeded_checkpoint):
    resp = await client.post(
        "/v1/peft/lora", json=_lora_payload(seeded_checkpoint, include_lora=False)
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "lora" in data.get("detail", "").lower()


@pytest.mark.asyncio
async def test_lora_missing_checkpoint_returns_404(client):
    payload = _lora_payload("nonexistent-base-model")
    resp = await client.post("/v1/peft/lora", json=payload)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_lora_path_traversal_returns_400(client, seeded_checkpoint):
    payload = _lora_payload(seeded_checkpoint)
    payload["config"]["data"]["dataset_path"] = "../../../etc/passwd"
    resp = await client.post("/v1/peft/lora", json=payload)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /v1/peft/dora
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dora_returns_202(client, seeded_checkpoint):
    resp = await client.post(
        "/v1/peft/dora", json=_dora_payload(seeded_checkpoint)
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_dora_without_dora_config_returns_422(client, seeded_checkpoint):
    resp = await client.post(
        "/v1/peft/dora", json=_dora_payload(seeded_checkpoint, include_dora=False)
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "dora" in data.get("detail", "").lower()


@pytest.mark.asyncio
async def test_dora_missing_checkpoint_returns_404(client):
    payload = _dora_payload("nonexistent-base-model")
    resp = await client.post("/v1/peft/dora", json=payload)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dora_path_traversal_returns_400(client, seeded_checkpoint):
    payload = _dora_payload(seeded_checkpoint)
    payload["config"]["data"]["dataset_path"] = "../../sensitive"
    resp = await client.post("/v1/peft/dora", json=payload)
    assert resp.status_code == 400
