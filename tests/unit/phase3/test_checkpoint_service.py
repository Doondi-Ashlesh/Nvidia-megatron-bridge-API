"""Unit tests for app.services.checkpoint_service."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from app.services.checkpoint_service import (
    delete_checkpoint,
    get_checkpoint,
    get_checkpoint_size,
    list_checkpoints,
    register_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(test_app):
    """Yield a live aiosqlite connection to the test database."""
    import aiosqlite
    from app import config as _cfg

    async with aiosqlite.connect(_cfg.settings.sqlite_path) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


# ---------------------------------------------------------------------------
# register + get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_and_get(db):
    ckpt_id = await register_checkpoint(
        db,
        name="my-model",
        fmt="megatron",
        path="/data/checkpoints/my-model",
    )
    assert ckpt_id is not None

    record = await get_checkpoint(db, ckpt_id)
    assert record is not None
    assert record["name"] == "my-model"
    assert record["format"] == "megatron"
    assert record["path"] == "/data/checkpoints/my-model"
    assert record["size_bytes"] is None
    assert record["model_arch"] is None
    assert record["created_from_job"] is None


@pytest.mark.asyncio
async def test_register_with_all_fields(db):
    ckpt_id = await register_checkpoint(
        db,
        name="llama3-hf",
        fmt="hf",
        path="/data/checkpoints/llama3-hf",
        size_bytes=1024,
        model_arch="llama3",
        created_from_job="job-abc-123",
    )
    record = await get_checkpoint(db, ckpt_id)
    assert record["size_bytes"] == 1024
    assert record["model_arch"] == "llama3"
    assert record["created_from_job"] == "job-abc-123"


@pytest.mark.asyncio
async def test_get_missing_returns_none(db):
    result = await get_checkpoint(db, "nonexistent-id")
    assert result is None


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_empty(db):
    items, total = await list_checkpoints(db)
    assert items == []
    assert total == 0


@pytest.mark.asyncio
async def test_list_returns_registered(db):
    await register_checkpoint(db, name="a", fmt="megatron", path="/data/checkpoints/a")
    await register_checkpoint(db, name="b", fmt="hf", path="/data/checkpoints/b")

    items, total = await list_checkpoints(db)
    assert total == 2
    assert len(items) == 2


@pytest.mark.asyncio
async def test_list_format_filter(db):
    await register_checkpoint(db, name="m1", fmt="megatron", path="/data/checkpoints/m1")
    await register_checkpoint(db, name="h1", fmt="hf", path="/data/checkpoints/h1")
    await register_checkpoint(db, name="m2", fmt="megatron", path="/data/checkpoints/m2")

    items, total = await list_checkpoints(db, fmt="megatron")
    assert total == 2
    for item in items:
        assert item["format"] == "megatron"


@pytest.mark.asyncio
async def test_list_limit_offset(db):
    for i in range(5):
        await register_checkpoint(
            db, name=f"ckpt-{i}", fmt="megatron", path=f"/data/checkpoints/ckpt-{i}"
        )

    items, total = await list_checkpoints(db, limit=2, offset=0)
    assert total == 5
    assert len(items) == 2

    items2, _ = await list_checkpoints(db, limit=2, offset=2)
    assert len(items2) == 2

    # Pages should not overlap
    ids_page1 = {r["id"] for r in items}
    ids_page2 = {r["id"] for r in items2}
    assert ids_page1.isdisjoint(ids_page2)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_existing(db):
    ckpt_id = await register_checkpoint(
        db, name="to-delete", fmt="hf", path="/data/checkpoints/to-delete"
    )
    result = await delete_checkpoint(db, ckpt_id)
    assert result is True

    record = await get_checkpoint(db, ckpt_id)
    assert record is None


@pytest.mark.asyncio
async def test_delete_missing_returns_false(db):
    result = await delete_checkpoint(db, "does-not-exist")
    assert result is False


# ---------------------------------------------------------------------------
# get_checkpoint_size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_checkpoint_size_returns_none_on_oserror(tmp_path):
    missing = tmp_path / "nonexistent_dir"
    result = await get_checkpoint_size(missing)
    assert result is None


@pytest.mark.asyncio
async def test_get_checkpoint_size_sums_files(tmp_path):
    (tmp_path / "file1.bin").write_bytes(b"A" * 100)
    (tmp_path / "file2.bin").write_bytes(b"B" * 200)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "file3.bin").write_bytes(b"C" * 50)

    size = await get_checkpoint_size(tmp_path)
    assert size == 350


@pytest.mark.asyncio
async def test_get_checkpoint_size_single_file(tmp_path):
    f = tmp_path / "weights.bin"
    f.write_bytes(b"X" * 512)
    size = await get_checkpoint_size(f)
    assert size == 512
