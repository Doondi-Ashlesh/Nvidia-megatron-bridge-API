"""Root test fixtures shared across all phases.

Fixtures defined here are available to every test file without explicit import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Environment override — must happen before app modules are imported ────────
# Use an in-memory SQLite DB and a temp DATA_DIR for all tests.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///test_megatronbridge.db")
os.environ.setdefault("DATA_DIR", "/tmp/megatronbridge_test")
os.environ.setdefault("CHECKPOINTS_ROOT", "/tmp/megatronbridge_test/checkpoints")
os.environ.setdefault("LOGS_ROOT", "/tmp/megatronbridge_test/logs")
os.environ.setdefault("HF_CACHE_DIR", "/tmp/megatronbridge_test/hf_cache")
os.environ.setdefault("CORS_ORIGINS", "")
os.environ.setdefault("MAX_CONCURRENT_JOBS", "1")


# ── megatron.bridge mock — injected before any import ─────────────────────────
@pytest.fixture(autouse=True)
def mock_megatron_bridge():
    """Inject a fake ``megatron.bridge`` into sys.modules.

    This prevents ImportError on machines without CUDA and ensures that
    executor.py is the only real contact point with the SDK in production.
    The fixture is ``autouse=True`` so ALL tests benefit automatically;
    GPU-tier tests can override it by removing it from their module scope.
    """
    fake_bridge = MagicMock()
    fake_megatron = MagicMock()
    fake_megatron.bridge = fake_bridge

    # Patch the full module hierarchy so executor.py's late imports all resolve.
    # Real import paths verified against megatron-bridge v0.3.1:
    #   from megatron.bridge import AutoBridge
    #   from megatron.bridge.peft.lora import LoRA
    #   from megatron.bridge.peft.dora import DoRA
    #   from megatron.bridge.training.pretrain import pretrain
    #   from megatron.bridge.training.finetune import finetune
    #   from megatron.bridge.training.gpt_step import forward_step
    #   from megatron.bridge.training.config import ConfigContainer, ...
    sys.modules.setdefault("megatron", fake_megatron)
    sys.modules.setdefault("megatron.bridge", fake_bridge)
    sys.modules.setdefault("megatron.bridge.peft", MagicMock())
    sys.modules.setdefault("megatron.bridge.peft.lora", MagicMock())
    sys.modules.setdefault("megatron.bridge.peft.dora", MagicMock())
    sys.modules.setdefault("megatron.bridge.training", MagicMock())
    sys.modules.setdefault("megatron.bridge.training.pretrain", MagicMock())
    sys.modules.setdefault("megatron.bridge.training.finetune", MagicMock())
    sys.modules.setdefault("megatron.bridge.training.gpt_step", MagicMock())
    sys.modules.setdefault("megatron.bridge.training.config", MagicMock())
    sys.modules.setdefault("megatron.core", MagicMock())
    sys.modules.setdefault("torch", MagicMock())
    yield fake_bridge
    # Do NOT remove from sys.modules after the test — other tests in the
    # same session may have already imported sub-modules.


# ── pynvml mock ───────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_pynvml():
    """Replace pynvml with a fake so GPU telemetry tests run without hardware."""
    fake = MagicMock()
    fake.nvmlInit.return_value = None
    fake.nvmlShutdown.return_value = None
    fake.nvmlDeviceGetCount.return_value = 2
    fake.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
    fake.nvmlDeviceGetName.return_value = b"NVIDIA H100 SXM5 80GB"
    fake.nvmlDeviceGetTemperature.return_value = 72
    fake.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=85, memory=60)
    fake.nvmlDeviceGetMemoryInfo.return_value = MagicMock(
        total=80 * 1024**3,
        used=48 * 1024**3,
        free=32 * 1024**3,
    )
    fake.nvmlSystemGetDriverVersion.return_value = b"550.90.07"
    fake.nvmlSystemGetCudaDriverVersion.return_value = 12040
    fake.NVMLError = Exception

    sys.modules["pynvml"] = fake
    yield fake


# ── FastAPI test app ──────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def test_app(tmp_path: Path):
    """Create a fresh FastAPI app backed by a real (file-based) SQLite DB in tmp_path.

    The dispatcher background task is patched to a no-op coroutine so tests
    never accidentally invoke real torchrun processes.
    """
    import asyncio
    import importlib
    from unittest.mock import patch

    db_path = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["CHECKPOINTS_ROOT"] = str(tmp_path / "checkpoints")
    os.environ["LOGS_ROOT"] = str(tmp_path / "logs")

    # Re-import settings so it picks up the new env vars
    import app.config as cfg_module
    importlib.reload(cfg_module)

    # No-op dispatcher: immediately suspends so CancelledError wakes it
    async def _noop_dispatcher(**_kw):
        await asyncio.sleep(3600)  # cancelled by lifespan shutdown

    from app.main import create_app
    application = create_app()

    # Run lifespan startup/shutdown with dispatcher replaced by noop
    with patch("app.worker.dispatcher.dispatcher_loop", side_effect=_noop_dispatcher):
        async with application.router.lifespan_context(application):
            yield application


@pytest_asyncio.fixture
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client wired to the test app."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as ac:
        yield ac


# ── Launcher mock (Phase 2+) ──────────────────────────────────────────────────
@pytest.fixture
def mock_launcher(monkeypatch):
    """Replace the real torchrun launcher with a controllable async mock."""
    mock = AsyncMock(return_value={"exit_code": 0, "pid": 99999})
    # Will be monkeypatched once app.worker.launcher exists
    return mock
