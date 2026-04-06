"""Phase 3 test fixtures.

Patches register_checkpoint_sync so executor unit tests don't need a real DB.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_register_checkpoint(tmp_path):
    """Stub out register_checkpoint_sync for all executor unit tests."""
    with patch("app.worker.executor.register_checkpoint_sync") as m:
        m.return_value = None
        yield m
