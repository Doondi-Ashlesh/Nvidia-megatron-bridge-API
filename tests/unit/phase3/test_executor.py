"""Unit tests for app.worker.executor (no real megatron.bridge needed).

The conftest.py autouse fixture injects a fake megatron.bridge into sys.modules
before any import, so executor.py is safe to import here without CUDA.

Real API used (megatron-bridge v0.3.1, verified March 2026):
  - AutoBridge.import_ckpt(hf_model_id, megatron_path)           — HF → Megatron
  - AutoBridge.from_hf_pretrained(arch).export_ckpt(src, dst)    — Megatron → HF
  - pretrain(ConfigContainer, forward_step)                       — pretraining
  - finetune(ConfigContainer, forward_step)                       — SFT / LoRA / DoRA
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

from app.worker.executor import (
    fail_job,
    handle_checkpoint_export,
    handle_checkpoint_import,
    main,
    mark_completed,
    mark_running,
)


def _get_autobridge_mock() -> MagicMock:
    """Return the AutoBridge mock from the injected megatron.bridge module."""
    mb_mock = sys.modules["megatron.bridge"]
    return mb_mock.AutoBridge


# ---------------------------------------------------------------------------
# handle_checkpoint_import — real API: AutoBridge.import_ckpt
# ---------------------------------------------------------------------------


def test_import_calls_autobridge_import_ckpt():
    """handle_checkpoint_import calls AutoBridge.import_ckpt with correct args."""
    ab_mock = _get_autobridge_mock()
    ab_mock.reset_mock()

    payload = {
        "source_path": "meta-llama/Llama-3-8B",
        "target_path": "/data/checkpoints/llama3",
    }
    job = {"payload": json.dumps(payload)}

    handle_checkpoint_import(job)

    ab_mock.import_ckpt.assert_called_once_with(
        hf_model_id="meta-llama/Llama-3-8B",
        megatron_path="/data/checkpoints/llama3",
    )


def test_import_sets_hf_token_env_var():
    """hf_token is set as HF_TOKEN env var and NOT passed to the SDK call."""
    ab_mock = _get_autobridge_mock()
    ab_mock.reset_mock()

    secret = "hf_mysecrettoken"
    payload = {
        "source_path": "meta-llama/Llama-3-8B",
        "target_path": "/data/checkpoints/llama3",
        "hf_token": secret,
    }
    job = {"payload": json.dumps(payload)}

    os.environ.pop("HF_TOKEN", None)
    handle_checkpoint_import(job)

    # Token must be set in the environment
    assert os.environ.get("HF_TOKEN") == secret

    # Token must NOT appear in any SDK call argument
    call_kwargs = ab_mock.import_ckpt.call_args.kwargs
    assert "hf_token" not in call_kwargs


def test_import_hf_token_popped_from_payload():
    """hf_token is removed from payload before any SDK call."""
    ab_mock = _get_autobridge_mock()
    ab_mock.reset_mock()

    payload = {
        "source_path": "model/repo",
        "target_path": "/data/checkpoints/out",
        "hf_token": "hf_tok",
    }
    job = {"payload": json.dumps(payload)}

    handle_checkpoint_import(job)

    call_kwargs = ab_mock.import_ckpt.call_args.kwargs
    assert "hf_token" not in call_kwargs


# ---------------------------------------------------------------------------
# handle_checkpoint_export — real API: AutoBridge.from_hf_pretrained + export_ckpt
# ---------------------------------------------------------------------------


def test_export_calls_from_hf_pretrained():
    """handle_checkpoint_export calls AutoBridge.from_hf_pretrained with model_arch."""
    ab_mock = _get_autobridge_mock()
    ab_mock.reset_mock()

    payload = {
        "source_path": "/data/checkpoints/megatron-model",
        "target_path": "/data/checkpoints/hf-out",
        "model_arch": "meta-llama/Llama-3-8B",
    }
    job = {"payload": json.dumps(payload)}

    handle_checkpoint_export(job)

    ab_mock.from_hf_pretrained.assert_called_once_with("meta-llama/Llama-3-8B")


def test_export_calls_export_ckpt_with_correct_paths():
    """handle_checkpoint_export calls bridge.export_ckpt with source and target paths."""
    ab_mock = _get_autobridge_mock()
    ab_mock.reset_mock()

    # The bridge instance returned by from_hf_pretrained
    bridge_instance = ab_mock.from_hf_pretrained.return_value

    payload = {
        "source_path": "/data/checkpoints/megatron-model",
        "target_path": "/data/checkpoints/hf-out",
        "model_arch": "meta-llama/Llama-3-8B",
    }
    job = {"payload": json.dumps(payload)}

    handle_checkpoint_export(job)

    bridge_instance.export_ckpt.assert_called_once_with(
        megatron_path="/data/checkpoints/megatron-model",
        hf_path="/data/checkpoints/hf-out",
        show_progress=True,
        strict=False,
    )


def test_export_empty_model_arch_still_calls_from_hf_pretrained():
    """When model_arch is absent, export still calls from_hf_pretrained with empty string."""
    ab_mock = _get_autobridge_mock()
    ab_mock.reset_mock()

    payload = {
        "source_path": "/data/checkpoints/src",
        "target_path": "/data/checkpoints/dst",
    }
    job = {"payload": json.dumps(payload)}

    handle_checkpoint_export(job)

    ab_mock.from_hf_pretrained.assert_called_once_with("")


# ---------------------------------------------------------------------------
# main() — unknown job type
# ---------------------------------------------------------------------------


def test_unknown_job_type_calls_fail_job():
    """An unrecognised job type must call fail_job and return early."""
    job = {
        "id": "test-job-001",
        "type": "unknown_op",
        "payload": json.dumps({}),
    }

    with (
        patch("app.worker.executor.load_job", return_value=job),
        patch("app.worker.executor.mark_running") as mock_running,
        patch("app.worker.executor.fail_job") as mock_fail,
        patch("sys.argv", ["executor.py", "--job-id", "test-job-001"]),
    ):
        main()

    mock_fail.assert_called_once()
    args = mock_fail.call_args[0]
    assert "Unknown job type" in args[1]
    mock_running.assert_not_called()


# ---------------------------------------------------------------------------
# main() — exception in handler calls fail_job
# ---------------------------------------------------------------------------


def test_exception_in_handler_calls_fail_job():
    """If the handler raises, fail_job should be called with the error message."""
    ab_mock = _get_autobridge_mock()
    ab_mock.reset_mock()
    ab_mock.import_ckpt.side_effect = RuntimeError("GPU OOM")

    job = {
        "id": "test-job-002",
        "type": "checkpoint_import",
        "payload": json.dumps(
            {"source_path": "model/repo", "target_path": "/data/checkpoints/out"}
        ),
    }

    with (
        patch("app.worker.executor.load_job", return_value=job),
        patch("app.worker.executor.mark_running"),
        patch("app.worker.executor.mark_completed") as mock_completed,
        patch("app.worker.executor.fail_job") as mock_fail,
        patch("sys.argv", ["executor.py", "--job-id", "test-job-002"]),
    ):
        main()

    mock_fail.assert_called_once()
    args = mock_fail.call_args[0]
    assert "GPU OOM" in args[1]
    mock_completed.assert_not_called()

    ab_mock.import_ckpt.side_effect = None


# ---------------------------------------------------------------------------
# main() — happy path
# ---------------------------------------------------------------------------


def test_main_happy_path_import():
    """On success, mark_running and mark_completed are both called."""
    ab_mock = _get_autobridge_mock()
    ab_mock.reset_mock()

    job = {
        "id": "test-job-003",
        "type": "checkpoint_import",
        "payload": json.dumps(
            {"source_path": "model/repo", "target_path": "/data/checkpoints/ok"}
        ),
    }

    with (
        patch("app.worker.executor.load_job", return_value=job),
        patch("app.worker.executor.mark_running") as mock_running,
        patch("app.worker.executor.mark_completed") as mock_completed,
        patch("app.worker.executor.fail_job") as mock_fail,
        patch("sys.argv", ["executor.py", "--job-id", "test-job-003"]),
    ):
        main()

    mock_running.assert_called_once_with("test-job-003")
    mock_completed.assert_called_once_with("test-job-003")
    mock_fail.assert_not_called()
