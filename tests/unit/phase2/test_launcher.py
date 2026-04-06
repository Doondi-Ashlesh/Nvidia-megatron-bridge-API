"""Phase 2 unit tests — torchrun launcher (launcher.py).

Security focus:
- shell=False enforced on every Popen call (AST-level assertion)
- Torchrun flag whitelist: any unknown flag raises LauncherValidationError
- CUDA_VISIBLE_DEVICES: only digits and commas allowed
- job_id must be UUID4

No actual torchrun binary is invoked — subprocess.Popen is monkeypatched.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.worker.launcher import (
    LauncherValidationError,
    _ALLOWED_TORCHRUN_FLAGS,
    _validate_cuda_devices,
    _validate_job_id,
    _validate_num_gpus,
    _validate_torchrun_flags,
    find_free_port,
    is_valid_uuid,
    launch_worker,
    terminate_worker,
)


# ---------------------------------------------------------------------------
# AST security test: shell=False enforced in source
# ---------------------------------------------------------------------------


class TestShellFalseEnforced:
    def test_no_shell_true_in_launcher_source(self):
        """Statically verify that shell=True never appears in launcher.py."""
        import app.worker.launcher as launcher_mod

        source = inspect.getsource(launcher_mod)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shell":
                        # shell= keyword must be False (or the name "False")
                        val = kw.value
                        if isinstance(val, ast.Constant):
                            assert val.value is False, "shell=True found in launcher!"
                        elif isinstance(val, ast.Name):
                            assert val.id == "False", f"shell={val.id} found in launcher!"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class TestValidateJobId:
    def test_valid_uuid4(self):
        _validate_job_id("550e8400-e29b-41d4-a716-446655440000")

    def test_invalid_string_raises(self):
        with pytest.raises(LauncherValidationError):
            _validate_job_id("not-a-uuid")

    def test_uuid3_raises(self):
        # UUID3: version nibble = 3, not 4
        with pytest.raises(LauncherValidationError):
            _validate_job_id("6ba7b810-9dad-31d1-80b4-00c04fd430c8")

    def test_empty_string_raises(self):
        with pytest.raises(LauncherValidationError):
            _validate_job_id("")


class TestValidateCudaDevices:
    def test_single_gpu(self):
        _validate_cuda_devices("0")

    def test_multiple_gpus(self):
        _validate_cuda_devices("0,1,2,3")

    def test_empty_string_allowed(self):
        _validate_cuda_devices("")

    def test_special_chars_raise(self):
        with pytest.raises(LauncherValidationError):
            _validate_cuda_devices("0;rm -rf /")

    def test_spaces_raise(self):
        with pytest.raises(LauncherValidationError):
            _validate_cuda_devices("0, 1")

    def test_letters_raise(self):
        with pytest.raises(LauncherValidationError):
            _validate_cuda_devices("gpu0")


class TestValidateNumGpus:
    def test_valid_1(self):
        _validate_num_gpus(1)

    def test_valid_8(self):
        _validate_num_gpus(8)

    def test_zero_raises(self):
        with pytest.raises(LauncherValidationError):
            _validate_num_gpus(0)

    def test_too_many_raises(self):
        with pytest.raises(LauncherValidationError):
            _validate_num_gpus(65)


class TestValidateTorchrunFlags:
    def test_empty_dict_passes(self):
        _validate_torchrun_flags({})

    def test_allowed_flag_passes(self):
        _validate_torchrun_flags({"--nnodes": "2"})

    def test_unknown_flag_raises(self):
        with pytest.raises(LauncherValidationError, match="whitelist"):
            _validate_torchrun_flags({"--unknown-flag": "value"})

    def test_injection_attempt_raises(self):
        with pytest.raises(LauncherValidationError):
            _validate_torchrun_flags({"--nproc-per-node; rm -rf /": "1"})

    def test_all_whitelisted_flags_pass(self):
        flags = {f: "1" for f in _ALLOWED_TORCHRUN_FLAGS}
        _validate_torchrun_flags(flags)


# ---------------------------------------------------------------------------
# find_free_port
# ---------------------------------------------------------------------------


class TestFindFreePort:
    def test_returns_int_in_range(self):
        port = find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535


# ---------------------------------------------------------------------------
# is_valid_uuid
# ---------------------------------------------------------------------------


class TestIsValidUuid:
    def test_valid_uuid(self):
        assert is_valid_uuid("550e8400-e29b-41d4-a716-446655440000") is True

    def test_invalid_string(self):
        assert is_valid_uuid("not-a-uuid") is False

    def test_empty_string(self):
        assert is_valid_uuid("") is False


# ---------------------------------------------------------------------------
# launch_worker (mocked Popen)
# ---------------------------------------------------------------------------


class TestLaunchWorker:
    @pytest.fixture
    def mock_proc(self):
        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 42
        return proc

    @pytest.fixture
    def job(self):
        return {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "num_gpus": 1,
            "type": "pretrain",
            "payload": "{}",
        }

    def test_launches_successfully(self, tmp_path, mock_proc, job):
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            proc = launch_worker(job, tmp_path / "test.log")
        assert proc.pid == 42
        mock_popen.assert_called_once()
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["shell"] is False

    def test_cmd_contains_job_id(self, tmp_path, mock_proc, job):
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            launch_worker(job, tmp_path / "test.log")
        cmd = mock_popen.call_args[0][0]
        assert "550e8400-e29b-41d4-a716-446655440000" in cmd

    def test_cmd_contains_nproc(self, tmp_path, mock_proc, job):
        job["num_gpus"] = 4
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            launch_worker(job, tmp_path / "test.log")
        cmd = mock_popen.call_args[0][0]
        assert any("nproc-per-node=4" in part for part in cmd)

    def test_extra_flags_included(self, tmp_path, mock_proc, job):
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            launch_worker(
                job,
                tmp_path / "test.log",
                extra_torchrun_flags={"--nnodes": "2"},
            )
        cmd = mock_popen.call_args[0][0]
        assert "--nnodes=2" in cmd

    def test_invalid_job_id_raises(self, tmp_path, job):
        job["id"] = "not-a-uuid"
        with pytest.raises(LauncherValidationError):
            launch_worker(job, tmp_path / "test.log")

    def test_invalid_cuda_raises(self, tmp_path, job):
        with pytest.raises(LauncherValidationError):
            launch_worker(job, tmp_path / "test.log", cuda_visible_devices="0;evil")

    def test_unknown_torchrun_flag_raises(self, tmp_path, job):
        with pytest.raises(LauncherValidationError):
            launch_worker(
                job,
                tmp_path / "test.log",
                extra_torchrun_flags={"--dangerous": "val"},
            )

    def test_log_dir_created(self, tmp_path, mock_proc, job):
        log_path = tmp_path / "subdir" / "test.log"
        with patch("subprocess.Popen", return_value=mock_proc):
            launch_worker(job, log_path)
        assert log_path.parent.exists()

    def test_start_new_session_true(self, tmp_path, mock_proc, job):
        """Ensures process group is created so SIGTERM hits all ranks."""
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            launch_worker(job, tmp_path / "test.log")
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs.get("start_new_session") is True


# ---------------------------------------------------------------------------
# terminate_worker
# ---------------------------------------------------------------------------


class TestTerminateWorker:
    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="POSIX killpg test skipped on Windows",
    )
    def test_sends_sigterm_to_pgid(self):
        import signal

        with (
            patch("os.getpgid", return_value=100) as mock_getpgid,
            patch("os.killpg") as mock_killpg,
        ):
            terminate_worker(42)
        mock_getpgid.assert_called_once_with(42)
        mock_killpg.assert_called_once_with(100, signal.SIGTERM)

    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="POSIX-specific OSError test skipped on Windows",
    )
    def test_oserror_logged_not_raised(self):
        with (
            patch("os.getpgid", side_effect=OSError("no such process")),
            patch("app.worker.launcher.logger") as mock_log,
        ):
            # Must not raise
            terminate_worker(99)
        mock_log.warning.assert_called_once()

    @pytest.mark.skipif(
        __import__("sys").platform != "win32",
        reason="Windows-only fallback test",
    )
    def test_windows_uses_os_kill(self):
        import signal

        with patch("os.kill") as mock_kill:
            terminate_worker(42)
        mock_kill.assert_called_once_with(42, signal.SIGTERM)
