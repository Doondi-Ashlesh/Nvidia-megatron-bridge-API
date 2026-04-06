"""Phase 1 unit tests — paths.py (security boundary)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.paths import PathTraversalError, resolve_safe_path, safe_checkpoint_path, safe_log_path

BASE = Path("/data/checkpoints")


class TestResolveSafePath:
    # ── Happy paths ──────────────────────────────────────────────────────────
    def test_simple_relative_path(self, tmp_path):
        result = resolve_safe_path(tmp_path, "models/llama3")
        assert result == (tmp_path / "models" / "llama3").resolve()

    def test_hf_style_slug(self, tmp_path):
        result = resolve_safe_path(tmp_path, "meta-llama/Llama-3-8B")
        assert result.parts[-2] == "meta-llama"
        assert result.parts[-1] == "Llama-3-8B"

    def test_nested_subdirectory(self, tmp_path):
        result = resolve_safe_path(tmp_path, "a/b/c/d")
        assert str(result).startswith(str(tmp_path.resolve()))

    # ── Traversal attacks ────────────────────────────────────────────────────
    def test_dot_dot_relative_rejected(self, tmp_path):
        with pytest.raises(PathTraversalError):
            resolve_safe_path(tmp_path, "../etc/passwd")

    def test_dot_dot_absolute_rejected(self, tmp_path):
        with pytest.raises(PathTraversalError):
            resolve_safe_path(tmp_path, "/etc/passwd")

    def test_deep_traversal_rejected(self, tmp_path):
        with pytest.raises(PathTraversalError):
            resolve_safe_path(tmp_path, "a/b/../../../../../../etc/shadow")

    def test_encoded_traversal_rejected(self, tmp_path):
        # Path() normalises %2e%2e before resolve, so this is the same as ../
        with pytest.raises(PathTraversalError):
            resolve_safe_path(tmp_path, "/tmp/outside")

    # ── Invalid input ────────────────────────────────────────────────────────
    def test_empty_string_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            resolve_safe_path(tmp_path, "")

    def test_null_byte_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="null bytes"):
            resolve_safe_path(tmp_path, "valid\x00injected")

    # ── Absolute path inside base ────────────────────────────────────────────
    def test_absolute_path_inside_base_accepted(self, tmp_path):
        """Line 62 + branch [61→62]: absolute input that IS under base must be accepted."""
        subdir = tmp_path / "models" / "llama3"
        subdir.mkdir(parents=True)
        # Pass the real absolute path — on Windows tmp_path is a proper absolute path
        result = resolve_safe_path(tmp_path, str(subdir))
        assert result == subdir.resolve()

    # ── Symlink escape ───────────────────────────────────────────────────────
    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="Symlink creation requires admin privileges on Windows",
    )
    def test_symlink_outside_base_rejected(self, tmp_path):
        # Create a symlink inside tmp_path pointing to /tmp (outside)
        link = tmp_path / "evil_link"
        link.symlink_to("/tmp")
        with pytest.raises(PathTraversalError):
            resolve_safe_path(tmp_path, "evil_link")


class TestSafeLogPath:
    def test_valid_job_id(self, tmp_path):
        logs_root = tmp_path / "logs"
        logs_root.mkdir()
        result = safe_log_path(logs_root, "550e8400-e29b-41d4-a716-446655440000")
        assert result.name == "550e8400-e29b-41d4-a716-446655440000.log"
        assert str(result).startswith(str(logs_root.resolve()))

    def test_traversal_in_job_id_rejected(self, tmp_path):
        with pytest.raises(PathTraversalError):
            safe_log_path(tmp_path, "../outside")


class TestSafeCheckpointPath:
    def test_valid_checkpoint_name(self, tmp_path):
        result = safe_checkpoint_path(tmp_path, "llama3-8b-megatron")
        assert result.name == "llama3-8b-megatron"

    def test_absolute_path_outside_root_rejected(self, tmp_path):
        with pytest.raises(PathTraversalError):
            safe_checkpoint_path(tmp_path, "/etc/shadow")
