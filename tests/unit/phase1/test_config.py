"""Phase 1 unit tests — Settings / config.py."""

from __future__ import annotations

import importlib
import os

import pytest
from pydantic import ValidationError


def _reload_settings(**env_overrides):
    """Reload config module with custom env vars, return fresh Settings()."""
    original = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update({k: str(v) for k, v in env_overrides.items()})
    try:
        import app.config as mod
        importlib.reload(mod)
        return mod.Settings()
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(mod)


class TestLogLevel:
    def test_valid_log_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            s = _reload_settings(LOG_LEVEL=level)
            assert s.log_level == level

    def test_log_level_case_insensitive(self):
        s = _reload_settings(LOG_LEVEL="debug")
        assert s.log_level == "DEBUG"

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValidationError):
            _reload_settings(LOG_LEVEL="VERBOSE")


class TestCorsOrigins:
    def test_empty_string_produces_empty_list(self):
        s = _reload_settings(CORS_ORIGINS="")
        assert s.cors_origins_list == []

    def test_comma_separated_string_parsed(self):
        s = _reload_settings(CORS_ORIGINS="http://a.com,http://b.com")
        assert s.cors_origins_list == ["http://a.com", "http://b.com"]

    def test_no_wildcard_default(self):
        s = _reload_settings(CORS_ORIGINS="")
        assert "*" not in s.cors_origins_list


class TestDatabaseUrl:
    def test_relative_url_resolved_under_data_dir(self, tmp_path):
        s = _reload_settings(
            DATA_DIR=str(tmp_path),
            DATABASE_URL="sqlite+aiosqlite:///megatronbridge.db",
        )
        assert s.sqlite_path == tmp_path / "megatronbridge.db"

    def test_absolute_url_not_modified(self, tmp_path):
        abs_db = str(tmp_path / "abs" / "db.sqlite")
        s = _reload_settings(
            DATA_DIR=str(tmp_path),
            DATABASE_URL=f"sqlite+aiosqlite:///{abs_db}",
        )
        assert s.sqlite_path == tmp_path / "abs" / "db.sqlite"


class TestVersion:
    def test_version_returns_string(self):
        from app.config import settings
        assert isinstance(settings.version, str)
        assert len(settings.version) > 0


class TestMaxConcurrentJobs:
    def test_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            _reload_settings(MAX_CONCURRENT_JOBS="0")


class TestDatabaseUrlNonSqlite:
    def test_non_sqlite_url_passes_through_unmodified(self):
        """Branch [109 → 115]: url does NOT start with sqlite+aiosqlite:///."""
        s = _reload_settings(DATABASE_URL="postgresql://user:pass@localhost/db")
        # resolve_database_url must not crash and must leave the URL intact
        assert s.database_url == "postgresql://user:pass@localhost/db"


class TestVersionFallback:
    def test_version_returns_dev_when_package_not_installed(self, monkeypatch):
        """Lines 127-128: PackageNotFoundError branch inside version property."""
        import importlib.metadata as meta

        monkeypatch.setattr(
            meta,
            "version",
            lambda _name: (_ for _ in ()).throw(meta.PackageNotFoundError("megatronbridge-api")),
        )
        # Re-read the property — it must fall back gracefully
        from app.config import settings
        assert settings.version == "0.1.0-dev"
