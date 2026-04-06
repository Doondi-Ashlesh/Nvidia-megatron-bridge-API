"""Application settings loaded from environment variables / .env file."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for MegatronBridge API.

    Values are read (in priority order) from:
    1. Real environment variables
    2. A `.env` file in the current working directory
    3. Defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Storage ──────────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=Path("/data"),
        description="Root directory for all runtime data. Created on startup if absent.",
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///megatronbridge.db",
        description="SQLite connection URL. Relative paths resolved under data_dir.",
    )
    checkpoints_root: Path = Field(
        default=Path("/data/checkpoints"),
        description="Root directory for model checkpoints. All user-supplied paths "
        "must resolve to a child of this directory.",
    )
    logs_root: Path = Field(
        default=Path("/data/logs"),
        description="Root directory for per-job log files.",
    )
    hf_cache_dir: Path = Field(
        default=Path("/data/hf_cache"),
        description="HuggingFace model cache directory.",
    )

    # ── Worker ───────────────────────────────────────────────────────────────
    max_concurrent_jobs: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Maximum number of GPU jobs that may run simultaneously.",
    )
    dispatcher_poll_interval_s: Annotated[float, Field(gt=0)] = Field(
        default=2.0,
        description="How often (seconds) the dispatcher checks for queued jobs.",
    )
    cuda_visible_devices: str = Field(
        default="0",
        description="Comma-separated GPU indices passed to worker subprocesses.",
    )

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")  # noqa: S104  # nosec B104 — binding all interfaces is intentional for a server
    api_port: Annotated[int, Field(ge=1, le=65535)] = Field(default=8000)
    log_level: str = Field(default="INFO")
    # Stored as a raw comma-separated string to avoid pydantic-settings JSON-parsing
    # an empty string as invalid JSON.  Use ``cors_origins_list`` everywhere in code.
    cors_origins: str = Field(
        default="",
        description=(
            "Comma-separated allowed CORS origins. "
            "Empty string disables CORS middleware. "
            "Example: 'http://localhost:3000,https://myapp.example.com'"
        ),
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a list, parsed from the raw comma-separated string."""
        if not self.cors_origins.strip():
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ── Optional: S3 ─────────────────────────────────────────────────────────
    s3_enabled: bool = Field(default=False)
    s3_bucket: str = Field(default="")
    s3_region: str = Field(default="")

    # ── Derived / computed ────────────────────────────────────────────────────
    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"LOG_LEVEL must be one of {allowed}, got {v!r}"
            raise ValueError(msg)
        return upper

    @model_validator(mode="after")
    def resolve_database_url(self) -> Settings:
        """Resolve a relative SQLite path against data_dir."""
        url = self.database_url
        prefix = "sqlite+aiosqlite:///"
        if url.startswith(prefix):
            raw_path = url[len(prefix):]
            p = Path(raw_path)
            if not p.is_absolute():
                p = self.data_dir / p
            self.database_url = f"{prefix}{p}"
        return self

    @property
    def sqlite_path(self) -> Path:
        """Convenience: extract the filesystem path from database_url."""
        prefix = "sqlite+aiosqlite:///"
        return Path(self.database_url[len(prefix):])

    @property
    def version(self) -> str:
        try:
            return importlib.metadata.version("megatronbridge-api")
        except importlib.metadata.PackageNotFoundError:
            return "0.1.0-dev"


# Module-level singleton — imported everywhere as `from app.config import settings`
settings = Settings()
