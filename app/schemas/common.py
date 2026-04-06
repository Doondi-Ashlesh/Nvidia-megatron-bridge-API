"""Common Pydantic schemas shared across all endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    """All valid states in the job state machine."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """All valid job types dispatched to workers."""

    CHECKPOINT_IMPORT = "checkpoint_import"
    CHECKPOINT_EXPORT = "checkpoint_export"
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    LORA = "lora"
    DORA = "dora"


class ErrorDetail(BaseModel):
    """Standard error response body."""

    model_config = ConfigDict(extra="forbid")

    detail: str


class JobResponse(BaseModel):
    """Full job record returned by GET /v1/jobs/{id}."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: JobType
    status: JobStatus
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    progress: dict[str, Any] | None = None
    log_path: str | None = None
    pid: int | None = None
    num_gpus: int = 1
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    model_config = ConfigDict(extra="forbid")

    items: list[JobResponse]
    total: int


class JobCreatedResponse(BaseModel):
    """Returned immediately (HTTP 202) when a job is enqueued."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus = JobStatus.QUEUED
