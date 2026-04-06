"""Pydantic schemas for checkpoint endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CheckpointFormat(str, Enum):
    MEGATRON = "megatron"
    HF = "hf"


class ImportRequest(BaseModel):
    """HuggingFace -> Megatron conversion request."""

    model_config = ConfigDict(extra="forbid")

    source_path: str
    target_name: str
    hf_token: str | None = None
    num_gpus: int = Field(default=1, ge=1, le=64)
    model_arch: str | None = None


class ExportRequest(BaseModel):
    """Megatron -> HuggingFace conversion request."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    target_name: str
    # Required: AutoBridge.from_hf_pretrained() needs the HF model ID to load
    # architecture config even when exporting. E.g. "meta-llama/Llama-3-8B".
    model_arch: str
    num_gpus: int = Field(default=1, ge=1, le=64)


class CheckpointRecord(BaseModel):
    """A checkpoint record as stored in the DB."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    format: CheckpointFormat
    path: str
    size_bytes: int | None = None
    model_arch: str | None = None
    created_from_job: str | None = None
    created_at: datetime


class CheckpointListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CheckpointRecord]
    total: int
