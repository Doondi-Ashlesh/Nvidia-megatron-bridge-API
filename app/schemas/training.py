"""Request schemas for training and PEFT endpoints — Phase 4."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.config_container import ConfigContainer


class PretrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: ConfigContainer
    num_gpus: int = Field(default=1, ge=1, le=64)
    checkpoint_name: str | None = None


class FinetuneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: ConfigContainer
    num_gpus: int = Field(default=1, ge=1, le=64)
    checkpoint_name: str


class LoRARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: ConfigContainer
    num_gpus: int = Field(default=1, ge=1, le=64)
    checkpoint_name: str


class DoRARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: ConfigContainer
    num_gpus: int = Field(default=1, ge=1, le=64)
    checkpoint_name: str
