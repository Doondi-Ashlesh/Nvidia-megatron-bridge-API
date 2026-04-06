"""Pydantic config tree for training jobs — Phase 4."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PrecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: Literal["float32", "float16", "bfloat16", "fp8"] = "bfloat16"
    fp8_margin: int = Field(default=0, ge=0)
    fp8_interval: int = Field(default=1, ge=1)
    fp8_amax_history_len: int = Field(default=1024, ge=1)
    fp8_amax_compute_algo: Literal["most_recent", "max"] = "most_recent"


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["adam", "sgd", "adamw"] = "adamw"
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    clip_grad: float = Field(default=1.0, ge=0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str
    seq_length: int = Field(default=2048, ge=1)
    micro_batch_size: int = Field(default=4, ge=1)
    global_batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=4, ge=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arch: str = "llama3"
    num_layers: int = Field(default=32, ge=1)
    hidden_size: int = Field(default=4096, ge=1)
    num_attention_heads: int = Field(default=32, ge=1)
    seq_length: int = Field(default=2048, ge=1)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_steps: int = Field(default=1000, ge=1)
    save_interval: int = Field(default=100, ge=1)
    log_interval: int = Field(default=10, ge=1)
    eval_interval: int = Field(default=100, ge=1)
    output_dir: str = ""


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(default=16, ge=1)
    alpha: float = Field(default=32.0, gt=0)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class DoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(default=16, ge=1)
    alpha: float = Field(default=32.0, gt=0)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    use_weight_decomposition: bool = True


class ConfigContainer(BaseModel):
    """Top-level config passed as job payload for all training jobs."""

    model_config = ConfigDict(extra="forbid")

    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig | None = None
    dora: DoRAConfig | None = None
