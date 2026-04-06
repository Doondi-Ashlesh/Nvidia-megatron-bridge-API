"""Unit tests for Phase 4 config container schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.config_container import (
    ConfigContainer,
    DataConfig,
    DoRAConfig,
    LoRAConfig,
    ModelConfig,
    OptimizerConfig,
    PrecisionConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# PrecisionConfig
# ---------------------------------------------------------------------------


def test_precision_defaults():
    pc = PrecisionConfig()
    assert pc.dtype == "bfloat16"
    assert pc.fp8_margin == 0
    assert pc.fp8_interval == 1
    assert pc.fp8_amax_history_len == 1024
    assert pc.fp8_amax_compute_algo == "most_recent"


def test_precision_fp8_dtype_accepted():
    pc = PrecisionConfig(dtype="fp8")
    assert pc.dtype == "fp8"


def test_precision_invalid_dtype_raises():
    with pytest.raises(ValidationError):
        PrecisionConfig(dtype="int8")  # type: ignore[arg-type]


def test_precision_extra_forbid_raises():
    with pytest.raises(ValidationError):
        PrecisionConfig(unknown_field="bad")  # type: ignore[call-arg]


def test_precision_fp8_margin_negative_raises():
    with pytest.raises(ValidationError):
        PrecisionConfig(fp8_margin=-1)


def test_precision_fp8_interval_zero_raises():
    with pytest.raises(ValidationError):
        PrecisionConfig(fp8_interval=0)


# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------


def test_optimizer_defaults():
    oc = OptimizerConfig()
    assert oc.name == "adamw"
    assert oc.lr == pytest.approx(3e-4)
    assert oc.weight_decay == pytest.approx(0.01)
    assert oc.clip_grad == pytest.approx(1.0)


def test_optimizer_extra_forbid_raises():
    with pytest.raises(ValidationError):
        OptimizerConfig(unknown="oops")  # type: ignore[call-arg]


def test_optimizer_lr_zero_raises():
    with pytest.raises(ValidationError):
        OptimizerConfig(lr=0.0)


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


def test_data_config_requires_dataset_path():
    with pytest.raises(ValidationError):
        DataConfig()  # type: ignore[call-arg]


def test_data_config_defaults():
    dc = DataConfig(dataset_path="/data/train")
    assert dc.seq_length == 2048
    assert dc.micro_batch_size == 4
    assert dc.global_batch_size == 32
    assert dc.num_workers == 4


def test_data_config_extra_forbid_raises():
    with pytest.raises(ValidationError):
        DataConfig(dataset_path="/data", extra_field="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


def test_model_config_defaults():
    mc = ModelConfig()
    assert mc.arch == "llama3"
    assert mc.num_layers == 32
    assert mc.hidden_size == 4096
    assert mc.num_attention_heads == 32


def test_model_config_extra_forbid_raises():
    with pytest.raises(ValidationError):
        ModelConfig(bad_key="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_defaults():
    tc = TrainingConfig()
    assert tc.train_steps == 1000
    assert tc.save_interval == 100
    assert tc.log_interval == 10
    assert tc.eval_interval == 100
    assert tc.output_dir == ""


def test_training_config_extra_forbid_raises():
    with pytest.raises(ValidationError):
        TrainingConfig(nope="nope")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# LoRAConfig
# ---------------------------------------------------------------------------


def test_lora_defaults():
    lc = LoRAConfig()
    assert lc.rank == 16
    assert lc.alpha == pytest.approx(32.0)
    assert lc.dropout == pytest.approx(0.05)
    assert lc.target_modules == ["q_proj", "v_proj"]


def test_lora_extra_forbid_raises():
    with pytest.raises(ValidationError):
        LoRAConfig(unknown="x")  # type: ignore[call-arg]


def test_lora_dropout_out_of_range_raises():
    with pytest.raises(ValidationError):
        LoRAConfig(dropout=1.0)


# ---------------------------------------------------------------------------
# DoRAConfig
# ---------------------------------------------------------------------------


def test_dora_defaults():
    dc = DoRAConfig()
    assert dc.rank == 16
    assert dc.alpha == pytest.approx(32.0)
    assert dc.use_weight_decomposition is True
    assert dc.target_modules == ["q_proj", "v_proj"]


def test_dora_extra_forbid_raises():
    with pytest.raises(ValidationError):
        DoRAConfig(nope="nope")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ConfigContainer
# ---------------------------------------------------------------------------


def test_config_container_requires_data():
    with pytest.raises(ValidationError):
        ConfigContainer()  # type: ignore[call-arg]


def test_config_container_minimal():
    cc = ConfigContainer(data=DataConfig(dataset_path="/data/train"))
    assert cc.precision.dtype == "bfloat16"
    assert cc.lora is None
    assert cc.dora is None


def test_config_container_with_lora():
    cc = ConfigContainer(
        data=DataConfig(dataset_path="/data/train"),
        lora=LoRAConfig(rank=8),
    )
    assert cc.lora is not None
    assert cc.lora.rank == 8


def test_config_container_with_dora():
    cc = ConfigContainer(
        data=DataConfig(dataset_path="/data/train"),
        dora=DoRAConfig(rank=4),
    )
    assert cc.dora is not None
    assert cc.dora.rank == 4


def test_config_container_extra_forbid_raises():
    with pytest.raises(ValidationError):
        ConfigContainer(
            data=DataConfig(dataset_path="/data/train"),
            rogue_field="oops",  # type: ignore[call-arg]
        )
