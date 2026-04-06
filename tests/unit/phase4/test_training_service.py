"""Unit tests for pydantic_to_sdk_config in training_service."""

from __future__ import annotations

import pytest

from app.schemas.config_container import (
    ConfigContainer,
    DataConfig,
    DoRAConfig,
    LoRAConfig,
    PrecisionConfig,
)
from app.services.training_service import pydantic_to_sdk_config


def _make_config(**kwargs) -> ConfigContainer:  # type: ignore[return]
    return ConfigContainer(data=DataConfig(dataset_path="/data/train"), **kwargs)


# ---------------------------------------------------------------------------
# Default config keys
# ---------------------------------------------------------------------------


def test_default_config_expected_keys():
    config = _make_config()
    result = pydantic_to_sdk_config(config)

    assert "dtype" in result
    assert result["dtype"] == "bfloat16"
    assert "optimizer" in result
    assert result["optimizer"] == "adamw"
    assert "dataset_path" in result
    assert result["dataset_path"] == "/data/train"
    assert "model_arch" in result
    assert result["model_arch"] == "llama3"
    assert "train_steps" in result
    assert result["train_steps"] == 1000
    assert "seq_length" in result
    assert "micro_batch_size" in result
    assert "global_batch_size" in result
    assert "num_layers" in result
    assert "hidden_size" in result
    assert "num_attention_heads" in result
    assert "save_interval" in result
    assert "log_interval" in result
    assert "output_dir" in result


# ---------------------------------------------------------------------------
# FP8 handling
# ---------------------------------------------------------------------------


def test_fp8_dtype_includes_fp8_flag():
    config = _make_config(precision=PrecisionConfig(dtype="fp8"))
    result = pydantic_to_sdk_config(config)

    assert result["dtype"] == "fp8"
    assert result.get("fp8") is True
    assert "fp8_margin" in result
    assert "fp8_interval" in result
    assert "fp8_amax_history_len" in result
    assert "fp8_amax_compute_algo" in result


def test_non_fp8_dtype_excludes_fp8_flag():
    config = _make_config(precision=PrecisionConfig(dtype="float16"))
    result = pydantic_to_sdk_config(config)

    assert result["dtype"] == "float16"
    assert "fp8" not in result
    assert "fp8_margin" not in result


def test_bfloat16_excludes_fp8_keys():
    config = _make_config(precision=PrecisionConfig(dtype="bfloat16"))
    result = pydantic_to_sdk_config(config)

    assert "fp8" not in result


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


def test_lora_config_included():
    config = _make_config(lora=LoRAConfig(rank=8, alpha=16.0))
    result = pydantic_to_sdk_config(config)

    assert result["lora_rank"] == 8
    assert result["lora_alpha"] == pytest.approx(16.0)
    assert "lora_dropout" in result
    assert "lora_target_modules" in result


def test_no_lora_excludes_lora_keys():
    config = _make_config()
    result = pydantic_to_sdk_config(config)

    assert "lora_rank" not in result
    assert "lora_alpha" not in result


# ---------------------------------------------------------------------------
# DoRA
# ---------------------------------------------------------------------------


def test_dora_config_included():
    config = _make_config(dora=DoRAConfig(rank=4, alpha=8.0))
    result = pydantic_to_sdk_config(config)

    assert result["dora_rank"] == 4
    assert result["dora_alpha"] == pytest.approx(8.0)
    assert "dora_dropout" in result
    assert "dora_target_modules" in result
    assert "dora_use_weight_decomposition" in result


def test_no_dora_excludes_dora_keys():
    config = _make_config()
    result = pydantic_to_sdk_config(config)

    assert "dora_rank" not in result
    assert "dora_alpha" not in result


# ---------------------------------------------------------------------------
# Combined lora + dora
# ---------------------------------------------------------------------------


def test_both_lora_and_dora_included():
    config = _make_config(
        lora=LoRAConfig(rank=16),
        dora=DoRAConfig(rank=8),
    )
    result = pydantic_to_sdk_config(config)

    assert "lora_rank" in result
    assert "dora_rank" in result
