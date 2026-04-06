"""Training service — converts Pydantic ConfigContainer to SDK flat dict."""

from __future__ import annotations

from typing import Any

from app.schemas.config_container import ConfigContainer


def pydantic_to_sdk_config(config: ConfigContainer) -> dict[str, Any]:
    """Flatten a ConfigContainer into the dict that MegatronBridge SDK expects."""
    result: dict[str, Any] = {}

    # precision
    result["dtype"] = config.precision.dtype
    if config.precision.dtype == "fp8":
        result["fp8"] = True
        result["fp8_margin"] = config.precision.fp8_margin
        result["fp8_interval"] = config.precision.fp8_interval
        result["fp8_amax_history_len"] = config.precision.fp8_amax_history_len
        result["fp8_amax_compute_algo"] = config.precision.fp8_amax_compute_algo

    # optimizer
    result["optimizer"] = config.optimizer.name
    result["lr"] = config.optimizer.lr
    result["weight_decay"] = config.optimizer.weight_decay
    result["clip_grad"] = config.optimizer.clip_grad

    # data
    result["dataset_path"] = config.data.dataset_path
    result["seq_length"] = config.data.seq_length
    result["micro_batch_size"] = config.data.micro_batch_size
    result["global_batch_size"] = config.data.global_batch_size

    # model
    result["model_arch"] = config.model.arch
    result["num_layers"] = config.model.num_layers
    result["hidden_size"] = config.model.hidden_size
    result["num_attention_heads"] = config.model.num_attention_heads

    # training
    result["train_steps"] = config.training.train_steps
    result["save_interval"] = config.training.save_interval
    result["log_interval"] = config.training.log_interval
    result["output_dir"] = config.training.output_dir

    # lora/dora
    if config.lora is not None:
        result["lora_rank"] = config.lora.rank
        result["lora_alpha"] = config.lora.alpha
        result["lora_dropout"] = config.lora.dropout
        result["lora_target_modules"] = config.lora.target_modules
    if config.dora is not None:
        result["dora_rank"] = config.dora.rank
        result["dora_alpha"] = config.dora.alpha
        result["dora_dropout"] = config.dora.dropout
        result["dora_target_modules"] = config.dora.target_modules
        result["dora_use_weight_decomposition"] = config.dora.use_weight_decomposition

    return result
