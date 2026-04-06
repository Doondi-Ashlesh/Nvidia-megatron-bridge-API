"""GPU worker executor — the ONLY file that imports megatron.bridge.

This module is launched by torchrun as a subprocess. It uses stdlib sqlite3
(synchronous) because asyncio cannot be used inside torchrun worker processes.

Real SDK used: megatron-bridge v0.3.1 (pypi: megatron-bridge)
Install: pip install --no-build-isolation megatron-bridge
Requires: CUDA >= 12.8, PyTorch >= 2.7, Python >= 3.12
NGC container: nvcr.io/nvidia/nemo:<TAG>

Import path verified against:
  github.com/NVIDIA-NeMo/Megatron-Bridge/src/megatron/bridge/models/conversion/auto_bridge.py
  docs.nvidia.com/nemo/megatron-bridge/latest/
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLite helpers (synchronous — stdlib sqlite3, not aiosqlite)
# ---------------------------------------------------------------------------


def _get_db_path() -> str:
    """Read DATABASE_URL from environment and extract the filesystem path."""
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///megatronbridge.db")
    prefix = "sqlite+aiosqlite:///"
    if url.startswith(prefix):
        return url[len(prefix):]
    return url


def load_job(job_id: str) -> dict:  # type: ignore[type-arg]
    """Read a job record from SQLite synchronously."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if row is None:
            msg = f"Job {job_id!r} not found in DB"
            raise ValueError(msg)
        return dict(row)
    finally:
        conn.close()


def mark_running(job_id: str) -> None:
    """Update job status to 'running'."""
    db_path = _get_db_path()
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE jobs SET status='running', started_at=? WHERE id=?",
            (now, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_completed(job_id: str) -> None:
    """Update job status to 'completed'."""
    db_path = _get_db_path()
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE jobs SET status='completed', completed_at=? WHERE id=?",
            (now, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def fail_job(job_id: str, error: str) -> None:
    """Update job status to 'failed' with an error message."""
    db_path = _get_db_path()
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE jobs SET status='failed', error=?, completed_at=? WHERE id=?",
            (error, now, job_id),
        )
        conn.commit()
    finally:
        conn.close()
    logger.error("Job %s failed: %s", job_id, error)


# ---------------------------------------------------------------------------
# Job handlers — real megatron-bridge API (verified v0.3.1)
# ---------------------------------------------------------------------------


def handle_checkpoint_import(job: dict) -> None:  # type: ignore[type-arg]
    """Convert a HuggingFace checkpoint to Megatron format.

    Real API (verified from auto_bridge.py):
        AutoBridge.import_ckpt(hf_model_id, megatron_path, **kwargs)

    Source: github.com/NVIDIA-NeMo/Megatron-Bridge/src/megatron/bridge/
                    models/conversion/auto_bridge.py
    """
    from megatron.bridge import AutoBridge  # noqa: PLC0415 — GPU-only, late import

    payload = json.loads(job["payload"])

    # Extract HF_TOKEN: set as env var so HuggingFace hub picks it up automatically.
    # Never pass as a function argument (would appear in stack traces / logs).
    hf_token = payload.pop("hf_token", None)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Verified signature: AutoBridge.import_ckpt(hf_model_id, megatron_path, **kwargs)
    AutoBridge.import_ckpt(
        hf_model_id=payload["source_path"],
        megatron_path=payload["target_path"],
    )


def handle_checkpoint_export(job: dict) -> None:  # type: ignore[type-arg]
    """Convert a Megatron checkpoint to HuggingFace format.

    Real API (verified from auto_bridge.py):
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        bridge.export_ckpt(megatron_path, hf_path, ...)

    The hf_model_id is needed to load the architecture config even during export.
    It is stored in the job payload as 'model_arch'.
    """
    from megatron.bridge import AutoBridge  # noqa: PLC0415 — GPU-only, late import

    payload = json.loads(job["payload"])

    # AutoBridge needs the HF model architecture to know how to export.
    # model_arch is the HF model ID or repo name (e.g. "meta-llama/Llama-3-8B").
    model_arch = payload.get("model_arch", "")

    bridge = AutoBridge.from_hf_pretrained(model_arch)

    # Verified signature: bridge.export_ckpt(megatron_path, hf_path, show_progress, strict)
    bridge.export_ckpt(
        megatron_path=payload["source_path"],
        hf_path=payload["target_path"],
        show_progress=True,
        strict=False,
    )


def handle_pretrain(job: dict) -> None:  # type: ignore[type-arg]
    """Run a pretraining job.

    Real API (verified from training/pretrain.py):
        from megatron.bridge.training.pretrain import pretrain
        from megatron.bridge.training.gpt_step import forward_step
        pretrain(config, forward_step)

    The config object is built from the payload dict that was already
    flattened by pydantic_to_sdk_config() before being stored in the DB.

    Note: this script is launched by torchrun — distributed init is handled
    by megatron.bridge internals, not here.
    """
    from megatron.bridge.training.config import (  # noqa: PLC0415
        CheckpointConfig,
        ConfigContainer,
        OptimizerConfig,
        TrainingConfig,
    )
    from megatron.bridge.training.gpt_step import forward_step  # noqa: PLC0415
    from megatron.bridge.training.pretrain import pretrain  # noqa: PLC0415

    payload = json.loads(job["payload"])
    sdk_config = payload.get("sdk_config", payload)

    config = ConfigContainer(
        train=TrainingConfig(
            train_iters=sdk_config.get("train_steps", 1000),
            global_batch_size=sdk_config.get("global_batch_size", 32),
            micro_batch_size=sdk_config.get("micro_batch_size", 4),
            log_interval=sdk_config.get("log_interval", 10),
            eval_interval=sdk_config.get("eval_interval", 100),
            save_interval=sdk_config.get("save_interval", 100),
        ),
        optimizer=OptimizerConfig(
            optimizer=sdk_config.get("optimizer", "adamw"),
            lr=sdk_config.get("lr", 3e-4),
            weight_decay=sdk_config.get("weight_decay", 0.01),
            clip_grad=sdk_config.get("clip_grad", 1.0),
            bf16=sdk_config.get("dtype") == "bfloat16",
            fp16=sdk_config.get("dtype") == "float16",
            fp8=sdk_config.get("dtype") == "fp8",
        ),
        checkpoint=CheckpointConfig(
            save=sdk_config.get("output_dir", "/data/checkpoints"),
            pretrained_checkpoint=sdk_config.get("pretrained_checkpoint"),
        ),
    )

    pretrain(config, forward_step)


def handle_finetune(job: dict) -> None:  # type: ignore[type-arg]
    """Run a supervised fine-tuning job.

    Real API (verified from training/finetune.py):
        from megatron.bridge.training.finetune import finetune
        finetune(config, forward_step)
    """
    from megatron.bridge.training.config import (  # noqa: PLC0415
        CheckpointConfig,
        ConfigContainer,
        OptimizerConfig,
        TrainingConfig,
    )
    from megatron.bridge.training.finetune import finetune  # noqa: PLC0415
    from megatron.bridge.training.gpt_step import forward_step  # noqa: PLC0415

    payload = json.loads(job["payload"])
    sdk_config = payload.get("sdk_config", payload)

    config = ConfigContainer(
        train=TrainingConfig(
            train_iters=sdk_config.get("train_steps", 1000),
            global_batch_size=sdk_config.get("global_batch_size", 32),
            micro_batch_size=sdk_config.get("micro_batch_size", 4),
        ),
        optimizer=OptimizerConfig(
            optimizer=sdk_config.get("optimizer", "adamw"),
            lr=sdk_config.get("lr", 3e-4),
            bf16=sdk_config.get("dtype") == "bfloat16",
        ),
        checkpoint=CheckpointConfig(
            save=sdk_config.get("output_dir", "/data/checkpoints"),
            pretrained_checkpoint=sdk_config.get("pretrained_checkpoint"),
        ),
    )

    finetune(config, forward_step)


def handle_lora(job: dict) -> None:  # type: ignore[type-arg]
    """Run a LoRA fine-tuning job.

    Real API (verified from peft/lora.py):
        from megatron.bridge.peft.lora import LoRA
        config.peft = LoRA(target_modules=[...], dim=16, alpha=32, dropout=0.1)

    docs: docs.nvidia.com/nemo/megatron-bridge/0.2.0/training/peft.html
    """
    from megatron.bridge.peft.lora import LoRA  # noqa: PLC0415
    from megatron.bridge.training.config import (  # noqa: PLC0415
        CheckpointConfig,
        ConfigContainer,
        OptimizerConfig,
        TrainingConfig,
    )
    from megatron.bridge.training.finetune import finetune  # noqa: PLC0415
    from megatron.bridge.training.gpt_step import forward_step  # noqa: PLC0415

    payload = json.loads(job["payload"])
    sdk_config = payload.get("sdk_config", payload)

    # Verified LoRA constructor: LoRA(target_modules, dim, alpha, dropout, ...)
    lora = LoRA(
        target_modules=sdk_config.get(
            "lora_target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        ),
        dim=sdk_config.get("lora_rank", 16),
        alpha=sdk_config.get("lora_alpha", 32.0),
        dropout=sdk_config.get("lora_dropout", 0.05),
    )

    config = ConfigContainer(
        train=TrainingConfig(
            train_iters=sdk_config.get("train_steps", 1000),
            global_batch_size=sdk_config.get("global_batch_size", 32),
            micro_batch_size=sdk_config.get("micro_batch_size", 4),
        ),
        optimizer=OptimizerConfig(
            optimizer=sdk_config.get("optimizer", "adamw"),
            lr=sdk_config.get("lr", 3e-4),
            bf16=sdk_config.get("dtype") == "bfloat16",
        ),
        checkpoint=CheckpointConfig(
            # pretrained_checkpoint is REQUIRED for PEFT — base model weights
            pretrained_checkpoint=sdk_config.get("pretrained_checkpoint", ""),
            save=sdk_config.get("output_dir", "/data/checkpoints"),
        ),
        peft=lora,
    )

    finetune(config, forward_step)


def handle_dora(job: dict) -> None:  # type: ignore[type-arg]
    """Run a DoRA fine-tuning job.

    Real API (verified from peft/dora.py):
        from megatron.bridge.peft.dora import DoRA
        config.peft = DoRA(target_modules=[...], dim=16, alpha=32, dropout=0.1)
    """
    from megatron.bridge.peft.dora import DoRA  # noqa: PLC0415
    from megatron.bridge.training.config import (  # noqa: PLC0415
        CheckpointConfig,
        ConfigContainer,
        OptimizerConfig,
        TrainingConfig,
    )
    from megatron.bridge.training.finetune import finetune  # noqa: PLC0415
    from megatron.bridge.training.gpt_step import forward_step  # noqa: PLC0415

    payload = json.loads(job["payload"])
    sdk_config = payload.get("sdk_config", payload)

    # Verified DoRA constructor: DoRA(target_modules, dim, alpha, dropout, ...)
    dora = DoRA(
        target_modules=sdk_config.get(
            "dora_target_modules", ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        ),
        dim=sdk_config.get("dora_rank", 16),
        alpha=sdk_config.get("dora_alpha", 32.0),
        dropout=sdk_config.get("dora_dropout", 0.05),
    )

    config = ConfigContainer(
        train=TrainingConfig(
            train_iters=sdk_config.get("train_steps", 1000),
            global_batch_size=sdk_config.get("global_batch_size", 32),
            micro_batch_size=sdk_config.get("micro_batch_size", 4),
        ),
        optimizer=OptimizerConfig(
            optimizer=sdk_config.get("optimizer", "adamw"),
            lr=sdk_config.get("lr", 3e-4),
            bf16=sdk_config.get("dtype") == "bfloat16",
        ),
        checkpoint=CheckpointConfig(
            pretrained_checkpoint=sdk_config.get("pretrained_checkpoint", ""),
            save=sdk_config.get("output_dir", "/data/checkpoints"),
        ),
        peft=dora,
    )

    finetune(config, forward_step)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> str:
    """Parse --job-id from sys.argv and return the job ID string."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--job-id" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--job-id="):
            return arg.split("=", 1)[1]
    msg = "--job-id argument is required"
    raise SystemExit(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    job_id = parse_args()
    job = load_job(job_id)

    handlers = {
        "checkpoint_import": handle_checkpoint_import,
        "checkpoint_export": handle_checkpoint_export,
        "pretrain": handle_pretrain,
        "finetune": handle_finetune,
        "lora": handle_lora,
        "dora": handle_dora,
    }

    handler = handlers.get(job["type"])
    if handler is None:
        fail_job(job_id, f"Unknown job type: {job['type']}")
        return

    try:
        mark_running(job_id)
        handler(job)
        mark_completed(job_id)
    except Exception as exc:  # noqa: BLE001
        fail_job(job_id, str(exc))


if __name__ == "__main__":
    main()
