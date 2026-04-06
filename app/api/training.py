"""Training endpoints: /v1/training — Phase 4."""

from __future__ import annotations

import logging

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db
from app.schemas.common import JobCreatedResponse, JobListResponse, JobResponse, JobType
from app.schemas.training import FinetuneRequest, PretrainRequest
from app.services import checkpoint_service, job_service
from app.services.training_service import pydantic_to_sdk_config
from app.utils.paths import PathTraversalError, resolve_safe_path, safe_checkpoint_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/training", tags=["Training"])


def _get_settings():  # type: ignore[return]
    from app import config as _cfg  # noqa: PLC0415
    return _cfg.settings


# ---------------------------------------------------------------------------
# Path validation helpers
# ---------------------------------------------------------------------------


def _validate_dataset_path(dataset_path: str) -> str:
    """Validate dataset_path against data_dir. Returns the original string on success."""
    settings = _get_settings()
    try:
        resolve_safe_path(settings.data_dir, dataset_path)
    except (PathTraversalError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dataset_path


def _validate_output_dir(output_dir: str) -> str:
    """Validate output_dir against checkpoints_root if non-empty."""
    if not output_dir:
        return output_dir
    settings = _get_settings()
    try:
        safe_checkpoint_path(settings.checkpoints_root, output_dir)
    except (PathTraversalError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return output_dir


async def _assert_checkpoint_name_exists(
    db: aiosqlite.Connection, checkpoint_name: str
) -> None:
    """Verify checkpoint_name exists in DB. Raises 404 if not found."""
    items, _ = await checkpoint_service.list_checkpoints(db)
    found = any(r["name"] == checkpoint_name for r in items)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint {checkpoint_name!r} not found",
        )


# ---------------------------------------------------------------------------
# POST /v1/training/pretrain
# ---------------------------------------------------------------------------


@router.post(
    "/pretrain",
    status_code=202,
    response_model=JobCreatedResponse,
    summary="Submit a pretraining job",
)
async def pretrain(
    request: PretrainRequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> JobCreatedResponse:
    _validate_dataset_path(request.config.data.dataset_path)
    _validate_output_dir(request.config.training.output_dir)

    if request.checkpoint_name is not None:
        await _assert_checkpoint_name_exists(db, request.checkpoint_name)

    sdk_config = pydantic_to_sdk_config(request.config)
    payload = {
        **request.model_dump(),
        "sdk_config": sdk_config,
    }

    job_id = await job_service.create_job(
        db,
        job_type=JobType.PRETRAIN,
        payload=payload,
        num_gpus=request.num_gpus,
    )
    logger.info("Pretrain job %s queued (gpus=%d)", job_id, request.num_gpus)
    return JobCreatedResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# POST /v1/training/finetune
# ---------------------------------------------------------------------------


@router.post(
    "/finetune",
    status_code=202,
    response_model=JobCreatedResponse,
    summary="Submit a finetuning job",
)
async def finetune(
    request: FinetuneRequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> JobCreatedResponse:
    _validate_dataset_path(request.config.data.dataset_path)
    _validate_output_dir(request.config.training.output_dir)

    await _assert_checkpoint_name_exists(db, request.checkpoint_name)

    sdk_config = pydantic_to_sdk_config(request.config)
    payload = {
        **request.model_dump(),
        "sdk_config": sdk_config,
    }

    job_id = await job_service.create_job(
        db,
        job_type=JobType.FINETUNE,
        payload=payload,
        num_gpus=request.num_gpus,
    )
    logger.info("Finetune job %s queued (gpus=%d)", job_id, request.num_gpus)
    return JobCreatedResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# GET /v1/training
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=JobListResponse,
    summary="List pretrain and finetune jobs",
)
async def list_training_jobs(
    db: aiosqlite.Connection = Depends(get_db),
) -> JobListResponse:
    pretrain_items, pretrain_total = await job_service.list_jobs(
        db, job_type=JobType.PRETRAIN
    )
    finetune_items, finetune_total = await job_service.list_jobs(
        db, job_type=JobType.FINETUNE
    )

    all_items = pretrain_items + finetune_items
    total = pretrain_total + finetune_total

    return JobListResponse(
        items=[JobResponse(**item) for item in all_items],
        total=total,
    )
