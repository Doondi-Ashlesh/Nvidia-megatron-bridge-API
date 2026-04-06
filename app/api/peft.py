"""PEFT endpoints: /v1/peft — Phase 4."""

from __future__ import annotations

import logging

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from app.database import get_db
from app.schemas.common import JobCreatedResponse, JobListResponse, JobResponse, JobType
from app.schemas.training import DoRARequest, LoRARequest
from app.services import checkpoint_service, job_service
from app.services.training_service import pydantic_to_sdk_config
from app.utils.paths import PathTraversalError, resolve_safe_path, safe_checkpoint_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/peft", tags=["PEFT"])


def _row_to_response(row: dict) -> JobResponse:  # type: ignore[type-arg]
    from app.schemas.common import JobStatus  # noqa: PLC0415
    return JobResponse(
        id=row["id"],
        type=JobType(row["type"]),
        status=JobStatus(row["status"]),
        payload=row.get("payload") or {},
        error=row.get("error"),
        progress=row.get("progress"),
        log_path=row.get("log_path"),
        pid=row.get("pid"),
        num_gpus=row.get("num_gpus", 1),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
    )


def _get_settings():  # type: ignore[return]
    from app import config as _cfg  # noqa: PLC0415
    return _cfg.settings


def _validate_dataset_path(dataset_path: str) -> str:
    settings = _get_settings()
    try:
        resolve_safe_path(settings.data_dir, dataset_path)
    except (PathTraversalError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dataset_path


def _validate_output_dir(output_dir: str) -> str:
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
    items, _ = await checkpoint_service.list_checkpoints(db)
    found = any(r["name"] == checkpoint_name for r in items)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint {checkpoint_name!r} not found",
        )


# ---------------------------------------------------------------------------
# GET /v1/peft
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=JobListResponse,
    summary="List LoRA and DoRA jobs",
)
async def list_peft_jobs(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: aiosqlite.Connection = Depends(get_db),
) -> JobListResponse:
    items, total = await job_service.list_jobs(
        db,
        job_types=[JobType.LORA, JobType.DORA],
        limit=limit,
        offset=offset,
    )
    return JobListResponse(
        items=[_row_to_response(r) for r in items],
        total=total,
    )


# ---------------------------------------------------------------------------
# POST /v1/peft/lora
# ---------------------------------------------------------------------------


@router.post(
    "/lora",
    status_code=202,
    response_model=JobCreatedResponse,
    summary="Submit a LoRA finetuning job",
)
async def lora_finetune(
    request: LoRARequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> JobCreatedResponse:
    if request.config.lora is None:
        raise HTTPException(
            status_code=422,
            detail="config.lora must be provided for LoRA jobs",
        )

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
        job_type=JobType.LORA,
        payload=payload,
        num_gpus=request.num_gpus,
    )
    logger.info("LoRA job %s queued (gpus=%d)", job_id, request.num_gpus)
    return JobCreatedResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# POST /v1/peft/dora
# ---------------------------------------------------------------------------


@router.post(
    "/dora",
    status_code=202,
    response_model=JobCreatedResponse,
    summary="Submit a DoRA finetuning job",
)
async def dora_finetune(
    request: DoRARequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> JobCreatedResponse:
    if request.config.dora is None:
        raise HTTPException(
            status_code=422,
            detail="config.dora must be provided for DoRA jobs",
        )

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
        job_type=JobType.DORA,
        payload=payload,
        num_gpus=request.num_gpus,
    )
    logger.info("DoRA job %s queued (gpus=%d)", job_id, request.num_gpus)
    return JobCreatedResponse(job_id=job_id)
