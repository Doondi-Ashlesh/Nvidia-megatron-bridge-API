"""REST endpoints for checkpoint management: /v1/checkpoints."""

from __future__ import annotations

import logging

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from app.database import get_db
from app.schemas.checkpoint import (
    CheckpointFormat,
    CheckpointListResponse,
    CheckpointRecord,
    ExportRequest,
    ImportRequest,
)
from app.schemas.common import JobCreatedResponse, JobType
from app.services import checkpoint_service, job_service
from app.utils.paths import PathTraversalError, safe_checkpoint_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/checkpoints", tags=["Checkpoints"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_record(row: dict) -> CheckpointRecord:  # type: ignore[type-arg]
    return CheckpointRecord(
        id=row["id"],
        name=row["name"],
        format=CheckpointFormat(row["format"]),
        path=row["path"],
        size_bytes=row.get("size_bytes"),
        model_arch=row.get("model_arch"),
        created_from_job=row.get("created_from_job"),
        created_at=row["created_at"],
    )


def _assert_checkpoint_exists(
    row: dict | None,  # type: ignore[type-arg]
    checkpoint_id: str,
) -> dict:  # type: ignore[type-arg]
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"Checkpoint {checkpoint_id!r} not found"
        )
    return row


def _get_checkpoints_root():  # type: ignore[return]
    from app import config as _cfg  # noqa: PLC0415
    return _cfg.settings.checkpoints_root


# ---------------------------------------------------------------------------
# POST /v1/checkpoints/import
# ---------------------------------------------------------------------------


@router.post(
    "/import",
    status_code=202,
    response_model=JobCreatedResponse,
    summary="Import a HuggingFace checkpoint into Megatron format",
)
async def import_checkpoint(
    req: ImportRequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> JobCreatedResponse:
    checkpoints_root = _get_checkpoints_root()

    # Resolve target path through the security boundary
    try:
        target_path = safe_checkpoint_path(checkpoints_root, req.target_name)
    except (PathTraversalError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Build payload — hf_token is included so the executor can set HF_TOKEN env var.
    # token_filter.py ensures it is scrubbed from all log output.
    payload: dict = {  # type: ignore[type-arg]
        "source_path": req.source_path,
        "target_path": str(target_path),
        "num_gpus": req.num_gpus,
    }
    if req.hf_token is not None:
        payload["hf_token"] = req.hf_token
    if req.model_arch is not None:
        payload["model_arch"] = req.model_arch

    job_id = await job_service.create_job(
        db,
        job_type=JobType.CHECKPOINT_IMPORT,
        payload=payload,
        num_gpus=req.num_gpus,
    )
    # Log without the token (token_filter covers hf_token in payload, but we
    # emit a safe message here for clarity)
    logger.info(
        "Checkpoint import job %s queued (source=%s, target=%s)",
        job_id,
        req.source_path,
        target_path,
    )
    return JobCreatedResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# POST /v1/checkpoints/export
# ---------------------------------------------------------------------------


@router.post(
    "/export",
    status_code=202,
    response_model=JobCreatedResponse,
    summary="Export a Megatron checkpoint to HuggingFace format",
)
async def export_checkpoint(
    req: ExportRequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> JobCreatedResponse:
    checkpoints_root = _get_checkpoints_root()

    # Resolve source and target paths through the security boundary
    try:
        source_path = safe_checkpoint_path(checkpoints_root, req.source_name)
        target_path = safe_checkpoint_path(checkpoints_root, req.target_name)
    except (PathTraversalError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Verify source checkpoint exists in the DB
    items, _ = await checkpoint_service.list_checkpoints(db)
    source_record = next(
        (r for r in items if r["path"] == str(source_path)), None
    )
    if source_record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Source checkpoint {req.source_name!r} not found",
        )

    payload = {
        "source_path": str(source_path),
        "target_path": str(target_path),
        "model_arch": req.model_arch,
        "num_gpus": req.num_gpus,
    }

    job_id = await job_service.create_job(
        db,
        job_type=JobType.CHECKPOINT_EXPORT,
        payload=payload,
        num_gpus=req.num_gpus,
    )
    logger.info(
        "Checkpoint export job %s queued (source=%s, target=%s)",
        job_id,
        source_path,
        target_path,
    )
    return JobCreatedResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# GET /v1/checkpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=CheckpointListResponse,
    summary="List all checkpoints",
)
async def list_checkpoints(
    fmt: CheckpointFormat | None = Query(default=None, description="Filter by format"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: aiosqlite.Connection = Depends(get_db),
) -> CheckpointListResponse:
    items, total = await checkpoint_service.list_checkpoints(
        db,
        fmt=fmt.value if fmt is not None else None,
        limit=limit,
        offset=offset,
    )
    return CheckpointListResponse(
        items=[_row_to_record(r) for r in items],
        total=total,
    )


# ---------------------------------------------------------------------------
# GET /v1/checkpoints/{checkpoint_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{checkpoint_id}",
    response_model=CheckpointRecord,
    summary="Get a single checkpoint",
    responses={404: {"description": "Checkpoint not found"}},
)
async def get_checkpoint(
    checkpoint_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> CheckpointRecord:
    row = _assert_checkpoint_exists(
        await checkpoint_service.get_checkpoint(db, checkpoint_id),
        checkpoint_id,
    )
    return _row_to_record(row)


# ---------------------------------------------------------------------------
# DELETE /v1/checkpoints/{checkpoint_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{checkpoint_id}",
    status_code=204,
    summary="Delete a checkpoint record (DB only; files are not removed)",
    responses={404: {"description": "Checkpoint not found"}},
)
async def delete_checkpoint(
    checkpoint_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> Response:
    deleted = await checkpoint_service.delete_checkpoint(db, checkpoint_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint {checkpoint_id!r} not found",
        )
    return Response(status_code=204)
