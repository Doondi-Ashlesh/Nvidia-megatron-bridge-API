"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.config import settings
from app.database import check_db_health

router = APIRouter()


@router.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 when the API process is alive. "
    "Returns 503 if the database is unreachable.",
    responses={
        200: {"description": "API is healthy"},
        503: {"description": "Database is unreachable"},
    },
)
async def health(response: Response) -> dict:
    db_ok = await check_db_health()
    db_status = "connected" if db_ok else "error"

    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if db_ok else "degraded",
        "version": settings.version,
        "db_status": db_status,
    }


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Returns 200 when the API is ready to serve traffic "
    "(database reachable). Suitable for Kubernetes readinessProbe.",
    responses={
        200: {"description": "API is ready"},
        503: {"description": "API is not yet ready"},
    },
)
async def ready(response: Response) -> dict:
    db_ok = await check_db_health()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "database_unavailable"}
    return {"status": "ready"}
