"""Aggregate all versioned API routers under the /v1 prefix."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import checkpoints, health, jobs, peft, system, training

router = APIRouter()

# System / health (no version prefix — standard convention)
router.include_router(health.router, tags=["System"])

# Phase 2: Job infrastructure
router.include_router(jobs.router)

# Phase 3: Checkpoint management
router.include_router(checkpoints.router)

# Phase 4: Training and PEFT
router.include_router(training.router)
router.include_router(peft.router)

# Phase 5: System info and GPU telemetry
router.include_router(system.router)
