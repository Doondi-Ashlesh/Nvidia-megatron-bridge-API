"""System information endpoint.

Returns GPU telemetry, driver info, supported models, and package version.
Never exposes hostname, internal IPs, or filesystem paths.
Always returns 200 — gracefully degrades on pynvml failure.
"""

from __future__ import annotations

import importlib.metadata

from fastapi import APIRouter

from app.services.gpu_service import get_all_gpu_info, get_driver_info

router = APIRouter(prefix="/v1/system", tags=["System"])

SUPPORTED_MODELS: list[str] = [
    "llama2",
    "llama3",
    "llama3_1",
    "qwen3",
    "deepseek_v3",
    "mistral",
    "gemma2",
    "phi3",
    "falcon",
    "starcoder2",
]


@router.get(
    "/info",
    summary="System information",
    description=(
        "Returns GPU telemetry, driver versions, supported models, and package version. "
        "Always returns 200. Gracefully degrades when pynvml is unavailable."
    ),
)
async def get_system_info() -> dict:
    """Return system information. Never raises — always 200."""
    try:
        version = importlib.metadata.version("megatron-bridge")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"

    driver_info = get_driver_info()
    devices = get_all_gpu_info()

    return {
        "megatron_bridge_version": version,
        "cuda_version": driver_info.get("cuda_version"),
        "driver_version": driver_info.get("driver_version"),
        "devices": devices,
        "supported_models": SUPPORTED_MODELS,
    }
