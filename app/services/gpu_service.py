"""GPU telemetry service using pynvml.

All public functions gracefully degrade (return empty/null) when pynvml is
unavailable or raises any exception.
"""

from __future__ import annotations

from typing import Any

# Known GPU peak TFLOPS for MFU calculation (bfloat16)
# Source: Nvidia official specs
GPU_PEAK_TFLOPS: dict[str, float] = {
    "NVIDIA H100 SXM5 80GB": 989.5,
    "NVIDIA H100 SXM 80GB": 989.5,
    "NVIDIA H100 PCIe 80GB": 756.0,
    "NVIDIA A100 SXM4 80GB": 312.0,
    "NVIDIA A100 SXM4 40GB": 312.0,
    "NVIDIA A100 PCIe 80GB": 312.0,
    "NVIDIA A100 PCIe 40GB": 312.0,
    "NVIDIA A10G": 125.0,
    "NVIDIA RTX 6000 Ada Generation": 91.1,
    # Blackwell (GB202) — RTX PRO 6000 Server Edition, 95.6 GB GDDR7
    # Source: Nvidia RTX PRO 6000 Blackwell datasheet (BF16 Tensor Core, dense)
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 838.0,
    "NVIDIA RTX 4090": 82.6,
    "NVIDIA RTX 3090": 35.6,
}


def _get_pynvml():  # type: ignore[return]
    """Import pynvml at call time so the mock in sys.modules is always used."""
    import sys

    return sys.modules.get("pynvml") or __import__("pynvml")


def get_gpu_count() -> int:
    """Return number of GPUs. Returns 0 on any error."""
    try:
        pynvml = _get_pynvml()
        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return 0


def get_all_gpu_info() -> list[dict[str, Any]]:
    """Return telemetry for all GPUs in a single nvmlInit/Shutdown cycle.

    Returns [] on any error.
    """
    try:
        pynvml = _get_pynvml()
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = (raw_name.decode("utf-8") if isinstance(raw_name, bytes)
                        else str(raw_name)).rstrip("\x00")

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, 0)

                gpus.append({
                    "id": i,
                    "name": name,
                    "util_pct": util.gpu,
                    "mem_used_gb": round(mem.used / 1024**3, 2),
                    "mem_total_gb": round(mem.total / 1024**3, 2),
                    "temp_c": temp,
                    "peak_tflops": GPU_PEAK_TFLOPS.get(name),
                })
            return gpus
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return []


def get_gpu_info(index: int) -> dict[str, Any]:
    """Return telemetry dict for a single GPU, or empty dict on any error.

    Prefer ``get_all_gpu_info()`` when querying multiple GPUs to avoid
    repeated nvmlInit/Shutdown overhead.
    """
    result = get_all_gpu_info()
    return next((g for g in result if g.get("id") == index), {})


def get_driver_info() -> dict[str, str | None]:
    """Return driver and CUDA version strings, or None values on any error."""
    try:
        pynvml = _get_pynvml()
        pynvml.nvmlInit()
        try:
            raw_driver = pynvml.nvmlSystemGetDriverVersion()
            driver_version: str | None = (
                raw_driver.decode("utf-8") if isinstance(raw_driver, bytes)
                else str(raw_driver)
            ).rstrip("\x00")

            cuda_int = pynvml.nvmlSystemGetCudaDriverVersion()
            major = cuda_int // 1000
            minor = (cuda_int % 1000) // 10
            cuda_version: str | None = f"{major}.{minor}"
        finally:
            pynvml.nvmlShutdown()

        return {"driver_version": driver_version, "cuda_version": cuda_version}
    except Exception:
        return {"driver_version": None, "cuda_version": None}


def calculate_mfu(
    *,
    tokens_per_sec: float,
    num_layers: int,
    hidden_size: int,
    num_gpus: int,
    gpu_name: str,
) -> float | None:
    """Calculate Model FLOPs Utilization.

    Formula (from Chinchilla/PaLM papers):
        model_flops_per_token = 6 * num_layers * hidden_size^2
        mfu = (tokens_per_sec * model_flops_per_token) / (num_gpus * gpu_peak_flops)

    Returns None if gpu_name not in GPU_PEAK_TFLOPS or result > 1.0 (clamp guard).
    """
    peak_tflops = GPU_PEAK_TFLOPS.get(gpu_name)
    if peak_tflops is None:
        return None

    model_flops_per_token = 6 * num_layers * (hidden_size**2)
    gpu_peak_flops = num_gpus * peak_tflops * 1e12

    mfu = (tokens_per_sec * model_flops_per_token) / gpu_peak_flops

    if mfu > 1.0:
        return None

    return mfu
