"""Unit tests for app.services.gpu_service.

The mock_pynvml fixture is autouse=True in conftest.py, so all tests here
run against the fake pynvml without any GPU hardware.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from app.services.gpu_service import (
    GPU_PEAK_TFLOPS,
    calculate_mfu,
    get_all_gpu_info,
    get_driver_info,
    get_gpu_count,
    get_gpu_info,
)


# ---------------------------------------------------------------------------
# get_gpu_count
# ---------------------------------------------------------------------------


def test_get_gpu_count_returns_two(mock_pynvml):
    assert get_gpu_count() == 2


def test_get_gpu_count_returns_zero_on_error(mock_pynvml):
    mock_pynvml.nvmlInit.side_effect = Exception("nvml init failed")
    assert get_gpu_count() == 0


# ---------------------------------------------------------------------------
# get_gpu_info
# ---------------------------------------------------------------------------


def test_get_gpu_info_returns_dict_with_correct_keys(mock_pynvml):
    info = get_gpu_info(0)
    expected_keys = {"id", "name", "util_pct", "mem_used_gb", "mem_total_gb", "temp_c", "peak_tflops"}
    assert expected_keys == set(info.keys())


def test_get_gpu_info_id_matches_index(mock_pynvml):
    info = get_gpu_info(0)
    assert info["id"] == 0


def test_get_gpu_info_name_decoded_from_bytes(mock_pynvml):
    # mock returns b"NVIDIA H100 SXM5 80GB"
    info = get_gpu_info(0)
    assert info["name"] == "NVIDIA H100 SXM5 80GB"
    assert isinstance(info["name"], str)


def test_get_gpu_info_mem_used_gb_correct(mock_pynvml):
    info = get_gpu_info(0)
    # 48 * 1024^3 bytes / 1024^3 == 48.0 GB
    assert info["mem_used_gb"] == pytest.approx(48.0)


def test_get_gpu_info_mem_total_gb_correct(mock_pynvml):
    info = get_gpu_info(0)
    # 80 * 1024^3 bytes / 1024^3 == 80.0 GB
    assert info["mem_total_gb"] == pytest.approx(80.0)


def test_get_gpu_info_peak_tflops_h100(mock_pynvml):
    info = get_gpu_info(0)
    assert info["peak_tflops"] == pytest.approx(989.5)


def test_get_gpu_info_util_pct(mock_pynvml):
    info = get_gpu_info(0)
    assert info["util_pct"] == 85


def test_get_gpu_info_temp_c(mock_pynvml):
    info = get_gpu_info(0)
    assert info["temp_c"] == 72


def test_get_gpu_info_returns_empty_dict_on_pynvml_error(mock_pynvml):
    mock_pynvml.nvmlInit.side_effect = Exception("hardware failure")
    info = get_gpu_info(0)
    assert info == {}


def test_get_gpu_info_peak_tflops_none_for_unknown_gpu(mock_pynvml):
    mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA Unknown GPU 999GB"
    info = get_gpu_info(0)
    assert info["peak_tflops"] is None


# ---------------------------------------------------------------------------
# get_all_gpu_info
# ---------------------------------------------------------------------------


def test_get_all_gpu_info_returns_list_of_two(mock_pynvml):
    infos = get_all_gpu_info()
    assert isinstance(infos, list)
    assert len(infos) == 2


def test_get_all_gpu_info_each_item_has_required_keys(mock_pynvml):
    infos = get_all_gpu_info()
    for item in infos:
        assert "id" in item
        assert "name" in item


def test_get_all_gpu_info_returns_empty_list_when_get_gpu_count_raises(mock_pynvml):
    mock_pynvml.nvmlDeviceGetCount.side_effect = Exception("count failed")
    result = get_all_gpu_info()
    assert result == []


# ---------------------------------------------------------------------------
# get_driver_info
# ---------------------------------------------------------------------------


def test_get_driver_info_returns_driver_version(mock_pynvml):
    info = get_driver_info()
    assert info["driver_version"] == "550.90.07"


def test_get_driver_info_returns_cuda_version(mock_pynvml):
    # nvmlSystemGetCudaDriverVersion returns 12040 -> "12.4"
    info = get_driver_info()
    assert info["cuda_version"] == "12.4"


def test_get_driver_info_returns_none_values_on_pynvml_error(mock_pynvml):
    mock_pynvml.nvmlInit.side_effect = Exception("nvml error")
    info = get_driver_info()
    assert info == {"driver_version": None, "cuda_version": None}


def test_get_driver_info_driver_version_is_string(mock_pynvml):
    info = get_driver_info()
    assert isinstance(info["driver_version"], str)


# ---------------------------------------------------------------------------
# calculate_mfu
# ---------------------------------------------------------------------------


def test_calculate_mfu_known_inputs():
    """
    tokens_per_sec = 142000
    num_layers = 32, hidden_size = 4096, num_gpus = 2
    gpu_name = "NVIDIA H100 SXM5 80GB" (peak = 989.5 TFLOPS)

    model_flops_per_token = 6 * 32 * 4096^2 = 3,221,225,472
    gpu_peak_flops = 2 * 989.5e12 = 1.979e15
    mfu ≈ 0.231
    """
    result = calculate_mfu(
        tokens_per_sec=142000,
        num_layers=32,
        hidden_size=4096,
        num_gpus=2,
        gpu_name="NVIDIA H100 SXM5 80GB",
    )
    assert result is not None
    assert result == pytest.approx(0.231, abs=0.001)


def test_calculate_mfu_returns_none_for_unknown_gpu():
    result = calculate_mfu(
        tokens_per_sec=100000,
        num_layers=32,
        hidden_size=4096,
        num_gpus=1,
        gpu_name="NVIDIA Unknown GPU",
    )
    assert result is None


def test_calculate_mfu_returns_none_if_result_exceeds_one():
    # Provide unrealistically high tokens_per_sec to exceed MFU > 1.0
    result = calculate_mfu(
        tokens_per_sec=1_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_gpus=1,
        gpu_name="NVIDIA H100 SXM5 80GB",
    )
    assert result is None


def test_calculate_mfu_returns_float_below_one_for_reasonable_inputs():
    result = calculate_mfu(
        tokens_per_sec=50000,
        num_layers=32,
        hidden_size=4096,
        num_gpus=8,
        gpu_name="NVIDIA H100 SXM5 80GB",
    )
    assert result is not None
    assert 0.0 < result <= 1.0


def test_calculate_mfu_a100_gpu():
    result = calculate_mfu(
        tokens_per_sec=80000,
        num_layers=32,
        hidden_size=4096,
        num_gpus=4,
        gpu_name="NVIDIA A100 SXM4 80GB",
    )
    assert result is not None
    assert 0.0 < result <= 1.0
