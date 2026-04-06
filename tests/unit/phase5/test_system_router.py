"""HTTP integration tests for GET /v1/system/info."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Basic response shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_info_returns_200(client):
    resp = await client.get("/v1/system/info")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_system_info_contains_required_fields(client):
    resp = await client.get("/v1/system/info")
    data = resp.json()
    assert "devices" in data
    assert "cuda_version" in data
    assert "driver_version" in data
    assert "supported_models" in data
    assert "megatron_bridge_version" in data


@pytest.mark.asyncio
async def test_system_info_devices_has_two_items(client, mock_pynvml):
    # mock_pynvml returns count=2
    resp = await client.get("/v1/system/info")
    data = resp.json()
    assert len(data["devices"]) == 2


@pytest.mark.asyncio
async def test_system_info_each_device_has_correct_keys(client, mock_pynvml):
    resp = await client.get("/v1/system/info")
    data = resp.json()
    for device in data["devices"]:
        assert "id" in device
        assert "name" in device
        assert "util_pct" in device
        assert "mem_used_gb" in device
        assert "mem_total_gb" in device
        assert "temp_c" in device


@pytest.mark.asyncio
async def test_system_info_cuda_version_from_mock(client, mock_pynvml):
    resp = await client.get("/v1/system/info")
    data = resp.json()
    assert data["cuda_version"] == "12.4"


@pytest.mark.asyncio
async def test_system_info_driver_version_from_mock(client, mock_pynvml):
    resp = await client.get("/v1/system/info")
    data = resp.json()
    assert data["driver_version"] == "550.90.07"


@pytest.mark.asyncio
async def test_system_info_supported_models_is_non_empty_list(client):
    resp = await client.get("/v1/system/info")
    data = resp.json()
    models = data["supported_models"]
    assert isinstance(models, list)
    assert len(models) > 0
    for model in models:
        assert isinstance(model, str)


@pytest.mark.asyncio
async def test_system_info_megatron_bridge_version_is_string(client):
    resp = await client.get("/v1/system/info")
    data = resp.json()
    assert isinstance(data["megatron_bridge_version"], str)
    # Either a real semver or "unknown" when package not installed
    assert len(data["megatron_bridge_version"]) > 0


# ---------------------------------------------------------------------------
# Security: must NOT expose hostname, internal IPs, or filesystem paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_info_does_not_expose_hostname(client):
    import socket

    resp = await client.get("/v1/system/info")
    body = resp.text
    hostname = socket.gethostname()
    assert hostname not in body


@pytest.mark.asyncio
async def test_system_info_does_not_expose_filesystem_paths(client):
    resp = await client.get("/v1/system/info")
    body = resp.text
    # No slash-based Unix paths or Windows drive letters in the response
    assert "/tmp" not in body
    assert "/home" not in body
    assert "/var" not in body
    assert "C:\\" not in body


@pytest.mark.asyncio
async def test_system_info_does_not_expose_internal_ips(client):
    resp = await client.get("/v1/system/info")
    body = resp.text
    # Common internal IP prefixes should not appear
    assert "192.168." not in body
    assert "10.0." not in body
    assert "172.16." not in body


# ---------------------------------------------------------------------------
# Graceful degradation when pynvml fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_info_returns_200_when_pynvml_fails(client, mock_pynvml, monkeypatch):
    """Even with pynvml broken, the endpoint must return 200."""
    mock_pynvml.nvmlInit.side_effect = Exception("no GPU hardware")
    mock_pynvml.nvmlDeviceGetCount.side_effect = Exception("no GPU hardware")
    mock_pynvml.nvmlSystemGetDriverVersion.side_effect = Exception("no GPU hardware")
    mock_pynvml.nvmlSystemGetCudaDriverVersion.side_effect = Exception("no GPU hardware")

    resp = await client.get("/v1/system/info")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_system_info_graceful_degradation_returns_empty_devices(client, mock_pynvml):
    """When pynvml is broken, devices should be [] and versions should be null."""
    mock_pynvml.nvmlInit.side_effect = Exception("no GPU hardware")
    mock_pynvml.nvmlDeviceGetCount.side_effect = Exception("no GPU hardware")

    resp = await client.get("/v1/system/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["devices"] == []
    assert data["cuda_version"] is None
    assert data["driver_version"] is None
