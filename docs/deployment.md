# Deployment Guide

This document covers how to run MegatronBridge API in different environments — from a quick Colab smoke test to a production bare-metal DGX node.

---

## Environments

| Environment | GPUs | Use case |
|---|---|---|
| [Google Colab H100](#google-colab-h100) | 1× H100 80GB | Quick smoke test, demos |
| [Bare metal / DGX](#bare-metal--dgx) | 4–8× H100/A100 | Production training |
| [Cloud GPU instance](#cloud-gpu-instance-awsgcplambda) | 1–8× H100/A100 | Production training |
| [Docker Compose (GPU node)](#docker-compose-gpu-node) | Any | Recommended for all deployments |

---

## Google Colab H100

### Verified environment (April 2026)

```
Python:  3.12.13
PyTorch: 2.10.0+cu128
CUDA:    12.8
GPU:     NVIDIA H100 80GB HBM3
Driver:  580.82.07
```

All three requirements met: Python ≥ 3.12 ✅, PyTorch ≥ 2.7 ✅, CUDA ≥ 12.8 ✅.

### Limitations vs production

- **1 GPU only** — Colab gives a single GPU per session. `num_gpus` must be set to `1` for all jobs.
- **No persistent storage** — files are wiped when the session ends. Use Google Drive mount or download checkpoints before the session expires.
- **Session timeout** — Colab disconnects idle sessions after ~90 minutes. Long training runs will be interrupted.
- **Port exposure** — Colab doesn't expose ports publicly. ngrok is required to reach the API from outside the notebook.

### Setup (step by step)

**Step 1 — Select H100 runtime**

Runtime → Change runtime type → Hardware accelerator: H100 → Save

**Step 2 — Verify environment**

```python
import sys, torch, subprocess

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
subprocess.run(["nvidia-smi"])
```

**Step 3 — Clone and install**

```bash
# Clone
!git clone https://github.com/Doondi-Ashlesh/Nvidia-megatron-bridge-API
%cd Nvidia-megatron-bridge-API

# Install base API dependencies (no GPU deps needed for the server itself)
!pip install -e ".[dev]" -q

# Install megatron-bridge (requires --no-build-isolation for TransformerEngine)
# This takes ~8-12 minutes on first install
!pip install setuptools pybind11 wheel_stub -q
!pip install --no-build-isolation megatron-bridge -q
```

**Step 4 — Configure environment**

```python
import os

os.environ["CHECKPOINTS_ROOT"] = "/content/checkpoints"
os.environ["LOGS_ROOT"]        = "/content/logs"
os.environ["DATABASE_URL"]     = "sqlite+aiosqlite:////content/megatronbridge.db"

# Create directories
!mkdir -p /content/checkpoints /content/logs
```

**Step 5 — Start the API server**

```python
import subprocess, time

server = subprocess.Popen(
    ["python", "-m", "uvicorn", "app.main:app",
     "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

time.sleep(3)  # wait for startup

import urllib.request
resp = urllib.request.urlopen("http://localhost:8000/health")
print("Health:", resp.read().decode())
# → {"status": "ok"}
```

**Step 6 — Expose with ngrok**

```bash
!pip install pyngrok -q
```

```python
from pyngrok import ngrok

# If you have an ngrok account (free), set your auth token:
# ngrok.set_auth_token("your_ngrok_token")

tunnel = ngrok.connect(8000)
print("Public URL:", tunnel.public_url)
# → https://xxxx-xx-xx.ngrok-free.app
```

You can now hit the API from anywhere using the ngrok URL.

**Step 7 — Submit a test job**

```python
import requests, json

BASE = "http://localhost:8000"  # or use tunnel.public_url from outside

# Import a small public HuggingFace model (no token required)
resp = requests.post(f"{BASE}/v1/checkpoints/import", json={
    "source_path": "facebook/opt-125m",
    "target_name": "opt-125m-megatron",
})
print(resp.json())
# → {"job_id": "...", "status": "queued"}

job_id = resp.json()["job_id"]
```

**Step 8 — Stream logs via WebSocket**

```python
import websocket, threading

def stream_logs(job_id):
    ws = websocket.WebSocket()
    ws.connect(f"ws://localhost:8000/v1/ws/jobs/{job_id}/logs")
    while True:
        msg = ws.recv()
        if not msg:
            break
        try:
            data = json.loads(msg)
            if data.get("type") == "stream_end":
                print(f"\n[Stream ended — status: {data['status']}]")
                break
        except json.JSONDecodeError:
            print(msg)  # plain log line
    ws.close()

t = threading.Thread(target=stream_logs, args=(job_id,), daemon=True)
t.start()
```

**Step 9 — Check progress**

```python
import time

while True:
    resp = requests.get(f"{BASE}/v1/jobs/{job_id}")
    job = resp.json()
    print(f"Status: {job['status']} | Progress: {job.get('progress')}")
    if job["status"] in ("completed", "failed", "cancelled"):
        break
    time.sleep(5)
```

### Open in Colab

A pre-built notebook with all cells above is available at:
`notebooks/colab_quickstart.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doondi-Ashlesh/Nvidia-megatron-bridge-API/blob/master/notebooks/colab_quickstart.ipynb)

---

## Bare Metal / DGX

### Recommended setup

```bash
# Clone
git clone https://github.com/Doondi-Ashlesh/Nvidia-megatron-bridge-API
cd Nvidia-megatron-bridge-API

# Install API server deps (CPU environment)
pip install -e "."

# Configure
cp .env.example .env
# Edit .env:
#   CHECKPOINTS_ROOT=/data/checkpoints
#   LOGS_ROOT=/data/logs
#   DATABASE_URL=sqlite+aiosqlite:////data/megatronbridge.db
#   API_KEY=your-secret-key   # optional but recommended

# Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The GPU worker is launched automatically by the dispatcher when a job is submitted. It does not need to be started separately. The dispatcher calls `torchrun` with `--nproc-per-node=<num_gpus>` per job.

### Install megatron-bridge on the worker environment

```bash
# Must be done in the Python environment where the worker runs
pip install torch>=2.7.0 setuptools pybind11 wheel_stub
pip install --no-build-isolation megatron-bridge
```

### Multi-GPU job example (DGX H100, 8 GPUs)

```bash
curl -X POST http://localhost:8000/v1/peft/lora \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "pretrained_checkpoint": "/data/checkpoints/llama3-megatron",
    "dataset_path": "/data/datasets/my_data",
    "output_dir": "/data/checkpoints/llama3-lora",
    "num_gpus": 8,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["linear_qkv", "linear_proj"]
  }'
```

The dispatcher will run:
```
torchrun --nproc-per-node=8 --master-port=<free-port> app/worker/executor.py --job-id <uuid>
```

All 8 H100s are used in parallel via NCCL over NVLink.

### Running as a systemd service

```ini
# /etc/systemd/system/megatronbridge-api.service
[Unit]
Description=MegatronBridge API Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/megatronbridge-api
EnvironmentFile=/opt/megatronbridge-api/.env
ExecStart=/opt/megatronbridge-api/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable megatronbridge-api
sudo systemctl start megatronbridge-api
```

---

## Cloud GPU Instance (AWS/GCP/Lambda)

### Recommended instance types

| Provider | Instance | GPUs | Memory |
|---|---|---|---|
| AWS | `p5.48xlarge` | 8× H100 80GB | 640 GB GPU |
| AWS | `p4d.24xlarge` | 8× A100 40GB | 320 GB GPU |
| GCP | `a3-highgpu-8g` | 8× H100 80GB | 640 GB GPU |
| Lambda Labs | `gpu_8x_h100_sxm5` | 8× H100 80GB | 640 GB GPU |

### Setup

Same as bare metal. Use an NGC base AMI/image if available — CUDA, PyTorch, and TransformerEngine are pre-installed, saving the `--no-build-isolation` build time.

**AWS Deep Learning AMI (recommended):**
```bash
# CUDA 12.x, PyTorch 2.x pre-installed
# Just install megatron-bridge on top:
pip install --no-build-isolation megatron-bridge
```

### Firewall

Open port 8000 inbound if you want to reach the API from outside the instance. Or use an Application Load Balancer in front for TLS termination.

---

## Docker Compose (GPU Node)

The recommended approach for any GPU deployment — consistent environment, easy updates.

```bash
# Clone
git clone https://github.com/Doondi-Ashlesh/Nvidia-megatron-bridge-API
cd Nvidia-megatron-bridge-API

# Configure
cp .env.example .env
# Edit .env as needed

# Start API server only (no GPU required for this)
docker compose up -d api

# Start with GPU worker (requires nvidia-container-toolkit)
docker compose --profile gpu up -d

# Check health
curl http://localhost:8000/health

# View logs
docker compose logs -f api
docker compose logs -f worker
```

### Prerequisites for GPU worker

```bash
# Install nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///megatronbridge.db` | SQLite file path |
| `CHECKPOINTS_ROOT` | `/data/checkpoints` | Root for all checkpoint paths |
| `LOGS_ROOT` | `/data/logs` | Root for job log files |
| `API_KEY` | *(empty — disabled)* | Enable Bearer token auth when set |
| `RATE_LIMIT_REQUESTS` | `60` | Write requests per minute per IP |
| `MAX_QUEUED_JOBS` | `100` | Max queued jobs before HTTP 429 |
| `CORS_ORIGINS` | *(empty — disabled)* | Comma-separated allowed origins |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Verifying a Deployment

```bash
# 1. Liveness
curl http://localhost:8000/health
# → {"status": "ok"}

# 2. Readiness (DB + worker check)
curl http://localhost:8000/health/ready
# → {"status": "ready", "database": "ok"}

# 3. System info (GPU details, CUDA version)
curl http://localhost:8000/v1/system/info

# 4. Submit a minimal test job
curl -X POST http://localhost:8000/v1/checkpoints/import \
  -H "Content-Type: application/json" \
  -d '{"source_path": "facebook/opt-125m", "target_name": "opt-125m-test"}'
# → {"job_id": "...", "status": "queued"}
```
