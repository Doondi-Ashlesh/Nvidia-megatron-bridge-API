# MegatronBridge API

An open-source **REST + WebSocket API** wrapper around [Nvidia's MegatronBridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) Python SDK, making large-scale LLM pretraining, fine-tuning, and checkpoint conversion accessible from any language, tool, or LLM agent over HTTP.

[![PR Checks](https://github.com/Doondi-Ashlesh/Nvidia-megatron-bridge-API/actions/workflows/pr.yml/badge.svg)](https://github.com/Doondi-Ashlesh/Nvidia-megatron-bridge-API/actions/workflows/pr.yml)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)](https://github.com/Doondi-Ashlesh/Nvidia-megatron-bridge-API)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Doondi-Ashlesh/Nvidia-megatron-bridge-API/blob/master/notebooks/colab_quickstart.ipynb)

---

## What it does

[MegatronBridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) is Nvidia's production Python library for:

- **Bidirectional checkpoint conversion** — HuggingFace ↔ Megatron-Core format
- **Pretraining** at scale on multi-GPU nodes (via torchrun)
- **Supervised fine-tuning (SFT)**
- **LoRA / DoRA PEFT** fine-tuning

No HTTP API exists for it anywhere. This project fills that gap — wrapping the SDK in a production-grade FastAPI server so any language, tool, or LLM agent can trigger and monitor GPU training jobs over standard HTTP and WebSocket.

---

## Target Use Case

This project is designed for a **single node with multiple GPUs** — the most common serious LLM training setup:

- A bare-metal **DGX H100** (8× H100 80GB SXM5)
- A cloud GPU instance: `p5.48xlarge` (8× H100), `p4d.24xlarge` (8× A100)
- An on-prem cluster where one node is allocated per job

The API server runs on the CPU side of the machine. The GPU worker is launched by `torchrun` and uses all GPUs on the node in parallel. Everything — the API, the database, the log files, the worker — lives on the same machine.

This is the right architecture for 7B–70B model fine-tuning. Spreading a single job across multiple nodes over InfiniBand (true multi-node) is in the v0.2 roadmap.

---

## Architecture

```
One physical machine (or cloud GPU instance)
│
├── FastAPI server (CPU cores)          GPU worker subprocess
│   ├── HTTP/WebSocket endpoints  ◄──   torchrun --nproc-per-node=N
│   ├── SQLite job queue (WAL)    ◄──   reads/writes same DB file
│   ├── dispatcher loop           ───►  launches worker per job
│   └── WebSocket log/progress    ◄──   tails /data/logs/<job_id>.log
│
└── /data/                              (shared volume)
    ├── megatronbridge.db               SQLite file
    ├── logs/<job_id>.log               worker stdout
    └── checkpoints/                    HF and Megatron checkpoint dirs
```

### Two-process split: why and how

The API server and the GPU worker are deliberately separate processes:

**API server (CPU-only)** — runs `uvicorn`, handles all HTTP and WebSocket traffic, reads/writes SQLite, streams log files. It never imports `megatron.bridge`, never touches CUDA. This means the API server starts instantly on any machine, even without a GPU.

**GPU worker** — launched by `torchrun` as a subprocess when a job is dequeued. `torchrun` spawns one process per GPU. Each process imports `megatron.bridge`, initialises its CUDA context, and joins the distributed process group via NCCL. The processes communicate over NVLink (on DGX hardware) or PCIe. All stdout from all ranks is captured to a single log file that the API server tails in real-time.

`app/worker/executor.py` is the **only file** in the entire codebase that imports `megatron.bridge`. This boundary is intentional and enforced — it means you can develop, test, and run the API server on a laptop with no GPU and no CUDA installation.

### Why SQLite instead of Redis or PostgreSQL

SQLite is a file, not a server. There is nothing to install, configure, or start. When the FastAPI app opens `megatronbridge.db`, it opens a file the same way you'd open a text file — no connection string, no daemon, no port.

This works correctly here because:

- **GPU is always the bottleneck.** You will never submit 1000 simultaneous jobs to a GPU node. Realistically, 1–2 jobs run at a time. SQLite handles hundreds of concurrent reads trivially.
- **Same machine.** The API server and the GPU worker share the same filesystem. SQLite's file-sharing model is designed for exactly this — one writer (worker updating job status) and many readers (API server polling for status).
- **WAL mode.** The database runs in Write-Ahead Logging mode, which allows the worker to write job status while the API server reads it simultaneously, with no blocking.
- **Zero infrastructure.** Anyone can clone this repo and run it without also installing and operating a Postgres server.

If you ever need multi-node (API server and workers on different machines), swap `DATABASE_URL` to a Postgres connection string — the rest of the code uses async abstractions that support both.

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| API server | Python 3.12+, any CPU machine |
| GPU worker | CUDA ≥ 12.8, PyTorch ≥ 2.7, megatron-bridge ≥ 0.3.1 |
| Recommended worker container | `nvcr.io/nvidia/nemo:25.02` |
| Supported GPUs | H100, A100, RTX 6000 Ada (MFU tracking); any CUDA GPU (training) |
| Container runtime (GPU) | `nvidia-container-toolkit` |

---

## Quick Start

> For full deployment instructions (Colab, bare metal DGX, AWS/GCP, Docker) see [`docs/deployment.md`](docs/deployment.md).

### CPU-only (API server, development)

```bash
git clone https://github.com/Doondi-Ashlesh/Nvidia-megatron-bridge-API
cd Nvidia-megatron-bridge-API

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env as needed

# Run tests (no GPU required)
python -m pytest tests/ -m "not gpu"

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Check
curl http://localhost:8000/health
# → {"status": "ok"}
```

### Docker Compose (recommended for GPU nodes)

```bash
cp .env.example .env
# Edit .env — set CHECKPOINTS_ROOT, LOGS_ROOT, optionally API_KEY

# Start API server only (CPU, no GPU required)
docker compose up -d api

# Check health
curl http://localhost:8000/health

# Start with GPU worker (requires nvidia-container-toolkit)
docker compose --profile gpu up -d
```

The GPU worker container is behind the `--profile gpu` flag so `docker compose up` works on any machine. The worker only starts when you explicitly opt in. This matters because the worker image is ~22 GB (based on the NeMo NGC image which pre-installs CUDA, TransformerEngine, and PyTorch).

---

## API Reference

### Checkpoints

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/v1/checkpoints/import` | HuggingFace → Megatron conversion | 202 + job_id |
| `POST` | `/v1/checkpoints/export` | Megatron → HuggingFace conversion | 202 + job_id |
| `GET`  | `/v1/checkpoints` | List all registered checkpoints | 200 |
| `GET`  | `/v1/checkpoints/{id}` | Checkpoint metadata | 200 |
| `DELETE` | `/v1/checkpoints/{id}` | Delete checkpoint files + DB record | 204 |

### Training

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/v1/training/pretrain` | Launch pretraining job | 202 + job_id |
| `POST` | `/v1/training/finetune` | Launch SFT fine-tuning job | 202 + job_id |
| `GET`  | `/v1/training` | List training jobs | 200 |

### PEFT

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `POST` | `/v1/peft/lora` | Launch LoRA fine-tuning | 202 + job_id |
| `POST` | `/v1/peft/dora` | Launch DoRA fine-tuning | 202 + job_id |
| `GET`  | `/v1/peft` | List LoRA/DoRA jobs | 200 |

### Jobs

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET`  | `/v1/jobs` | List all jobs (filterable by type/status) | 200 |
| `GET`  | `/v1/jobs/{id}` | Job status + progress + GPU metrics | 200 |
| `DELETE` | `/v1/jobs/{id}` | Cancel running job (SIGTERM process group) | 202 |
| `GET`  | `/v1/jobs/{id}/logs` | Fetch log file over HTTP (WebSocket fallback) | 200 |

### WebSocket Streaming

| Path | Description |
|------|-------------|
| `ws://host/v1/ws/jobs/{id}/logs` | Stream log lines in real-time |
| `ws://host/v1/ws/jobs/{id}/progress` | Stream progress frames every 2 s |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/health/ready` | Readiness: DB + worker alive |
| `GET` | `/v1/system/info` | CUDA version, GPU names, NCCL version, supported models |

---

## WebSocket Protocol

Both endpoints close with code `1008` for any invalid or unknown `job_id`. Authentication via `?api_key=` query parameter is supported (Bearer header is not available for WebSocket connections in most clients).

### Log stream (`/v1/ws/jobs/{id}/logs`)

On connect, all existing log lines are sent immediately (catch-up delivery), then new lines are polled every 200 ms as the worker writes them. When the job reaches a terminal state and no more new lines arrive, the server sends a final JSON frame and closes:

```json
{"type": "stream_end", "status": "completed"}
```

`status` can be `"completed"`, `"failed"`, or `"cancelled"`.

### Progress stream (`/v1/ws/jobs/{id}/progress`)

A JSON frame is sent every 2 seconds while the job is running:

```json
{
  "type": "progress",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": {
    "step": 450,
    "total_steps": 1000,
    "loss": 1.3421,
    "lr": 3e-5,
    "grad_norm": 0.812,
    "samples_per_sec": 25.6,
    "tflops": 42.3
  },
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA H100 SXM5 80GB",
      "util_pct": 94,
      "mem_used_gb": 38.2,
      "mem_total_gb": 80.0,
      "temp_c": 72
    },
    {
      "id": 1,
      "name": "NVIDIA H100 SXM5 80GB",
      "util_pct": 91,
      "mem_used_gb": 37.8,
      "mem_total_gb": 80.0,
      "temp_c": 71
    }
  ]
}
```

GPU telemetry is collected by the worker process (which has direct access to the hardware via pynvml) and written to the `jobs.progress` column in SQLite every 10 seconds. The API server reads it from there — it never calls pynvml itself. This is the correct data flow for a two-process architecture where only the worker has GPU access.

When the job finishes, the server sends a terminal frame and closes:

```json
{"type": "job_complete", "status": "completed"}
```

---

## Example: Import a HuggingFace Checkpoint

Convert a gated HuggingFace model to Megatron-Core format for training:

```bash
curl -X POST http://localhost:8000/v1/checkpoints/import \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "meta-llama/Llama-3.1-8B",
    "target_name": "llama31-8b-megatron",
    "hf_token": "hf_..."
  }'
# → {"job_id": "550e8400-e29b-41d4-a716-446655440000", "status": "queued"}

# Watch the conversion in real-time
wscat -c ws://localhost:8000/v1/ws/jobs/550e8400-.../logs

# Or poll status
curl http://localhost:8000/v1/jobs/550e8400-...
```

The `hf_token` is passed to the worker as an environment variable and is never written to logs, the database, or API responses.

---

## Example: Launch LoRA Fine-Tuning on 4 GPUs

```bash
curl -X POST http://localhost:8000/v1/peft/lora \
  -H "Content-Type: application/json" \
  -d '{
    "pretrained_checkpoint": "/data/checkpoints/llama31-8b-megatron",
    "dataset_path": "/data/datasets/my_sft_data",
    "output_dir": "/data/checkpoints/llama31-8b-lora",
    "num_gpus": 4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
  }'
```

This enqueues a job. The dispatcher picks it up and runs:

```
torchrun --nproc-per-node=4 app/worker/executor.py --job-id <uuid>
```

`torchrun` spawns 4 worker processes — one per GPU. Each process imports `megatron.bridge`, initialises its CUDA context, and joins the NCCL distributed group. The 4 GPUs then run the training in parallel using data parallelism, coordinated over NVLink. All output is captured to a single log file.

---

## Job State Machine

```
POST → QUEUED → RUNNING → COMPLETED
                  │      → FAILED
                  │      → CANCELLING → CANCELLED
                  └──────────────────► CANCELLED  (if still queued)
```

On API server restart, any job that was `RUNNING` is reset to `FAILED` with `error="server restarted"`. This prevents jobs from being stuck in a running state permanently after a crash.

Cancellation sends `SIGTERM` to the **entire torchrun process group**, not just the group leader. This ensures all GPU ranks are terminated — not just rank 0.

---

## Nvidia-Native Features

### Multi-GPU via torchrun

Without `torchrun`, a Python training script can only use one GPU. `torchrun` is PyTorch's built-in distributed launcher — it spawns one process per GPU, assigns each process a unique `LOCAL_RANK`, and sets up the NCCL process group so the GPUs can communicate.

```
torchrun --nproc-per-node=8   ← 8 processes, one per GPU
         --master-port=<free> ← coordinator port (auto-assigned)
         app/worker/executor.py --job-id <uuid>
```

On DGX hardware, GPU-to-GPU communication happens over NVLink (600 GB/s bandwidth). On cloud instances without NVLink, it falls back to PCIe. NCCL handles this automatically.

`num_gpus` is specified per job in the request body. You can run a 2-GPU LoRA job and a 4-GPU pretraining job simultaneously if the hardware has 8 GPUs — the dispatcher respects `num_gpus` per job.

### MFU Tracking

Model FLOP Utilization measures how efficiently your hardware is being used — 1.0 means you're achieving the theoretical peak FLOP/s of your GPUs.

```
MFU = (tokens_per_sec × model_FLOPs_per_token) / (num_gpus × gpu_peak_FLOPS)
```

- `tokens_per_sec` is parsed from the Megatron training log output
- `model_FLOPs_per_token` is computed from the model architecture (num_layers × hidden_size²)
- `gpu_peak_FLOPS` is looked up from a table of known GPU specs (H100: 989 TFLOPS BF16, A100: 312 TFLOPS BF16)

Falls back to `null` for unknown GPU models. A100/H100/RTX 6000 Ada are in the known table.

### FP8 Training (H100+)

Full `PrecisionConfig` schema for TransformerEngine FP8 training, which can deliver up to 2× throughput over BF16 on H100 hardware:

```json
{
  "precision": {
    "dtype": "fp8",
    "fp8_margin": 0,
    "fp8_interval": 1,
    "fp8_amax_history_len": 1024,
    "fp8_amax_compute_algo": "most_recent"
  }
}
```

FP8 requires an H100 or newer GPU. On older hardware, use `"dtype": "bfloat16"`.

---

## Security

| Area | How it's handled |
|------|-----------------|
| **Path traversal** | All filesystem ops go through `app/utils/paths.py`. Any path outside `CHECKPOINTS_ROOT` returns 400. Traversal payloads (`../`, `%2e%2e`, null bytes) all rejected. |
| **`hf_token` scrubbing** | `app/security/token_filter.py` redacts HuggingFace tokens, API keys, Bearer tokens, and passwords from all log output. |
| **Shell injection** | `launcher.py` uses `shell=False` with a whitelisted argument set. Enforced by AST test and bandit B602/B603 in CI. |
| **WebSocket** | Non-UUID `job_id` values close with code 1008 before any DB or filesystem access. |
| **CORS** | No `*` default. CORS origins require explicit `CORS_ORIGINS=https://your-ui.example.com` in `.env`. |
| **API auth** | Optional `API_KEY` setting. When set, all non-health endpoints require `Authorization: Bearer <key>` or `?api_key=<key>`. Disabled by default for self-hosted use. |
| **Rate limiting** | 60 write requests/minute/IP by default (`RATE_LIMIT_REQUESTS`). Sliding-window, no external dependency. |
| **Container** | Both Docker images run as non-root (uid 1001). No secrets in image layers. |

---

## Configuration

All settings are environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///megatronbridge.db` | SQLite file path |
| `CHECKPOINTS_ROOT` | `/data/checkpoints` | Root for all checkpoint paths |
| `LOGS_ROOT` | `/data/logs` | Root for job log files |
| `API_KEY` | *(empty — disabled)* | Enable Bearer token auth when set |
| `RATE_LIMIT_REQUESTS` | `60` | Write requests per minute per IP |
| `MAX_QUEUED_JOBS` | `100` | Max jobs in QUEUED state before 429 |
| `CORS_ORIGINS` | *(empty — disabled)* | Comma-separated allowed origins |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Development

```bash
# Lint + format
python -m ruff check app/
python -m ruff format app/

# Type check
python -m mypy app/

# Security scan
python -m bandit -r app/ -ll

# Full test suite (no GPU required)
python -m pytest tests/ -m "not gpu"

# GPU hardware tests (requires actual GPU)
python -m pytest tests/ -m gpu
```

### Test Structure

| Tier | Marker | Runs on | What it covers |
|------|--------|---------|----------------|
| Unit + integration | *(default)* | Every PR, CPU-only | All endpoints, WebSocket, auth, rate limiting, job state machine — all with mocked GPU |
| GPU hardware tests | `@pytest.mark.gpu` | Merge to main, self-hosted runner | Real torchrun subprocess, actual CUDA |

**Current coverage: 93.61%** across 315 tests.

The test suite mocks `megatron.bridge` via `sys.modules` injection in `conftest.py` so the full API can be tested without CUDA. `pynvml` is similarly mocked so GPU telemetry tests run on CPU machines.

---

## CI/CD

```
Every PR:
  lint (ruff + mypy)
    → security (bandit + pip-audit + secret scan)
      → test-cpu (315 tests, mocked GPU, ≥80% coverage)
        → schema-validate (schemathesis fuzz)
          → docker-lint (hadolint + actionlint)

Every merge to main:
  [all PR jobs]
    → test-gpu (self-hosted H100/A100 runner)
      → docker-build-push (ghcr.io)
        → trivy scan (zero HIGH/CRITICAL CVEs)
```

The merge pipeline reuses the PR pipeline via `workflow_call` — the gate set is always identical, never a divergent copy.

---

## Job State Machine

```
POST → QUEUED → RUNNING → COMPLETED
                  │      → FAILED
                  │      → CANCELLING → CANCELLED
                  └──────────────────► CANCELLED  (if still queued)
```

On server restart, any job that was `RUNNING` is reset to `FAILED` with `error="server restarted"`. This prevents stuck jobs after a crash.

---

## Roadmap (v0.2)

- [ ] **True multi-node training** — Slurm `sbatch` / Kubernetes Job launcher. Requires shared filesystem (NFS/Lustre) for checkpoints and a real database (Postgres) for the job queue, since SQLite cannot be shared across machines.
- [ ] **S3 checkpoint storage** — schema is ready, implementation behind `S3_ENABLED=true`
- [ ] **JWT auth** — stub middleware is in place, disabled by default
- [ ] **MFU for non-standard GPU models** — currently falls back to `null` for GPUs not in the known FLOP table

---

## License

MIT — see [LICENSE](LICENSE).
