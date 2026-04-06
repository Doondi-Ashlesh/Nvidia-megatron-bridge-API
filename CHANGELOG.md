# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-06

Initial release. Full REST + WebSocket API wrapper around Nvidia's MegatronBridge SDK (`megatron-bridge>=0.3.1`), targeting single-node multi-GPU training setups (DGX H100, A100 cloud instances).

### Added

#### Core infrastructure
- SQLite job queue with WAL mode, startup recovery (stuck `RUNNING` jobs reset to `FAILED` on boot)
- Two-process architecture: CPU-only FastAPI server + GPU worker subprocess launched via `torchrun`
- `app/worker/executor.py` — isolated worker entry point, the only file that imports `megatron.bridge`
- `app/worker/dispatcher.py` — asyncio polling loop that dequeues jobs and launches `torchrun`
- `app/worker/launcher.py` — `torchrun` command builder with argument whitelist, free-port detection, SIGTERM to process group for cancellation

#### REST endpoints
- `POST /v1/checkpoints/import` — HuggingFace → Megatron-Core checkpoint conversion (202 + job_id)
- `POST /v1/checkpoints/export` — Megatron-Core → HuggingFace checkpoint conversion (202 + job_id)
- `GET /v1/checkpoints` — list all registered checkpoints
- `GET /v1/checkpoints/{id}` — checkpoint metadata
- `DELETE /v1/checkpoints/{id}` — delete checkpoint files + DB record
- `POST /v1/training/pretrain` — launch pretraining job
- `POST /v1/training/finetune` — launch SFT fine-tuning job
- `GET /v1/training` — list training jobs
- `POST /v1/peft/lora` — launch LoRA fine-tuning
- `POST /v1/peft/dora` — launch DoRA fine-tuning
- `GET /v1/peft` — list LoRA/DoRA jobs
- `GET /v1/jobs` — list all jobs (filterable by type and status)
- `GET /v1/jobs/{id}` — job status, progress, GPU telemetry
- `DELETE /v1/jobs/{id}` — cancel running job (SIGTERM process group) or permanently delete terminal job
- `GET /v1/jobs/{id}/logs` — fetch log file over HTTP with `line_offset` / `line_limit` pagination
- `GET /health` — liveness probe
- `GET /health/ready` — readiness probe (DB + worker alive)
- `GET /v1/system/info` — CUDA version, GPU names, NCCL version, megatron-bridge version, supported models

#### WebSocket streaming
- `WS /v1/ws/jobs/{id}/logs` — real-time log streaming (catch-up + live poll every 200 ms), terminal frame `{"type": "stream_end", "status": "..."}`
- `WS /v1/ws/jobs/{id}/progress` — progress frames every 2 s with step, loss, lr, grad_norm, samples/sec, TFLOPs + per-GPU telemetry (util%, mem, temp), terminal frame `{"type": "job_complete", "status": "..."}`
- WebSocket auth via `?api_key=` query parameter (Bearer header not available for WS clients)
- Invalid/unknown `job_id` closes with code 1008 before any DB or filesystem access

#### Nvidia-native features
- `app/services/gpu_service.py` — pynvml GPU telemetry (utilisation %, memory used/total, temperature)
- `app/services/log_service.py` — Megatron training log parser: extracts step, total_steps, loss, lr, grad_norm, samples/sec, TFLOPs using independent per-field regex (order-independent)
- `app/worker/executor.py` `_ProgressWatcher` — background daemon thread inside the GPU worker that polls the log file + pynvml every 10 s and writes to `jobs.progress` in SQLite
- MFU tracking for H100, A100, RTX 6000 Ada (falls back to `null` for unknown GPU models)
- FP8 training support via `PrecisionConfig` schema (TransformerEngine, H100+)
- Multi-GPU support via `torchrun --nproc-per-node=N`, configurable per job

#### Security
- `app/utils/paths.py` — path traversal protection for all filesystem operations; rejects `../`, `%2e%2e`, null bytes
- `app/security/auth.py` — optional `ApiKeyMiddleware` (Bearer token + `?api_key=` query param); `hmac.compare_digest` timing-safe comparison; health endpoints always exempt
- `app/security/rate_limit.py` — `RateLimitMiddleware`, in-memory sliding window, 60 write requests/minute/IP default, no external dependency
- `app/security/token_filter.py` — logging filter that redacts `hf_token`, `api_key`, Bearer tokens, and passwords from all log output
- Queue depth limit (`MAX_QUEUED_JOBS`, default 100) — returns HTTP 429 when exceeded
- All Pydantic schemas use `extra="forbid"`
- `shell=False` enforced on all subprocess calls; argument whitelist in `launcher.py`
- `hf_token` passed to worker via environment variable only, never written to DB or logs

#### Schemas
- `ImportRequest`, `ExportRequest` (with required `model_arch`)
- `PretrainRequest`, `FinetuneRequest`
- `LoRARequest`, `DoRARequest`
- `ConfigContainer` — full Pydantic tree for MegatronBridge SDK config including `PrecisionConfig` (FP8), `OptimizerConfig`, `LoRAConfig`, `DoRAConfig`
- `JobResponse`, `JobListResponse` with pagination (`limit`, `offset`)

#### Infrastructure
- `docker/Dockerfile` — CPU-only API server (`python:3.12-slim`, non-root uid 1001, HEALTHCHECK)
- `docker/Dockerfile.worker` — GPU worker (`nvcr.io/nvidia/nemo:25.02`, CUDA 12.8+, megatron-bridge installed with `--no-build-isolation`)
- `docker-compose.yml` — GPU worker behind `--profile gpu`, `depends_on: service_healthy`, named volume `megatronbridge_data`
- `.github/workflows/pr.yml` — lint → security → test-cpu → schema-validate → docker-lint
- `.github/workflows/merge.yml` — reuses PR pipeline via `workflow_call`, adds GPU tests + docker build/push + trivy scan
- `.gitignore`, `.dockerignore`, `LICENSE` (MIT)

#### Testing
- 315 tests, 93.61% coverage
- Full test suite runs on CPU — `megatron.bridge` and `pynvml` mocked via `sys.modules` injection
- Test tiers: unit/integration (default, CPU), GPU hardware (`@pytest.mark.gpu`, self-hosted runner)
- Security fuzz: path traversal probes on all filesystem-touching endpoints; WebSocket fuzz with malformed job_ids

### Architecture decisions

See `docs/decision-log.md` for the full record of every place the implementation differs from the original plan, with rationale. Key decisions:

- SQLite over Redis/Celery — GPU is the bottleneck; single-node; zero infrastructure
- `torchrun` for multi-GPU — required for NCCL process group setup; without it only GPU 0 runs
- `python:3.12-slim` for API, `nvcr.io/nvidia/nemo:25.02` for worker — megatron-bridge requires CUDA 12.8+ and PyTorch 2.7+, which only the NeMo image ships pre-built
- Disk-polling for WebSocket log streaming — worker is a separate process, no shared asyncio loop; filesystem is the only shared interface
- GPU telemetry written by worker to `jobs.progress`, read by API server from DB — API server is CPU-only and must not call pynvml directly

---

## [Unreleased]

### Planned for v0.2
- True multi-node training (Slurm / Kubernetes job launching)
- S3 checkpoint storage (`S3_ENABLED=true`)
- JWT auth (stub middleware ready, disabled by default)
- MFU tracking for non-standard GPU models
