# Contributing to MegatronBridge API

Thank you for contributing. This document covers everything you need to go from a fresh clone to a passing PR.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Lint, Type Check, Security Scan](#lint-type-check-security-scan)
- [Docker Dev Workflow](#docker-dev-workflow)
- [How to Add a New Job Type](#how-to-add-a-new-job-type)
- [Architecture Rules (Non-Negotiable)](#architecture-rules-non-negotiable)
- [PR Checklist](#pr-checklist)

---

## Development Setup

You do not need a GPU to develop or test this project. The full test suite runs on CPU with all GPU calls mocked.

```bash
git clone https://github.com/Doondi-Ashlesh/Nvidia-megatron-bridge-API
cd Nvidia-megatron-bridge-API

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install with all dev dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Most defaults are fine for local development — no edits required to run tests
```

The `[gpu]` extra (`torch`, `megatron-bridge`) is intentionally not installed here. `megatron.bridge` is mocked in the test suite — see [Architecture Rules](#architecture-rules-non-negotiable).

---

## Project Structure

```
app/
├── main.py                  # FastAPI app factory, lifespan, middleware registration
├── config.py                # All settings (pydantic-settings, env vars)
├── database.py              # SQLite init, WAL mode, startup recovery
│
├── api/                     # HTTP + WebSocket route handlers
│   ├── checkpoints.py       # /v1/checkpoints
│   ├── training.py          # /v1/training
│   ├── peft.py              # /v1/peft
│   ├── jobs.py              # /v1/jobs
│   ├── system.py            # /v1/system/info
│   ├── health.py            # /health, /health/ready
│   └── ws.py                # WebSocket: /v1/ws/jobs/{id}/logs|progress
│
├── schemas/                 # Pydantic request/response models
│   ├── common.py            # JobStatus enum, shared types
│   ├── checkpoint.py        # ImportRequest, ExportRequest
│   ├── training.py          # PretrainRequest, FinetuneRequest
│   ├── peft.py              # LoRARequest, DoRARequest
│   └── config_container.py  # Full ConfigContainer tree (incl. FP8/PrecisionConfig)
│
├── services/                # Business logic, no HTTP concerns
│   ├── job_service.py       # Job CRUD, state transitions, queue depth
│   ├── checkpoint_service.py# Path resolution, metadata, checkpoint CRUD
│   ├── log_service.py       # Megatron log parsing, progress DB writer
│   ├── gpu_service.py       # pynvml telemetry (used by executor, not API server)
│   └── training_service.py  # SDK config conversion (pydantic → ConfigContainer)
│
├── security/
│   ├── auth.py              # ApiKeyMiddleware (Bearer + ?api_key= for WebSocket)
│   ├── rate_limit.py        # RateLimitMiddleware (sliding window, no dependency)
│   └── token_filter.py      # Logging filter — scrubs hf_token from all log output
│
├── worker/
│   ├── dispatcher.py        # Asyncio loop: polls SQLite, launches torchrun per job
│   ├── launcher.py          # torchrun command builder, argument whitelist, SIGTERM
│   └── executor.py          # *** ONLY file that imports megatron.bridge ***
│
└── utils/
    └── paths.py             # Path validation — security boundary for all filesystem ops

tests/
├── conftest.py              # Root fixtures: test_app, client, mock megatron.bridge
├── unit/
│   ├── phase1/              # config, paths, database, health
│   ├── phase2/              # job state machine, dispatcher, launcher
│   ├── phase3/              # checkpoints router + service
│   ├── phase4/              # training + PEFT router, config_container
│   ├── phase5/              # GPU telemetry, system/info
│   ├── phase6/              # WebSocket log + progress streams
│   ├── phase7/              # log_service, auth, rate_limit
│   └── security/            # token_filter, hf_token scrubbing
└── gpu/                     # @pytest.mark.gpu — requires real hardware
```

---

## Running Tests

```bash
# Full test suite — no GPU required (315 tests, ~20s)
python -m pytest tests/ -m "not gpu"

# Single phase
python -m pytest tests/unit/phase3/ -v

# Specific test file
python -m pytest tests/unit/phase6/test_streaming_router.py -v

# With coverage report
python -m pytest tests/ -m "not gpu" --cov=app --cov-report=html
# → open htmlcov/index.html

# GPU hardware tests (requires an actual H100/A100)
python -m pytest tests/ -m gpu
```

The coverage gate is **80% minimum**. The full suite currently sits at **93.61%**. Do not let a PR drop it below 80%.

### What is mocked

`conftest.py` injects mock objects into `sys.modules` before any test runs:

- `megatron.bridge` — the entire SDK is replaced with a `MagicMock`. This means `executor.py` can be imported and tested without CUDA.
- `pynvml` — replaced with a fixture that returns predictable GPU telemetry. GPU service tests run on CPU.

If your change touches `executor.py`, write tests that verify the mock SDK is called with the correct arguments — not that the SDK itself works (that's what GPU hardware tests are for).

---

## Lint, Type Check, Security Scan

All of these must pass before a PR is mergeable. Run them locally before pushing:

```bash
# Lint (ruff — covers style, imports, security patterns, unused code)
python -m ruff check app/
python -m ruff format app/       # auto-fix formatting

# Type check (mypy — strict on security-critical modules)
python -m mypy app/

# Security scan (bandit — B602/B603/B604/B605/B606/B607 always on)
python -m bandit -r app/ -ll

# Dependency vulnerability scan
python -m pip_audit
```

### Common ruff issues and how to fix them

| Error | Cause | Fix |
|-------|-------|-----|
| `S603` | `subprocess` without `shell=False` check | Always pass `shell=False` explicitly |
| `T201` | `print()` in production code | Use `logger.info()` instead |
| `PTH123` | `open(path)` instead of `Path.open()` | Use `path.open()` or `aiofiles.open()` |
| `ERA001` | Commented-out code | Delete it — use git history instead |
| `S101` | `assert` in non-test code | Use `if not x: raise ValueError(...)` |

---

## Docker Dev Workflow

### CPU-only development (no GPU needed)

```bash
# Build and start the API server
docker compose up -d api

# View logs
docker compose logs -f api

# Run tests inside the container
docker compose run --rm api python -m pytest tests/ -m "not gpu"

# Rebuild after code changes
docker compose up -d --build api
```

### With GPU worker

Requires `nvidia-container-toolkit` installed on the host.

```bash
# Start both API server and GPU worker
docker compose --profile gpu up -d

# View worker logs for a specific job
docker compose logs -f worker
# or stream via WebSocket:
# wscat -c ws://localhost:8000/v1/ws/jobs/<job_id>/logs
```

### Two-image design

| Image | Base | Size | Purpose |
|-------|------|------|---------|
| `megatronbridge-api` | `python:3.12-slim` | ~200 MB | API server (CPU-only) |
| `megatronbridge-worker` | `nvcr.io/nvidia/nemo:25.02` | ~22 GB | GPU worker (CUDA 12.8+) |

The worker image is large because the NeMo NGC base ships CUDA, PyTorch 2.7+, and TransformerEngine pre-built. This is intentional — building TransformerEngine from source in CI is fragile and slow.

---

## How to Add a New Job Type

Adding a new operation type (e.g., `quantize`, `merge_lora`) follows the same pattern every time. Here are all the files you need to touch, in order:

### 1. Add the enum value

```python
# app/schemas/common.py
class JobType(str, Enum):
    ...
    QUANTIZE = "quantize"    # ← add here
```

### 2. Create the request schema

```python
# app/schemas/quantize.py  (new file)
from pydantic import BaseModel, Field

class QuantizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_checkpoint: str
    output_dir: str
    num_gpus: int = Field(default=1, ge=1, le=64)
    bits: Literal[4, 8] = 8
```

All schemas must use `extra="forbid"` — unknown fields return 422 instead of being silently ignored.

### 3. Add the route handler

```python
# app/api/quantize.py  (new file)
from fastapi import APIRouter, Depends
import aiosqlite
from app.database import get_db
from app.schemas.quantize import QuantizeRequest
from app.services import job_service
from app.utils.paths import safe_checkpoint_path

router = APIRouter(prefix="/v1/quantize", tags=["Quantize"])

@router.post("", status_code=202)
async def launch_quantize(req: QuantizeRequest, db: aiosqlite.Connection = Depends(get_db)):
    source_path = safe_checkpoint_path(settings.checkpoints_root, req.source_checkpoint)
    output_path = safe_checkpoint_path(settings.checkpoints_root, req.output_dir)

    payload = {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "bits": req.bits,
        "num_gpus": req.num_gpus,
    }
    job = await job_service.create_job(db, job_type=JobType.QUANTIZE, payload=payload, num_gpus=req.num_gpus)
    return {"job_id": job["id"], "status": job["status"]}
```

### 4. Register the router

```python
# app/api/router.py
from app.api import quantize
router.include_router(quantize.router)
```

### 5. Add the executor handler

```python
# app/worker/executor.py  — add inside _run_job()
elif job_type == "quantize":
    handle_quantize(job, payload)

def handle_quantize(job: dict, payload: dict) -> None:
    from megatron.bridge import AutoBridge
    bridge = AutoBridge.from_megatron_pretrained(payload["source_path"])
    bridge.quantize(output_path=payload["output_path"], bits=payload["bits"])
```

### 6. Write tests

```
tests/unit/phaseN/
├── test_quantize_router.py    # POST /v1/quantize → 202, validates payload
└── test_quantize_service.py   # executor handler called with correct args (mocked SDK)
```

At minimum: happy path (202 + job_id), missing required field (422), path traversal attempt (400).

---

## Architecture Rules (Non-Negotiable)

These rules are enforced by CI. A PR that breaks any of them will not merge.

### 1. `executor.py` is the only file that imports `megatron.bridge`

Every other file in `app/` must be importable without CUDA. If you find yourself writing `from megatron.bridge import ...` outside of `executor.py`, stop and restructure.

### 2. All filesystem operations go through `app/utils/paths.py`

Never construct a `Path` from user input directly. Always use `safe_checkpoint_path()` or `safe_log_path()`:

```python
# Wrong — path traversal vulnerability
path = Path(settings.checkpoints_root) / user_input

# Correct
from app.utils.paths import safe_checkpoint_path
path = safe_checkpoint_path(settings.checkpoints_root, user_input)
# raises PathTraversalError (→ HTTP 400) if user_input contains ../
```

### 3. No `shell=True` in subprocess calls

All `subprocess.Popen` / `subprocess.run` calls must use `shell=False` and a list of arguments. The argument whitelist in `launcher.py` is enforced by an AST test — if you add a new torchrun flag, add it to the whitelist first.

### 4. `hf_token` must never appear in logs, DB columns, or API responses

The `app/security/token_filter.py` logging filter scrubs tokens from log output. When adding new fields to a schema, if the field is a secret, mark it:

```python
hf_token: str | None = Field(default=None, exclude=True)  # excluded from serialisation
```

And add a test asserting the value is not present in any log output.

### 5. All Pydantic models use `extra="forbid"`

```python
class MyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
```

This makes unknown fields return 422 instead of being silently accepted. Checked by a ruff rule.

---

## PR Checklist

Before opening a PR, verify:

- [ ] `python -m pytest tests/ -m "not gpu"` passes with ≥ 80% coverage
- [ ] `python -m ruff check app/` — zero errors
- [ ] `python -m mypy app/` — zero errors on touched modules
- [ ] `python -m bandit -r app/ -ll` — zero findings
- [ ] New endpoints have tests for: happy path, validation error (422), path traversal (400 if applicable)
- [ ] No `hf_token`, `api_key`, or `password` literals hardcoded anywhere
- [ ] `executor.py` is still the only file importing `megatron.bridge`
- [ ] Decision log updated if the implementation differs from the plan (`docs/decision-log.md`)
