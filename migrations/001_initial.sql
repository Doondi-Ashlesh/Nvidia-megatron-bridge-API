-- MegatronBridge API — Initial Schema
-- SQLite with WAL mode enabled by the application layer on first connection.

-- ── Jobs ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,          -- UUID v4
    type         TEXT NOT NULL,             -- checkpoint_import | checkpoint_export |
                                            -- pretrain | finetune | lora | dora
    status       TEXT NOT NULL DEFAULT 'queued',
                                            -- queued | running | completed |
                                            -- failed | cancelling | cancelled
    payload      TEXT NOT NULL,             -- JSON-serialised request body
    error        TEXT,                      -- error message when status = failed
    progress     TEXT,                      -- JSON: {step, total_steps, loss, lr,
                                            --        mfu, tokens_per_sec, gpus:[…]}
    log_path     TEXT,                      -- absolute path to job log file
    pid          INTEGER,                   -- torchrun process-group leader PID
    num_gpus     INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,             -- ISO-8601 UTC
    started_at   TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status    ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_type      ON jobs (type);
CREATE INDEX IF NOT EXISTS idx_jobs_created   ON jobs (created_at);

-- ── Checkpoints ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS checkpoints (
    id               TEXT PRIMARY KEY,      -- UUID v4
    name             TEXT NOT NULL,
    format           TEXT NOT NULL,         -- 'megatron' | 'hf'
    path             TEXT NOT NULL UNIQUE,  -- absolute filesystem path
    size_bytes       INTEGER,
    model_arch       TEXT,                  -- e.g. 'llama3', 'qwen3'
    created_from_job TEXT,                  -- FK → jobs.id (nullable)
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ckpt_format    ON checkpoints (format);
CREATE INDEX IF NOT EXISTS idx_ckpt_created   ON checkpoints (created_at);
