"""Filesystem path resolution and security boundary.

ALL path construction that uses user-supplied input MUST go through
``resolve_safe_path``. No other module may call ``Path(user_input)``
directly — this is enforced by ruff's ``banned-api`` rule.

Security model
--------------
* The ``base`` directory is the trust boundary (e.g. CHECKPOINTS_ROOT or LOGS_ROOT).
* Any resolved path that is not strictly under ``base`` raises ``PathTraversalError``
  which the API layer converts to HTTP 400.
* Symlinks are resolved (``Path.resolve()``) before the containment check so that
  a symlink pointing outside ``base`` is also caught.
"""

from __future__ import annotations

from pathlib import Path


class PathTraversalError(ValueError):
    """Raised when a user-supplied path resolves outside the allowed base."""


def resolve_safe_path(base: Path, user_input: str) -> Path:
    """Resolve *user_input* relative to *base* and assert containment.

    Parameters
    ----------
    base:
        The trusted root directory.  Must be an absolute path.
    user_input:
        A relative path fragment, an absolute path, or a HuggingFace
        model-id slug (e.g. ``"meta-llama/Llama-3-8B"``).
        Null bytes are rejected immediately.

    Returns
    -------
    Path
        The resolved absolute path, guaranteed to be a descendant of *base*.

    Raises
    ------
    ValueError
        If *user_input* is empty or contains null bytes.
    PathTraversalError
        If the resolved path is not under *base*.
    """
    if not user_input:
        msg = "Path input must not be empty."
        raise ValueError(msg)
    if "\x00" in user_input:
        msg = "Path input must not contain null bytes."
        raise ValueError(msg)

    # Resolve base to an absolute, symlink-free path.
    resolved_base = base.resolve()

    # Build candidate path and resolve symlinks.
    candidate = Path(user_input)
    if candidate.is_absolute():
        resolved_candidate = candidate.resolve()
    else:
        resolved_candidate = (resolved_base / candidate).resolve()

    # Containment check — resolved_candidate must start with resolved_base.
    try:
        resolved_candidate.relative_to(resolved_base)
    except ValueError as exc:
        msg = (
            f"Path {user_input!r} resolves to {resolved_candidate} which is "
            f"outside the allowed base {resolved_base}."
        )
        raise PathTraversalError(msg) from exc

    return resolved_candidate


def safe_log_path(logs_root: Path, job_id: str) -> Path:
    """Return the log file path for *job_id*, validated against *logs_root*.

    ``job_id`` must be a UUID string (validated upstream), but we still run
    it through ``resolve_safe_path`` as a defence-in-depth measure.
    """
    return resolve_safe_path(logs_root, f"{job_id}.log")


def safe_checkpoint_path(checkpoints_root: Path, user_input: str) -> Path:
    """Validate and resolve a checkpoint path supplied by the user."""
    return resolve_safe_path(checkpoints_root, user_input)
