"""Logging filter that redacts sensitive tokens before they reach any handler.

Usage
-----
Apply once at application startup in ``app/main.py``::

    import logging
    from app.security.token_filter import TokenRedactFilter

    logging.getLogger().addFilter(TokenRedactFilter())

The filter is applied to the **root logger** so it covers every library that
uses Python's standard ``logging`` module, including uvicorn, FastAPI, and
MegatronBridge itself.
"""

from __future__ import annotations

import logging
import re

# Patterns that may carry sensitive values in log messages.
_REDACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(hf[_-]?token\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(password\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(secret\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"(token\s*[=:]\s*)(?!None|null|true|false)\S+", re.IGNORECASE),
    # Bearer tokens in Authorization headers
    re.compile(r"(Bearer\s+)\S+", re.IGNORECASE),
]

_REPLACEMENT = r"\1***REDACTED***"


def _redact(text: str) -> str:
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub(_REPLACEMENT, text)
    return text


class TokenRedactFilter(logging.Filter):
    """Scrub token-like values from every log record before emission."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.msg = _redact(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: _redact(str(v)) for k, v in record.args.items()}
            else:
                record.args = tuple(_redact(str(a)) for a in record.args)
        return True
