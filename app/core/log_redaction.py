"""Keeps API keys out of the logs (URLs with ?key=, exception messages and tracebacks)."""

from __future__ import annotations

import logging
import re

_SECRET_RE = re.compile(r"((?:[?&]|\b)(?:key|api_key|apikey|access_token)=)[^&\s'\"]+", re.IGNORECASE)
_HEADER_RE = re.compile(r"(x-goog-api-key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE)


def redact(text: str) -> str:
    if not text:
        return text
    return _HEADER_RE.sub(r"\1***", _SECRET_RE.sub(r"\1***", text))


class RedactingFormatter(logging.Formatter):
    """Wraps a formatter and redacts the fully rendered line, traceback included."""

    def __init__(self, inner: logging.Formatter):
        super().__init__()
        self._inner = inner

    def format(self, record: logging.LogRecord) -> str:
        return redact(self._inner.format(record))


def install_log_redaction() -> None:
    """Quiet per-request httpx logs and redact secrets on every root handler."""
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for logger_name in (None, "uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(logger_name).handlers:
            if not isinstance(handler.formatter, RedactingFormatter):
                handler.setFormatter(RedactingFormatter(handler.formatter or logging.Formatter()))
