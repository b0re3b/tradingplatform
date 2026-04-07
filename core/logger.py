"""
core/logger.py

Production-grade centralized logging for the trading system.

Features:
- Console + optional rotating file logging
- JSON or plain text formatting
- Context fields support via LoggerAdapter
- Redaction of sensitive values
- Correlation / trace id support
- Safe reconfiguration guard
"""

from __future__ import annotations

import json
import logging
import logging.config
import logging.handlers
import os
import re
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# Context variables for request / event correlation
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[str | None] = ContextVar("span_id", default=None)
service_var: ContextVar[str | None] = ContextVar("service_name", default=None)

# Avoid duplicate handler setup if init_logger() is called multiple times
_LOGGER_CONFIGURED = False

# Common secret keys we never want to leak into logs
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "api_secret",
    "passphrase",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "private_key",
    "signature",
}

# Regex patterns for accidental secret-like values in text
SECRET_PATTERNS = [
    re.compile(r"(ghp_[A-Za-z0-9]{20,})"),   # GitHub personal access token-like
    re.compile(r"(sk-[A-Za-z0-9]{20,})"),    # OpenAI-like token prefix
    re.compile(r"([A-Za-z0-9_\-]{32,})"),    # Generic long token fallback
]


def set_trace_context(
    trace_id: str | None = None,
    span_id: str | None = None,
    service_name: str | None = None,
) -> None:
    """Set correlation context for subsequent logs in the current context."""
    if trace_id is not None:
        trace_id_var.set(trace_id)
    if span_id is not None:
        span_id_var.set(span_id)
    if service_name is not None:
        service_var.set(service_name)


def clear_trace_context() -> None:
    """Clear correlation context."""
    trace_id_var.set(None)
    span_id_var.set(None)
    service_var.set(None)


def _sanitize_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive keys in nested dicts."""
    sanitized: dict[str, Any] = {}

    for key, value in data.items():
        lowered = key.lower()
        if lowered in SENSITIVE_KEYS:
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_mapping(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_mapping(item) if isinstance(item, dict) else _sanitize_text(item)
                for item in value
            ]
        else:
            sanitized[key] = _sanitize_text(value)

    return sanitized


def _sanitize_text(value: Any) -> Any:
    """Redact secret-like values in strings; pass through non-strings."""
    if not isinstance(value, str):
        return value

    sanitized = value
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("***REDACTED***", sanitized)
    return sanitized


class ContextFilter(logging.Filter):
    """Inject context variables into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get() or "-"
        record.span_id = span_id_var.get() or "-"
        record.service_name = service_var.get() or "-"
        return True


class SensitiveDataFilter(logging.Filter):
    """Redact secrets from log record msg and args."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _sanitize_text(record.msg)

        if isinstance(record.args, dict):
            record.args = _sanitize_mapping(record.args)
        elif isinstance(record.args, tuple):
            record.args = tuple(_sanitize_text(arg) for arg in record.args)

        return True


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for ingestion by log pipelines."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": _sanitize_text(message),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "service": getattr(record, "service_name", "-"),
            "trace_id": getattr(record, "trace_id", "-"),
            "span_id": getattr(record, "span_id", "-"),
        }

        # Include known extra fields if present
        for field in (
            "exchange",
            "symbol",
            "strategies",
            "signal_id",
            "order_id",
            "position_id",
            "event_type",
        ):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = _sanitize_text(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class PlainTextFormatter(logging.Formatter):
    """Readable formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, self.datefmt)} | "
            f"{record.levelname:<8} | "
            f"{record.name} | "
            f"svc={getattr(record, 'service_name', '-')} | "
            f"trace={getattr(record, 'trace_id', '-')} | "
            f"span={getattr(record, 'span_id', '-')} | "
            f"{record.getMessage()}"
        )
        if record.exc_info:
            return f"{base}\n{self.formatException(record.exc_info)}"
        return base


class TradingLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter for attaching consistent domain metadata.

    Example:
        logger = get_logger(__name__, exchange="binance", symbol="BTCUSDT")
        logger.info("Connected")
    """

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        extra.update(self.extra)
        return msg, kwargs


def _build_handlers(
    *,
    log_dir: str | None,
    log_level: str,
    json_logs: bool,
    enable_file_logging: bool,
    max_bytes: int,
    backup_count: int,
) -> dict[str, dict[str, Any]]:
    handlers: dict[str, dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "json" if json_logs else "plain",
            "stream": "ext://sys.stdout",
            "filters": ["context", "sensitive"],
        }
    }

    if enable_file_logging:
        if not log_dir:
            raise ValueError("log_dir must be provided when enable_file_logging=True")

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / "trading_system.log"

        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json" if json_logs else "plain",
            "filename": str(log_path),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
            "filters": ["context", "sensitive"],
        }

    return handlers


def init_logger(
    *,
    service_name: str = "trading_system",
    log_level: str | None = None,
    json_logs: bool | None = None,
    log_dir: str | None = None,
    enable_file_logging: bool | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    force_reconfigure: bool = False,
) -> None:
    """
    Initialize root logging configuration.

    Priority:
    1. Explicit args
    2. Environment variables
    3. Safe defaults
    """
    global _LOGGER_CONFIGURED

    if _LOGGER_CONFIGURED and not force_reconfigure:
        return

    resolved_log_level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    resolved_json_logs = (
        json_logs if json_logs is not None
        else os.getenv("LOG_JSON", "false").lower() == "true"
    )
    resolved_enable_file_logging = (
        enable_file_logging if enable_file_logging is not None
        else os.getenv("LOG_TO_FILE", "false").lower() == "true"
    )
    resolved_log_dir = log_dir or os.getenv("LOG_DIR", "logs")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "context": {
                "()": ContextFilter,
            },
            "sensitive": {
                "()": SensitiveDataFilter,
            },
        },
        "formatters": {
            "json": {
                "()": JsonFormatter,
            },
            "plain": {
                "()": PlainTextFormatter,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": _build_handlers(
            log_dir=resolved_log_dir,
            log_level=resolved_log_level,
            json_logs=resolved_json_logs,
            enable_file_logging=resolved_enable_file_logging,
            max_bytes=max_bytes,
            backup_count=backup_count,
        ),
        "root": {
            "level": resolved_log_level,
            "handlers": ["console", "file"] if resolved_enable_file_logging else ["console"],
        },
        "loggers": {
            "uvicorn": {"level": resolved_log_level, "propagate": True},
            "uvicorn.error": {"level": resolved_log_level, "propagate": True},
            "uvicorn.access": {"level": resolved_log_level, "propagate": True},
            "websockets": {"level": "WARNING", "propagate": True},
            "aiohttp": {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(config)
    set_trace_context(service_name=service_name)
    _LOGGER_CONFIGURED = True


def get_logger(name: str, **extra: Any) -> logging.Logger | TradingLoggerAdapter:
    """
    Return a logger or adapter with optional default metadata.
    """
    logger = logging.getLogger(name)
    if extra:
        return TradingLoggerAdapter(logger, extra)
    return logger


if __name__ == "__main__":
    init_logger(
        service_name="logger_demo",
        log_level="DEBUG",
        json_logs=False,
        enable_file_logging=False,
        force_reconfigure=True,
    )

    set_trace_context(trace_id="trace-123", span_id="span-abc", service_name="logger_demo")

    log = get_logger(__name__, exchange="binance", symbol="BTCUSDT", strategy="orderflow")
    log.info("Logger initialized successfully")
    log.warning("This token should never appear: ghp_abcdefghijklmnopqrstuvwxyz123456")
    try:
        raise RuntimeError("Demo exception")
    except RuntimeError:
        log.exception("Something went wrong")