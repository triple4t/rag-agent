"""Structured Logging Setup."""

import logging
import sys
from typing import Any

import structlog
from app.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Configure structured logging for the application."""
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )

    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Add JSON renderer for production, or console for development
    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_request(request_id: str, method: str, path: str, **kwargs: Any) -> None:
    """Log HTTP request."""
    logger = get_logger(__name__)
    logger.info(
        "http_request",
        request_id=request_id,
        method=method,
        path=path,
        **kwargs,
    )


def log_response(
    request_id: str, status_code: int, duration_ms: float, **kwargs: Any
) -> None:
    """Log HTTP response."""
    logger = get_logger(__name__)
    logger.info(
        "http_response",
        request_id=request_id,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_error(error: Exception, **kwargs: Any) -> None:
    """Log error with context."""
    logger = get_logger(__name__)
    logger.error(
        "error_occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        **kwargs,
        exc_info=True,
    )

