"""LangSmith Integration for Monitoring and Tracing."""

import os
from app.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


def setup_langsmith() -> None:
    """Setup LangSmith tracing if enabled."""
    if not settings.LANGCHAIN_TRACING_V2:
        logger.debug("langsmith_disabled")
        return

    if not settings.LANGCHAIN_API_KEY:
        logger.warning(
            "langsmith_api_key_missing",
            message="LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is not set",
        )
        return

    # Set environment variables for LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

    if settings.LANGCHAIN_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT

    logger.info(
        "langsmith_enabled",
        project=settings.LANGCHAIN_PROJECT,
        endpoint=settings.LANGCHAIN_ENDPOINT or "default",
    )


def is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    return settings.LANGCHAIN_TRACING_V2 and bool(settings.LANGCHAIN_API_KEY)

