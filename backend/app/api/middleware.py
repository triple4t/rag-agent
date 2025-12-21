"""API Middleware for CORS, Logging, and Exception Handling."""

import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.core.exceptions import RAGSystemException
from app.core.logging import get_logger, log_request, log_response, log_error

settings = get_settings()
logger = get_logger(__name__)


def setup_cors(app: FastAPI) -> None:
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Log request
        log_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
        )

        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            log_response(
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

            # Add request ID to response header
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_error(e, request_id=request_id, duration_ms=duration_ms)
            raise


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for exception handling."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle exceptions and return appropriate responses."""
        try:
            return await call_next(request)
        except RAGSystemException as e:
            from fastapi.responses import JSONResponse

            log_error(e, request_id=getattr(request.state, "request_id", None))
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.message,
                    "details": e.details,
                    "request_id": getattr(request.state, "request_id", None),
                },
            )
        except Exception as e:
            from fastapi.responses import JSONResponse

            log_error(e, request_id=getattr(request.state, "request_id", None))
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e) if settings.DEBUG else "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

