"""FastAPI Application Entry Point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.middleware import setup_cors, ExceptionHandlerMiddleware, LoggingMiddleware
from app.api.v1.router import api_router
from app.config import get_settings
from app.core.exceptions import RAGSystemException
from app.core.langsmith import setup_langsmith
from app.core.logging import setup_logging

settings = get_settings()

# Setup logging
setup_logging()

# Setup LangSmith if enabled
setup_langsmith()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    from app.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("application_starting", environment=settings.ENVIRONMENT)

    yield

    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready RAG System API with LangGraph, Hybrid Search, and Reranking",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Setup middleware
setup_cors(app)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ExceptionHandlerMiddleware)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(RAGSystemException)
async def rag_system_exception_handler(request, exc: RAGSystemException):
    """Handle custom exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG System API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.APP_VERSION}


@app.get("/health/ready")
async def readiness():
    """Readiness check endpoint."""
    # Check if system is ready (documents loaded, etc.)
    from app.api.v1.routes import documents

    docs = documents.get_all_documents()
    return {
        "status": "ready" if docs else "not_ready",
        "documents_loaded": len(docs),
    }

