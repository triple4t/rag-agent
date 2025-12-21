"""API v1 Router."""

from fastapi import APIRouter

from app.api.v1.routes import documents, metrics, queries

api_router = APIRouter()

# Include route modules
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(queries.router, prefix="/queries", tags=["queries"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])

