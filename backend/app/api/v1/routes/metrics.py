"""Metrics Endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.logging import get_logger
from app.optimization.caching import cache_manager
from app.optimization.cost_tracker import cost_tracker
from app.utils.metrics import system_metrics

logger = get_logger(__name__)
router = APIRouter()


class MetricsResponse(BaseModel):
    """Metrics response model."""

    total_queries: int
    mrr_at_5: float
    quality_score: float
    latency_p95: float
    cost_per_query: float
    success_criteria: dict
    cache_stats: dict
    cost_report: str


@router.get("", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    # Calculate MRR (simplified - in production, use actual relevance scores)
    mrr = (
        system_metrics.calculate_mrr_at_5(
            [[s] for s in system_metrics.mrr_scores[:5]]
            if system_metrics.mrr_scores
            else []
        )
        if system_metrics.mrr_scores
        else 0.0
    )

    return MetricsResponse(
        total_queries=system_metrics.total_queries,
        mrr_at_5=mrr,
        quality_score=system_metrics.get_quality_score(),
        latency_p95=system_metrics.get_latency_p95(),
        cost_per_query=system_metrics.get_avg_cost_per_query(),
        success_criteria=system_metrics.check_success_criteria(),
        cache_stats=cache_manager.get_stats(),
        cost_report=cost_tracker.get_report(),
    )


@router.get("/success-criteria")
async def check_success_criteria():
    """Check success criteria status."""
    return system_metrics.check_success_criteria()


@router.get("/cost")
async def get_cost_report():
    """Get cost report."""
    return {"report": cost_tracker.get_report()}


@router.get("/report")
async def get_full_report():
    """Get full metrics report."""
    return {"report": system_metrics.get_report()}

