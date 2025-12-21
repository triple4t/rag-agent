"""Success Metrics Tracking (from roadmap lines 96-100)."""

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import List, Optional

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    query: str
    latency: float  # seconds
    quality_score: float
    cost: float  # dollars
    mrr_at_5: Optional[float] = None  # Mean Reciprocal Rank @ 5
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemMetrics:
    """Aggregated system metrics."""

    total_queries: int = 0
    latencies: List[float] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    mrr_scores: List[float] = field(default_factory=list)

    def add_query(self, metrics: QueryMetrics):
        """Add query metrics."""
        self.total_queries += 1
        self.latencies.append(metrics.latency)
        self.quality_scores.append(metrics.quality_score)
        self.costs.append(metrics.cost)
        if metrics.mrr_at_5 is not None:
            self.mrr_scores.append(metrics.mrr_at_5)

    def calculate_mrr_at_5(self, relevance_scores: List[List[float]]) -> float:
        """
        Calculate MRR@5 (Mean Reciprocal Rank) - Line 97.
        MRR@5 ≥ 0.7 target
        """
        if not relevance_scores:
            return 0.0

        # For each query, find rank of first relevant result (score > threshold)
        reciprocal_ranks = []
        threshold = 0.7  # Relevance threshold

        for scores in relevance_scores:
            # Take top 5 only
            top_5_scores = scores[:5]
            for rank, score in enumerate(top_5_scores, start=1):
                if score >= threshold:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)  # No relevant result in top 5

        return mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def get_quality_score(self) -> float:
        """
        Get average quality score - Line 98.
        Quality score ≥ 0.75 target
        """
        return mean(self.quality_scores) if self.quality_scores else 0.0

    def get_latency_p95(self) -> float:
        """
        Get p95 latency - Line 99.
        Latency < 2s p95 target
        """
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        return (
            sorted_latencies[p95_index]
            if p95_index < len(sorted_latencies)
            else sorted_latencies[-1]
        )

    def get_avg_cost_per_query(self) -> float:
        """
        Get average cost per query - Line 100.
        Cost per query < $0.01 target
        """
        return mean(self.costs) if self.costs else 0.0

    def check_success_criteria(self) -> dict:
        """Check if all success metrics are met."""
        # Calculate MRR if we have relevance scores
        mrr = (
            self.calculate_mrr_at_5(
                [[s] for s in self.mrr_scores[:5]] if self.mrr_scores else []
            )
            if self.mrr_scores
            else 0.0
        )
        quality = self.get_quality_score()
        latency = self.get_latency_p95()
        cost = self.get_avg_cost_per_query()

        return {
            "mrr_at_5": mrr >= settings.TARGET_MRR,
            "quality_score": quality >= settings.TARGET_QUALITY,
            "latency_p95": latency <= settings.TARGET_LATENCY_P95,
            "cost_per_query": cost <= settings.TARGET_COST_PER_QUERY,
        }

    def get_report(self) -> str:
        """Generate metrics report."""
        mrr = (
            self.calculate_mrr_at_5(
                [[s] for s in self.mrr_scores[:5]] if self.mrr_scores else []
            )
            if self.mrr_scores
            else 0.0
        )
        quality = self.get_quality_score()
        latency = self.get_latency_p95()
        cost = self.get_avg_cost_per_query()

        criteria = self.check_success_criteria()

        report = f"""
{'=' * 70}
SUCCESS METRICS REPORT (Lines 96-100)
{'=' * 70}

Total Queries: {self.total_queries}

1. MRR@5 (Mean Reciprocal Rank): {mrr:.3f}
   Target: ≥ {settings.TARGET_MRR}  {'✓' if criteria['mrr_at_5'] else '✗'}

2. Quality Score: {quality:.3f}
   Target: ≥ {settings.TARGET_QUALITY}  {'✓' if criteria['quality_score'] else '✗'}

3. Latency (p95): {latency:.3f}s
   Target: < {settings.TARGET_LATENCY_P95}s  {'✓' if criteria['latency_p95'] else '✗'}

4. Cost per Query: ${cost:.4f}
   Target: < ${settings.TARGET_COST_PER_QUERY}  {'✓' if criteria['cost_per_query'] else '✗'}

Overall Status: {'✓ ALL METRICS MET' if all(criteria.values()) else '✗ SOME METRICS NOT MET'}
{'=' * 70}
"""
        return report


# Global metrics instance
system_metrics = SystemMetrics()

