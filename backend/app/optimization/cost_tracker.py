"""Cost Tracking per Query (from roadmap line 60)."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Pricing (as of 2024, update as needed)
PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},  # per token
    "gpt-4": {"input": 30.00 / 1_000_000, "output": 60.00 / 1_000_000},
    "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    "text-embedding-3-small": 0.02 / 1_000_000,  # per token
    "text-embedding-3-large": 0.13 / 1_000_000,
    "cohere-rerank": 1.00 / 1_000,  # per document
}


@dataclass
class CostEntry:
    """Single cost entry."""

    service: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class CostTracker:
    """Track costs across the system."""

    def __init__(self):
        self.entries: List[CostEntry] = []
        self.total_cost: float = 0.0

    def track_llm_call(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Track LLM API call cost."""
        if model not in PRICING:
            logger.warning("unknown_model_pricing", model=model)
            return 0.0

        pricing = PRICING[model]
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

        entry = CostEntry(
            service=f"llm-{model}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

        self.entries.append(entry)
        self.total_cost += cost

        logger.debug(
            "cost_tracked",
            service=entry.service,
            cost=cost,
            total_cost=self.total_cost,
        )

        return cost

    def track_embedding(self, model: str, tokens: int) -> float:
        """Track embedding cost."""
        if model not in PRICING:
            logger.warning("unknown_embedding_model", model=model)
            return 0.0

        cost = tokens * PRICING[model]
        entry = CostEntry(service=f"embedding-{model}", input_tokens=tokens, cost=cost)
        self.entries.append(entry)
        self.total_cost += cost

        logger.debug(
            "embedding_cost_tracked",
            model=model,
            tokens=tokens,
            cost=cost,
        )

        return cost

    def track_rerank(self, num_documents: int) -> float:
        """Track reranking cost."""
        cost = num_documents * PRICING["cohere-rerank"]
        entry = CostEntry(service="rerank", input_tokens=num_documents, cost=cost)
        self.entries.append(entry)
        self.total_cost += cost

        logger.debug("rerank_cost_tracked", documents=num_documents, cost=cost)

        return cost

    def get_cost_per_query(self) -> float:
        """Get average cost per query."""
        if not self.entries:
            return 0.0

        # Group by date to get per-query average
        unique_dates = set(e.timestamp.date() for e in self.entries)
        if not unique_dates:
            return 0.0

        return self.total_cost / len(unique_dates)

    def get_report(self) -> str:
        """Generate cost report."""
        return f"""
Cost Tracking Report:
  Total Cost: ${self.total_cost:.4f}
  Total Calls: {len(self.entries)}
  Avg Cost per Query: ${self.get_cost_per_query():.4f}
  
Breakdown:
  LLM Calls: {len([e for e in self.entries if e.service.startswith('llm')])}
  Embedding Calls: {len([e for e in self.entries if e.service.startswith('embedding')])}
  Rerank Calls: {len([e for e in self.entries if e.service == 'rerank'])}
"""


# Global cost tracker instance
cost_tracker = CostTracker()

