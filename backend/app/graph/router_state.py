"""Router State Definitions for Query Classification and Routing."""

from typing import Literal
from typing_extensions import TypedDict


class Classification(TypedDict):
    """Query classification result."""
    route: Literal["general", "rag", "web_search"]
    confidence: float
    reasoning: str


class RouterState(TypedDict):
    """Main router state that tracks query, classification, and results."""
    query: str
    conversation_history: list[dict]  # List of previous messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    images: list[str]  # List of base64-encoded images (data URLs or base64 strings)
    classification: Classification
    answer: str
    quality_score: float
    reasoning: str
    sources: list[dict]
    latency: float
    cost: float
    error: str | None
    metadata: dict

