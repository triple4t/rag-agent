"""RAG State Definitions for LangGraph."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class RAGState(TypedDict):
    """Shared state across all RAG agents."""

    query: str
    vector_results: List[Dict[str, Any]]
    keyword_results: List[Dict[str, Any]]
    hybrid_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    final_answer: str
    quality_score: float
    reasoning: str
    metadata: Dict[str, Any]
    error: Optional[str]

