"""RAG State Definitions for LangGraph."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class RAGState(TypedDict):
    """Shared state across all RAG agents."""

    query: str
    conversation_history: List[Dict[str, Any]]  # Conversation history for context
    images: List[str]  # List of base64-encoded images (data URLs or base64 strings)
    expanded_queries: List[str]  # Query expansion variations
    vector_results: List[Dict[str, Any]]
    keyword_results: List[Dict[str, Any]]
    hybrid_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    compressed_context: List[Dict[str, Any]]  # Compressed context for answer generation
    final_answer: str
    quality_score: float
    reasoning: str
    metadata: Dict[str, Any]
    error: Optional[str]

