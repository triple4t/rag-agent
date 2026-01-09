"""Query Endpoints."""

import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.core.exceptions import SearchError
from app.core.logging import get_logger
from app.graph.router_graph import build_router_graph
from app.graph.router_state import RouterState
from app.optimization.caching import cache_manager
from app.optimization.cost_tracker import cost_tracker
from app.search.hybrid_search import HybridSearchEngine
from app.search.reranker import RerankerAgent
from app.utils.metrics import QueryMetrics, system_metrics
from app.api.v1.routes import documents

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# Global search engine, reranker, and router graph (initialized on first query)
search_engine: Optional[HybridSearchEngine] = None
reranker: Optional[RerankerAgent] = None
router_graph = None
_last_document_count: int = 0


def _initialize_system():
    """Initialize search engine and router graph - reinitializes if documents changed."""
    global search_engine, reranker, router_graph, _last_document_count

    docs = documents.get_all_documents()
    current_doc_count = len(docs)
    
    # Initialize or reinitialize if document count changed
    if search_engine is None or current_doc_count != _last_document_count:
        if current_doc_count > 0:
            logger.info(
                "initializing_search_engine",
                document_count=current_doc_count,
                previous_count=_last_document_count,
            )
            search_engine = HybridSearchEngine(docs)
            reranker = RerankerAgent()
            
            # Clear cache when documents change
            if _last_document_count > 0 and current_doc_count != _last_document_count:
                cache_manager.clear()
                logger.info("cache_cleared_due_to_document_change")
        else:
            # No documents, but still initialize router (will route to general)
            search_engine = None
            reranker = None
            logger.info("no_documents_available_routing_to_general")
        
        # Build router graph (works with or without documents)
        router_graph = build_router_graph(search_engine, reranker)
        _last_document_count = current_doc_count
    elif router_graph is None:
        # Router not initialized yet
        router_graph = build_router_graph(search_engine, reranker)


class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    chat_id: Optional[int] = None  # Optional chat ID for conversation context
    conversation_history: Optional[List[dict]] = None  # Optional conversation history: [{"role": "user", "content": "..."}, ...]
    images: Optional[List[str]] = None  # Optional list of base64-encoded images (data URLs or base64 strings)


class QueryResponse(BaseModel):
    """Query response model."""

    query_id: str
    query: str
    answer: str
    quality_score: float
    reasoning: str
    sources: List[dict]  # Now includes: doc_id, chunk_idx, filename, score, content
    latency: float
    cost: float
    error: Optional[str] = None


@router.post("", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """Submit query to RAG system."""
    query_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Initialize system if needed
        _initialize_system()

        # Quick check for real-time keywords to skip cache for web_search queries
        query_lower = request.query.lower()
        real_time_indicators = [
            "current", "latest", "recent", "today", "now", "this week",
            "price of", "price is", "what's the price",
            "news", "what happened", "what's happening", "breaking",
            "weather", "forecast", "stock price", "crypto price"
        ]
        is_likely_realtime = any(indicator in query_lower for indicator in real_time_indicators)
        
        # Only check cache if it's not likely a real-time query
        # (web_search queries shouldn't be cached as they're real-time)
        if not is_likely_realtime:
            # Try semantic cache first, then exact match
            cached_result = cache_manager.get_semantic(query=request.query)
            is_semantic_hit = cached_result is not None
            if not cached_result:
                cached_result = cache_manager.get(query=request.query)
            
            if cached_result:
                logger.info("cache_hit", query_id=query_id, semantic=is_semantic_hit)
                return QueryResponse(**cached_result)

        # Load conversation history if chat_id is provided
        conversation_history = request.conversation_history or []
        if request.chat_id and not conversation_history:
            # Try to load from database if chat_id provided but no history
            try:
                from app.database.session import get_db
                from app.models.database import Chat, Message
                from sqlalchemy import select, asc
                from sqlalchemy.ext.asyncio import AsyncSession
                
                # Note: This is a sync function, so we can't use async DB directly
                # For now, rely on conversation_history being passed from frontend
                logger.debug("chat_id_provided_but_no_history", chat_id=request.chat_id)
            except Exception as e:
                logger.debug("could_not_load_chat_history", chat_id=request.chat_id, error=str(e))
        
        # Create initial router state
        initial_state: RouterState = {
            "query": request.query,
            "conversation_history": conversation_history,
            "images": request.images or [],  # Include images in state
            "classification": {
                "route": "general",
                "confidence": 0.0,
                "reasoning": "",
            },
            "answer": "",
            "quality_score": 0.0,
            "reasoning": "",
            "sources": [],
            "latency": 0.0,
            "cost": 0.0,
            "error": None,
            "metadata": {},
        }

        # Execute router graph
        result = router_graph.invoke(initial_state)
        
        # Get the route from classification
        route = result.get("classification", {}).get("route", "general")

        # Calculate metrics
        latency = time.time() - start_time
        cost = cost_tracker.get_cost_per_query()

        # Track metrics
        query_metrics = QueryMetrics(
            query=request.query,
            latency=latency,
            quality_score=result.get("quality_score", 0.0),
            cost=cost,
        )
        system_metrics.add_query(query_metrics)

        # Prepare response
        response = QueryResponse(
            query_id=query_id,
            query=request.query,
            answer=result.get("answer", ""),
            quality_score=result.get("quality_score", 0.0),
            reasoning=result.get("reasoning", ""),
            sources=result.get("sources", []),
            latency=latency,
            cost=cost,
            error=result.get("error"),
        )

        # Only cache non-web_search queries (web_search results are real-time and shouldn't be cached)
        if route != "web_search":
            cache_manager.set(
                query=request.query,
                value=response.dict(),
                enable_semantic=settings.SEMANTIC_CACHE_ENABLED,
            )
        else:
            logger.debug("skipping_cache_for_web_search", query=request.query[:50])

        logger.info(
            "query_processed",
            query_id=query_id,
            route=result.get("classification", {}).get("route", "unknown"),
            latency=latency,
            quality_score=result.get("quality_score", 0.0),
        )

        return response

    except Exception as e:
        logger.error("query_processing_failed", query_id=query_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{query_id}")
async def get_query_result(query_id: str):
    """Get query result by ID."""
    # In production, store query results in database
    raise HTTPException(
        status_code=501, detail="Query result retrieval not yet implemented"
    )

