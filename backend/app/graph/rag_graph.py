"""Production LangGraph RAG System Workflow with Query Expansion and Context Compression."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.answer_generator import answer_generation_node
from app.agents.context_compressor import context_compression_node
from app.agents.quality_reflector import quality_reflection_node
from app.agents.query_expander import query_expansion_node
from app.config import get_settings
from app.core.logging import get_logger
from app.graph.state import RAGState
from app.search.hybrid_search import HybridSearchEngine
from app.search.reranker import RerankerAgent

logger = get_logger(__name__)
settings = get_settings()


def build_rag_graph(
    search_engine: HybridSearchEngine,
    reranker: RerankerAgent,
    enable_checkpointing: bool = False,
):
    """Build production LangGraph RAG system with query expansion and context compression."""

    def initialize_state(state: RAGState) -> RAGState:
        """Initialize state with default values for new fields."""
        if "expanded_queries" not in state:
            state["expanded_queries"] = []
        if "compressed_context" not in state:
            state["compressed_context"] = []
        if "conversation_history" not in state:
            state["conversation_history"] = []
        if "images" not in state:
            state["images"] = []
        return state

    def query_expansion_wrapper(state: RAGState) -> RAGState:
        """Wrapper for query expansion node with search engine."""
        return query_expansion_node(state, search_engine)

    def hybrid_search_node(state: RAGState) -> RAGState:
        """Hybrid search node with error handling (fallback if expansion disabled)."""
        # If query expansion already ran, skip this
        if state.get("hybrid_results") and len(state.get("hybrid_results", [])) > 0:
            logger.debug("skipping_hybrid_search_expansion_already_ran")
            return state

        try:
            results = search_engine.hybrid_search(
                state["query"], k=settings.TOP_K_SEARCH
            )
            state["hybrid_results"] = results
            state["metadata"]["search_completed"] = True
            logger.info("hybrid_search_completed", results_count=len(results))
        except Exception as e:
            logger.error("search_failed", error=str(e), exc_info=True)
            state["error"] = f"Search failed: {str(e)}"
            state["hybrid_results"] = []
        return state

    def rerank_node(state: RAGState) -> RAGState:
        """Rerank node with error handling."""
        try:
            if not state["hybrid_results"]:
                logger.warning("no_results_to_rerank")
                state["reranked_results"] = []
                return state

            reranked = reranker.rerank(
                state["query"],
                state["hybrid_results"],
                top_k=settings.TOP_K_RERANK,
            )
            state["reranked_results"] = reranked
            state["metadata"]["rerank_completed"] = True
            logger.info("rerank_completed", results_count=len(reranked))
        except Exception as e:
            logger.error("reranking_failed", error=str(e), exc_info=True)
            state["error"] = f"Reranking failed: {str(e)}"
            # Preserve scores from hybrid search when reranking fails
            # Use None checks instead of 'or' to handle 0.0 values correctly
            state["reranked_results"] = []
            for doc in state["hybrid_results"][: settings.TOP_K_RERANK]:
                fusion_score = doc.get("fusion_score")
                original_score = doc.get("score")
                score = fusion_score if fusion_score is not None else (original_score if original_score is not None else 0.0)
                state["reranked_results"].append({**doc, "score": score})
        return state

    # Build graph
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("initialize", initialize_state)
    
    # Query expansion or direct search
    if settings.QUERY_EXPANSION_ENABLED:
        graph.add_node("expand", query_expansion_wrapper)
    else:
        graph.add_node("search", hybrid_search_node)
    
    graph.add_node("rerank", rerank_node)
    
    # Context compression
    if settings.CONTEXT_COMPRESSION_ENABLED:
        graph.add_node("compress", context_compression_node)
    
    graph.add_node("answer", answer_generation_node)
    graph.add_node("reflect", quality_reflection_node)

    # Add edges
    graph.add_edge(START, "initialize")
    
    # Route from initialize to expansion or search
    if settings.QUERY_EXPANSION_ENABLED:
        graph.add_edge("initialize", "expand")
        graph.add_edge("expand", "rerank")
    else:
        graph.add_edge("initialize", "search")
        graph.add_edge("search", "rerank")
    
    # Route from rerank to compression or answer
    if settings.CONTEXT_COMPRESSION_ENABLED:
        graph.add_edge("rerank", "compress")
        graph.add_edge("compress", "answer")
    else:
        graph.add_edge("rerank", "answer")
    
    graph.add_edge("answer", "reflect")
    graph.add_edge("reflect", END)

    # Compile with optional checkpointing
    if enable_checkpointing:
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)

    return graph.compile()

