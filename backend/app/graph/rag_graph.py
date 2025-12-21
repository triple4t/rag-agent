"""Production LangGraph RAG System Workflow."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.answer_generator import answer_generation_node
from app.agents.quality_reflector import quality_reflection_node
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
    """Build production LangGraph RAG system."""

    def hybrid_search_node(state: RAGState) -> RAGState:
        """Hybrid search node with error handling."""
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
            state["reranked_results"] = [
                {**doc, "score": doc.get("score") or doc.get("fusion_score") or 0.0}
                for doc in state["hybrid_results"][: settings.TOP_K_RERANK]
            ]
        return state

    # Build graph
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("search", hybrid_search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("answer", answer_generation_node)
    graph.add_node("reflect", quality_reflection_node)

    # Add edges
    graph.add_edge(START, "search")
    graph.add_edge("search", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", "reflect")
    graph.add_edge("reflect", END)

    # Compile with optional checkpointing
    if enable_checkpointing:
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)

    return graph.compile()

