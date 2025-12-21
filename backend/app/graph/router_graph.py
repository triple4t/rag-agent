"""Router Graph for Query Classification and Routing to General or RAG Agents."""

from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.config import get_settings
from app.core.logging import get_logger
from app.graph.router_state import RouterState, Classification
from app.graph.rag_graph import build_rag_graph
from app.search.hybrid_search import HybridSearchEngine
from app.search.reranker import RerankerAgent
from app.api.v1.routes import documents

logger = get_logger(__name__)
settings = get_settings()


class ClassificationResult(BaseModel):
    """Structured output for query classification."""
    route: Literal["general", "rag"]
    confidence: float
    reasoning: str


def build_router_graph(
    search_engine: HybridSearchEngine | None,
    reranker: RerankerAgent | None,
):
    """Build router graph that classifies queries and routes to appropriate agent."""
    
    # Initialize LLM for classification
    router_llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0.0,
        api_key=settings.OPENAI_API_KEY,
    )
    
    # Get RAG graph if documents are available
    rag_graph = None
    if search_engine and reranker:
        rag_graph = build_rag_graph(search_engine, reranker)
    
    # General conversation LLM
    general_llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
    )
    
    def classify_query(state: RouterState) -> dict:
        """Classify query to determine if it needs RAG or general conversation."""
        docs = documents.get_all_documents()
        has_documents = len(docs) > 0
        
        # If no documents, always route to general
        if not has_documents:
            return {
                "classification": {
                    "route": "general",
                    "confidence": 1.0,
                    "reasoning": "No documents available, using general conversation.",
                }
            }
        
        # Use structured output for classification
        structured_llm = router_llm.with_structured_output(ClassificationResult)
        
        system_prompt = """You are a query classifier for a RAG (Retrieval-Augmented Generation) system.

Analyze the user's query and determine if it requires:
1. **RAG route**: Questions about specific documents, content, or information that was uploaded.
   Examples:
   - "What does the document say about X?"
   - "Summarize the uploaded PDF"
   - "What are the key points in the document?"
   - Questions referencing specific content, data, or details from uploaded files
   
2. **General route**: Conversational questions, general knowledge, or questions not related to uploaded documents.
   Examples:
   - "What is machine learning?"
   - "How does Python work?"
   - "Tell me a joke"
   - General questions that don't reference uploaded content

Return your classification with confidence and reasoning."""

        try:
            result = structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["query"]}
            ])
            
            classification = {
                "route": result.route,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            }
            
            logger.info(
                "query_classified",
                route=classification["route"],
                confidence=classification["confidence"],
                query=state["query"][:50],
            )
            
            return {"classification": classification}
            
        except Exception as e:
            logger.error("classification_failed", error=str(e), exc_info=True)
            # Default to RAG if classification fails and documents exist
            return {
                "classification": {
                    "route": "rag" if has_documents else "general",
                    "confidence": 0.5,
                    "reasoning": f"Classification failed, defaulting to {'rag' if has_documents else 'general'}: {str(e)}",
                }
            }
    
    def route_to_agent(state: RouterState) -> str:
        """Route to appropriate agent based on classification."""
        return state["classification"]["route"]
    
    def handle_general_query(state: RouterState) -> dict:
        """Handle general conversational queries without RAG."""
        try:
            response = general_llm.invoke([
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant. Answer questions conversationally 
                    and provide accurate, helpful information. If the question is about uploaded 
                    documents or specific content, politely suggest that the user rephrase their 
                    question to reference the documents explicitly.""",
                },
                {"role": "user", "content": state["query"]}
            ])
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("general_query_processed", query=state["query"][:50])
            
            return {
                "answer": answer,
                "quality_score": 0.8,  # Default quality for general queries
                "reasoning": "General conversational response without document search.",
                "sources": [],
            }
            
        except Exception as e:
            logger.error("general_query_failed", error=str(e), exc_info=True)
            return {
                "answer": "I apologize, but I encountered an error processing your question.",
                "quality_score": 0.0,
                "reasoning": f"Error: {str(e)}",
                "sources": [],
                "error": str(e),
            }
    
    def handle_rag_query(state: RouterState) -> dict:
        """Handle RAG queries using document search."""
        if not rag_graph:
            return {
                "answer": "I apologize, but the document search system is not available. Please upload documents first.",
                "quality_score": 0.0,
                "reasoning": "RAG system not initialized.",
                "sources": [],
                "error": "RAG system not initialized",
            }
        
        try:
            from app.graph.state import RAGState
            
            # Create RAG state
            rag_state: RAGState = {
                "query": state["query"],
                "vector_results": [],
                "keyword_results": [],
                "hybrid_results": [],
                "reranked_results": [],
                "final_answer": "",
                "quality_score": 0.0,
                "reasoning": "",
                "metadata": {},
                "error": None,
            }
            
            # Execute RAG graph
            result = rag_graph.invoke(rag_state)
            
            # Get document store for filename mapping
            docs_store = documents.documents_store
            
            # Format sources with filenames
            # Use all reranked results (up to TOP_K_RERANK) for better context
            sources = []
            for d in result.get("reranked_results", [])[:settings.TOP_K_RERANK]:
                doc_id = d.get("doc_id", "")
                doc = docs_store.get(doc_id) if doc_id in docs_store else None
                filename = doc.metadata.get("filename", doc_id) if doc else "Unknown"
                
                # Get score with proper fallback: rerank_score > fusion_score > score > 0.0
                score = (
                    d.get("rerank_score") 
                    or d.get("fusion_score") 
                    or d.get("score") 
                    or 0.0
                )
                
                sources.append({
                    "doc_id": doc_id,
                    "chunk_idx": d.get("chunk_idx", 0),
                    "filename": filename,
                    "score": float(score),
                    "content": d.get("content", "")[:200] + "..." if len(d.get("content", "")) > 200 else d.get("content", ""),
                })
            
            logger.info("rag_query_processed", query=state["query"][:50], sources_count=len(sources))
            
            return {
                "answer": result.get("final_answer", ""),
                "quality_score": result.get("quality_score", 0.0),
                "reasoning": result.get("reasoning", ""),
                "sources": sources,
                "error": result.get("error"),
            }
            
        except Exception as e:
            logger.error("rag_query_failed", error=str(e), exc_info=True)
            return {
                "answer": "I apologize, but I encountered an error searching the documents.",
                "quality_score": 0.0,
                "reasoning": f"RAG search failed: {str(e)}",
                "sources": [],
                "error": str(e),
            }
    
    # Build router graph
    graph = StateGraph(RouterState)
    
    # Add nodes
    graph.add_node("classify", classify_query)
    graph.add_node("general", handle_general_query)
    graph.add_node("rag", handle_rag_query)
    
    # Add edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_to_agent,
        {
            "general": "general",
            "rag": "rag",
        }
    )
    graph.add_edge("general", END)
    graph.add_edge("rag", END)
    
    return graph.compile()

