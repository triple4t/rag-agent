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
from app.search.web_search import WebSearchEngine
from app.api.v1.routes import documents

logger = get_logger(__name__)
settings = get_settings()


class ClassificationResult(BaseModel):
    """Structured output for query classification."""
    route: Literal["general", "rag", "web_search"]
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
    
    # Initialize web search engine
    web_search_engine = None
    if settings.WEB_SEARCH_ENABLED and settings.TAVILY_API_KEY:
        try:
            web_search_engine = WebSearchEngine()
            # Verify search_tool was actually initialized
            if not web_search_engine.search_tool:
                logger.warning(
                    "tavily_client_not_initialized",
                    api_key_present=bool(settings.TAVILY_API_KEY),
                )
                web_search_engine = None
            else:
                logger.info("web_search_engine_initialized_successfully")
        except Exception as e:
            logger.warning(
                "web_search_engine_init_failed",
                error=str(e),
                exc_info=True,
            )
    else:
        logger.info(
            "web_search_disabled",
            enabled=settings.WEB_SEARCH_ENABLED,
            api_key_set=bool(settings.TAVILY_API_KEY),
        )
    
    # General conversation LLM
    general_llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0.7,
        api_key=settings.OPENAI_API_KEY,
    )
    
    def classify_query(state: RouterState) -> dict:
        """Classify query to determine if it needs RAG, web search, or general conversation."""
        docs = documents.get_all_documents()
        has_documents = len(docs) > 0
        
        query_lower = state["query"].lower()
        
        # Keyword-based pre-filter for real-time queries (before LLM classification)
        # This ensures queries about prices, current data, etc. go to web_search
        real_time_keywords = [
            "current price", "current price of", "price of", "price is", "what's the price",
            "latest price", "today's price", "current rate", "current value",
            "what happened", "latest news", "recent news", "today's news", "top news", "current news",
            "news in", "news about", "what's the news", "what is the news",
            "current", "latest", "recent", "today", "now", "this week", "this month",
            "what's happening", "what happened", "breaking", "live", "real-time",
            "stock price", "crypto price", "bitcoin price", "ethereum price",
            "weather", "forecast", "current weather",
        ]
        
        # Check if query contains real-time indicators
        is_real_time_query = any(keyword in query_lower for keyword in real_time_keywords)
        
        # Also check for price-related patterns (e.g., "price of X", "X price")
        price_patterns = ["price of", "price is", "what is the price", "how much is", "what's the price"]
        is_price_query = any(pattern in query_lower for pattern in price_patterns)
        
        # Log keyword detection for debugging
        logger.info(
            "keyword_detection_check",
            query=state["query"][:50],
            is_real_time=is_real_time_query,
            is_price=is_price_query,
            web_search_available=bool(web_search_engine),
            web_search_enabled=settings.WEB_SEARCH_ENABLED,
            tavily_key_set=bool(settings.TAVILY_API_KEY),
        )
        
        # If it's clearly a real-time query, route to web_search if available
        # If web_search is not available but query needs it, we'll still try to route there
        # and let handle_web_search_query handle the fallback
        if is_real_time_query or is_price_query:
            if settings.WEB_SEARCH_ENABLED and settings.TAVILY_API_KEY:
                logger.info(
                    "query_routed_to_web_search_by_keywords",
                    query=state["query"][:50],
                    is_real_time=is_real_time_query,
                    is_price=is_price_query,
                )
                return {
                    "classification": {
                        "route": "web_search",
                        "confidence": 0.95,
                        "reasoning": f"Query contains real-time indicators (keywords: {is_real_time_query}, price: {is_price_query}). Routing to web_search.",
                    }
                }
        
        # Use structured output for classification
        structured_llm = router_llm.with_structured_output(ClassificationResult)
        
        # Build conversation context if available
        conversation_history = state.get("conversation_history", [])
        conversation_context = ""
        if conversation_history:
            # Format conversation history for context
            recent_messages = conversation_history[-4:]  # Last 4 messages (2 exchanges)
            context_parts = []
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"User: {content}")
                elif role == "assistant":
                    context_parts.append(f"Assistant: {content}")
            conversation_context = "\n".join(context_parts)
        
        system_prompt = """You are a query classifier for a RAG (Retrieval-Augmented Generation) system with web search capabilities.

IMPORTANT: Prioritize web_search for ANY query that requires CURRENT, REAL-TIME, or RECENT information.

IMPORTANT: If this is a follow-up question in a conversation, consider the conversation context when classifying.
- If the previous conversation was about documents, follow-up questions are likely also about documents (RAG route)
- If the previous conversation was about general topics, follow-up questions are likely general (general route)
- Only route to web_search if the query explicitly needs current/real-time information

Analyze the user's query and determine if it requires:
1. **RAG route**: Questions about specific documents, content, or information that was uploaded.
   Examples:
   - "What does the document say about X?"
   - "Summarize the uploaded PDF"
   - "What are the key points in the document?"
   - Questions explicitly referencing uploaded files or documents
   
2. **web_search route**: ALWAYS use this for queries about:
   - Current prices, rates, or real-time data (stocks, crypto, currency, etc.)
   - Latest news, recent events, or "what happened" questions
   - Today's, this week's, or recent information
   - Current status, live data, or up-to-date information
   - Questions with words like: "current", "latest", "recent", "today", "now", "this week", "what happened"
   
   Examples (ALWAYS route to web_search):
   - "Current price of Bitcoin" → web_search
   - "What is the price of Bitcoin?" → web_search (implies current price)
   - "Latest news about AI" → web_search
   - "What happened today?" → web_search
   - "Current weather" → web_search
   - "Latest developments in X" → web_search
   - "What's happening with Y?" → web_search
   
3. **General route**: ONLY for static, general knowledge questions that don't need current information.
   Examples:
   - "What is machine learning?" (general concept, not current events)
   - "How does Python work?" (general knowledge)
   - "Tell me a joke" (conversational)
   - "Explain quantum computing" (general knowledge)
   
   DO NOT route to general if the query asks for:
   - Current prices, rates, or real-time data
   - Latest news or recent events
   - Today's information
   - What's happening now

Return your classification with confidence and reasoning. When in doubt between web_search and general, choose web_search if the query could benefit from current information."""

        try:
            # Build user message with conversation context if available
            user_message = state["query"]
            if conversation_context:
                user_message = f"""Previous conversation:
{conversation_context}

Current question: {state["query"]}

Classify the current question considering the conversation context."""
            
            result = structured_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ])
            
            route = result.route
            reasoning = result.reasoning
            
            # Override LLM classification if keywords suggest real-time query
            # Override to web_search if: (1) keywords match AND (2) web_search is enabled
            if (is_real_time_query or is_price_query) and settings.WEB_SEARCH_ENABLED and settings.TAVILY_API_KEY:
                if route != "web_search":
                    logger.info(
                        "overriding_classification_to_web_search",
                        original_route=route,
                        query=state["query"][:50],
                    )
                    route = "web_search"
                    reasoning = f"Overridden from '{result.route}' to 'web_search' due to real-time indicators. Original: {reasoning}"
            
            classification = {
                "route": route,
                "confidence": result.confidence,
                "reasoning": reasoning,
            }
            
            logger.info(
                "query_classified",
                route=classification["route"],
                confidence=classification["confidence"],
                query=state["query"][:50],
                final_reasoning=reasoning[:100],
            )
            
            return {"classification": classification}
            
        except Exception as e:
            logger.error("classification_failed", error=str(e), exc_info=True)
            # If classification fails but keywords suggest web_search, try web_search
            if (is_real_time_query or is_price_query) and settings.WEB_SEARCH_ENABLED and settings.TAVILY_API_KEY:
                logger.info("classification_failed_routing_to_web_search_by_keywords")
                return {
                    "classification": {
                        "route": "web_search",
                        "confidence": 0.9,
                        "reasoning": f"Classification failed, but keywords suggest web_search: {str(e)}",
                    }
                }
            # Default to RAG if classification fails and documents exist, otherwise general
            default_route = "rag" if has_documents else "general"
            return {
                "classification": {
                    "route": default_route,
                    "confidence": 0.5,
                    "reasoning": f"Classification failed, defaulting to {default_route}: {str(e)}",
                }
            }
    
    def route_to_agent(state: RouterState) -> str:
        """Route to appropriate agent based on classification."""
        return state["classification"]["route"]
    
    def handle_general_query(state: RouterState) -> dict:
        """Handle general conversational queries without RAG."""
        try:
            images = state.get("images", [])
            has_images = len(images) > 0
            
            if has_images:
                # Use vision API for images
                from langchain_core.messages import HumanMessage, SystemMessage
                from app.agents.answer_generator import parse_image_data
                
                image_contents = [parse_image_data(img) for img in images]
                
                messages = [
                    SystemMessage(content="""You are a helpful AI assistant. Answer questions conversationally 
                    and provide accurate, helpful information. Analyze any images provided and answer questions about them.
                    If the question is about uploaded documents or specific content, politely suggest that the user 
                    rephrase their question to reference the documents explicitly."""),
                    HumanMessage(content=[
                        {"type": "text", "text": state["query"]},
                        *image_contents
                    ])
                ]
                
                response = general_llm.invoke(messages)
            else:
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
            
            # Create RAG state with new fields
            rag_state: RAGState = {
                "query": state["query"],
                "conversation_history": state.get("conversation_history", []),
                "images": state.get("images", []),
                "expanded_queries": [],
                "vector_results": [],
                "keyword_results": [],
                "hybrid_results": [],
                "reranked_results": [],
                "compressed_context": [],
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
            # First pass: collect all results with scores
            all_results_with_scores = []
            for d in result.get("reranked_results", [])[:settings.TOP_K_RERANK]:
                doc_id = d.get("doc_id", "")
                doc = docs_store.get(doc_id) if doc_id in docs_store else None
                filename = doc.metadata.get("filename", doc_id) if doc else "Unknown"
                
                # Get score with proper fallback: rerank_score > fusion_score > score > 0.0
                # Use None checks instead of 'or' to handle 0.0 values correctly
                rerank_score = d.get("rerank_score")
                fusion_score = d.get("fusion_score")
                original_score = d.get("score")
                
                if rerank_score is not None:
                    score = rerank_score
                elif fusion_score is not None:
                    score = fusion_score
                elif original_score is not None:
                    score = original_score
                else:
                    score = 0.0
                
                all_results_with_scores.append({
                    "doc_id": doc_id,
                    "doc": doc,
                    "filename": filename,
                    "score": score,
                    "data": d,
                })
            
            # Adaptive thresholding: if all scores are below threshold, use relative ranking
            # This handles cases where Cohere reranker scores are low but still relevant
            max_score = max((r["score"] for r in all_results_with_scores), default=0.0)
            
            # If max score is below threshold, use adaptive threshold
            # Keep top results even if absolute scores are low (common with Cohere reranker)
            if max_score < settings.MIN_RELEVANCE_SCORE and max_score > 0:
                # Use adaptive threshold: 50% of max score, but at least 0.005
                # This ensures we keep the most relevant results even with low absolute scores
                adaptive_threshold = max(max_score * 0.5, 0.005)
                logger.info(
                    "using_adaptive_threshold",
                    max_score=max_score,
                    original_threshold=settings.MIN_RELEVANCE_SCORE,
                    adaptive_threshold=adaptive_threshold,
                )
                effective_threshold = adaptive_threshold
            elif max_score == 0:
                # If all scores are 0, keep all results (no filtering)
                effective_threshold = -1  # Negative threshold means accept all
                logger.info("all_scores_zero_keeping_all_results")
            else:
                effective_threshold = settings.MIN_RELEVANCE_SCORE
            
            # Filter results using effective threshold
            sources = []
            for r in all_results_with_scores:
                # Negative threshold means accept all (when all scores are 0)
                if effective_threshold >= 0 and r["score"] < effective_threshold:
                    logger.debug(
                        "filtering_low_score_result",
                        score=r["score"],
                        doc_id=r["doc_id"],
                        threshold=effective_threshold,
                    )
                    continue
                
                # Get page number from metadata
                d = r["data"]
                chunk_metadata = d.get("metadata", {})
                page_number = chunk_metadata.get("page_number") or d.get("page_number")
                
                # Normalize content text to fix spacing issues
                from app.utils.document_loader import normalize_text
                content = d.get("content", "")
                content = normalize_text(content)
                content_preview = content[:200] + "..." if len(content) > 200 else content
                
                sources.append({
                    "doc_id": r["doc_id"],
                    "chunk_idx": d.get("chunk_idx", 0),
                    "page_number": page_number,
                    "filename": r["filename"],
                    "score": float(r["score"]),
                    "content": content_preview,
                })
            
            # If no sources passed the relevance threshold, provide helpful message
            if not sources and result.get("reranked_results"):
                max_score = max(
                    (
                        d.get("rerank_score") or d.get("fusion_score") or d.get("score") or 0.0
                        for d in result.get("reranked_results", [])
                    ),
                    default=0.0,
                )
                logger.warning(
                    "no_sources_passed_threshold",
                    query=state["query"][:50],
                    max_score=max_score,
                    threshold=settings.MIN_RELEVANCE_SCORE,
                )
                return {
                    "answer": f"I couldn't find relevant information in the uploaded documents to answer your question. The documents appear to be about a course syllabus, but your question asks about '{state['query']}'. Please try asking about topics covered in the course syllabus, or upload documents that contain information about your question.",
                    "quality_score": 0.0,
                    "reasoning": f"No sources met the minimum relevance threshold of {settings.MIN_RELEVANCE_SCORE}. Highest score was {max_score:.4f}.",
                    "sources": [],
                    "error": None,
                }
            
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
    
    def handle_web_search_query(state: RouterState) -> dict:
        """Handle web search queries using Tavily."""
        if not web_search_engine:
            # Fallback to general if web search is not available
            logger.warning(
                "web_search_not_available_fallback_to_general",
                tavily_api_key_set=bool(settings.TAVILY_API_KEY),
                web_search_enabled=settings.WEB_SEARCH_ENABLED,
            )
            return handle_general_query(state)

        try:
            # Perform web search
            logger.info("performing_web_search", query=state["query"][:50])
            search_results = web_search_engine.search(
                state["query"],
                num_results=settings.WEB_SEARCH_NUM_RESULTS,
                search_depth=settings.WEB_SEARCH_DEPTH,
            )

            if not search_results:
                logger.warning(
                    "web_search_returned_no_results",
                    query=state["query"][:50],
                )
                return {
                    "answer": "I couldn't find relevant information from web search. This could be due to:\n1. The search query not matching available web content\n2. Temporary API issues\n3. The query requiring very specific or recent information\n\nPlease try rephrasing your query or check if your question requires information from uploaded documents.",
                    "quality_score": 0.0,
                    "reasoning": "No web search results found.",
                    "sources": [],
                }

            # Format context from web results with URLs for markdown linking
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(
                    f"[{i}] {result.title}\n{result.snippet}\nSource URL: {result.url}\n"
                    f"Use this URL in markdown format: [{result.title}]({result.url})"
                )

            context_text = "\n\n".join(context_parts)

            # Check if images are present
            images = state.get("images", [])
            has_images = len(images) > 0
            
            if has_images:
                # Use vision API for images with web search context
                from langchain_core.messages import HumanMessage, SystemMessage
                from app.agents.answer_generator import parse_image_data
                
                image_contents = [parse_image_data(img) for img in images]
                
                messages = [
                    SystemMessage(content="""You are a helpful AI assistant. Answer questions using the provided 
                    web search results and any images provided. Format your response using Markdown for better readability.

IMPORTANT FORMATTING RULES:
1. Use **bold** for key terms, headings, and important information
2. Use markdown links: [link text](URL) for all URLs from sources
3. Use numbered lists (1., 2., 3.) for multiple points or features
4. Use bullet points (- or *) for lists
5. Use ## for section headings when appropriate
6. Cite sources using [1], [2], etc. when referencing specific results, and include the actual link: [Source Name](URL)
7. Make links clickable and visible - always format URLs as [descriptive text](URL)
8. Structure your response clearly with proper headings and sections
9. Use proper spacing and line breaks for readability

Be accurate and provide up-to-date information based on the search results.
If the search results don't fully answer the question, acknowledge what information
is available and what might be missing."""),
                    HumanMessage(content=[
                        {"type": "text", "text": f"Context from web search:\n{context_text}\n\nQuestion: {state['query']}\n\nProvide a well-formatted answer using Markdown. Include clickable links using [link text](URL) format for all sources referenced. Use headings, lists, and bold text to make the response clear and readable."},
                        *image_contents
                    ])
                ]
                
                response = general_llm.invoke(messages)
            else:
                # Generate answer using LLM with web context (no images)
                response = general_llm.invoke(
                    [
                        {
                            "role": "system",
                            "content": """You are a helpful AI assistant. Answer questions using the provided 
                            web search results. Format your response using Markdown for better readability.

IMPORTANT FORMATTING RULES:
1. Use **bold** for key terms, headings, and important information
2. Use markdown links: [link text](URL) for all URLs from sources
3. Use numbered lists (1., 2., 3.) for multiple points or features
4. Use bullet points (- or *) for lists
5. Use ## for section headings when appropriate
6. Cite sources using [1], [2], etc. when referencing specific results, and include the actual link: [Source Name](URL)
7. Make links clickable and visible - always format URLs as [descriptive text](URL)
8. Structure your response clearly with proper headings and sections
9. Use proper spacing and line breaks for readability

Be accurate and provide up-to-date information based on the search results.
If the search results don't fully answer the question, acknowledge what information
is available and what might be missing.""",
                        },
                        {
                            "role": "user",
                            "content": f"Context from web search:\n{context_text}\n\nQuestion: {state['query']}\n\nProvide a well-formatted answer using Markdown. Include clickable links using [link text](URL) format for all sources referenced. Use headings, lists, and bold text to make the response clear and readable.",
                        },
                    ]
                )

            answer = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Format sources for frontend
            sources = [
                {
                    "doc_id": result.url,
                    "chunk_idx": idx,
                    "filename": result.title,
                    "score": result.score,
                    "content": result.snippet,
                }
                for idx, result in enumerate(search_results)
            ]

            logger.info(
                "web_search_query_processed",
                query=state["query"][:50],
                sources_count=len(sources),
            )

            return {
                "answer": answer,
                "quality_score": 0.85,  # Good quality for web search
                "reasoning": f"Answer generated from {len(sources)} web search results.",
                "sources": sources,
            }

        except Exception as e:
            logger.error("web_search_query_failed", error=str(e), exc_info=True)
            return {
                "answer": "I apologize, but I encountered an error performing web search. Please try again or rephrase your query.",
                "quality_score": 0.0,
                "reasoning": f"Web search failed: {str(e)}",
                "sources": [],
                "error": str(e),
            }

    # Build router graph
    graph = StateGraph(RouterState)
    
    # Add nodes
    graph.add_node("classify", classify_query)
    graph.add_node("general", handle_general_query)
    graph.add_node("rag", handle_rag_query)
    graph.add_node("web_search", handle_web_search_query)

    # Add edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_to_agent,
        {
            "general": "general",
            "rag": "rag",
            "web_search": "web_search",
        },
    )
    graph.add_edge("general", END)
    graph.add_edge("rag", END)
    graph.add_edge("web_search", END)
    
    return graph.compile()

