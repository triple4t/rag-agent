"""Query Expansion Agent for Generating Multiple Query Variations."""

import json
from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.logging import get_logger
from app.graph.state import RAGState
from app.search.hybrid_search import HybridSearchEngine

logger = get_logger(__name__)
settings = get_settings()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def query_expansion_node(
    state: RAGState, search_engine: HybridSearchEngine
) -> RAGState:
    """Generate multiple query variations for better retrieval."""
    if not settings.QUERY_EXPANSION_ENABLED:
        logger.debug("query_expansion_disabled")
        state["expanded_queries"] = [state["query"]]
        return state

    try:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.7,  # Higher temperature for variation
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.REQUEST_TIMEOUT,
            max_retries=settings.MAX_RETRIES,
        )

        parser = JsonOutputParser()

        prompt = ChatPromptTemplate.from_template(
            """
Generate {num_variations} variations of this query to improve search results.

Original query: {query}

Create variations that:
1. Paraphrase the original query using different wording
2. Extract key entities/concepts and create entity-focused queries
3. Create broader or more specific versions when appropriate
4. Use synonyms and related terms

Return as JSON array of strings: ["variation1", "variation2", "variation3"]

Example:
Original: "What are the benefits of machine learning?"
Variations: ["What advantages does machine learning provide?", "How does machine learning help?", "What are the pros of ML?"]

{format_instructions}
"""
        )

        chain = prompt | llm | parser

        result = chain.invoke(
            {
                "query": state["query"],
                "num_variations": settings.QUERY_EXPANSION_NUM_VARIATIONS,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # Extract variations from result
        if isinstance(result, dict):
            variations = result.get("variations", result.get("queries", []))
        elif isinstance(result, list):
            variations = result
        else:
            variations = [state["query"]]

        # Ensure we have at least the original query
        if not variations or len(variations) == 0:
            variations = [state["query"]]

        # Limit to configured number
        variations = variations[: settings.QUERY_EXPANSION_NUM_VARIATIONS]

        # Always include original query as first variation
        if state["query"] not in variations:
            variations = [state["query"]] + variations[: settings.QUERY_EXPANSION_NUM_VARIATIONS - 1]

        state["expanded_queries"] = variations
        state["metadata"]["query_expansion_completed"] = True
        logger.info(
            "query_expansion_completed",
            original_query=state["query"][:50],
            num_variations=len(variations),
        )

        # Perform hybrid search with all variations and combine results
        all_results = []
        seen_keys = set()

        for variation in variations:
            try:
                results = search_engine.hybrid_search(
                    variation, k=settings.TOP_K_SEARCH
                )
                # Deduplicate by doc_id and chunk_idx
                for result in results:
                    key = f"{result.get('doc_id', '')}_{result.get('chunk_idx', 0)}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_results.append(result)
            except Exception as e:
                logger.warning(
                    "search_failed_for_variation",
                    variation=variation[:50],
                    error=str(e),
                )
                continue

        # Sort by score (highest first)
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Update hybrid_results with expanded search results
        state["hybrid_results"] = all_results[: settings.TOP_K_SEARCH * 2]  # Get more for reranking
        state["metadata"]["expanded_search_completed"] = True

        logger.info(
            "expanded_search_completed",
            total_results=len(all_results),
            unique_results=len(state["hybrid_results"]),
        )

    except json.JSONDecodeError as e:
        logger.error("json_parse_error_in_expansion", error=str(e))
        # Fallback to original query
        state["expanded_queries"] = [state["query"]]
        state["metadata"]["query_expansion_failed"] = True
    except Exception as e:
        logger.error(
            "query_expansion_failed", error=str(e), exc_info=True
        )
        # Fallback to original query
        state["expanded_queries"] = [state["query"]]
        state["error"] = f"Query expansion failed: {str(e)}"
        state["metadata"]["query_expansion_failed"] = True

    return state
