"""Web Search Service using Tavily API via LangChain."""

from typing import List

from pydantic import BaseModel

from app.config import get_settings
from app.core.logging import get_logger

try:
    from langchain_tavily import TavilySearch
except ImportError:
    TavilySearch = None

logger = get_logger(__name__)
settings = get_settings()


class WebSearchResult(BaseModel):
    """Web search result model."""

    title: str
    url: str
    snippet: str
    score: float = 1.0  # Default relevance score


class WebSearchEngine:
    """Web search engine using Tavily API via LangChain."""

    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY
        self.search_tool = None

        if TavilySearch and self.api_key:
            try:
                # TavilySearch uses TAVILY_API_KEY from environment
                # Set it explicitly if not already set
                import os
                if not os.environ.get("TAVILY_API_KEY"):
                    os.environ["TAVILY_API_KEY"] = self.api_key
                
                self.search_tool = TavilySearch(
                    max_results=settings.WEB_SEARCH_NUM_RESULTS,
                )
                logger.info("tavily_client_initialized")
            except Exception as e:
                logger.warning("tavily_client_init_failed", error=str(e), exc_info=True)
        else:
            if not TavilySearch:
                logger.warning("langchain_tavily_package_not_installed")
            if not self.api_key:
                logger.warning("tavily_api_key_not_set")

    def search(
        self, query: str, num_results: int = 5, search_depth: str = "basic"
    ) -> List[WebSearchResult]:
        """
        Perform web search and return results.

        Args:
            query: Search query
            num_results: Number of results to return
            search_depth: "basic" or "advanced" (not used with LangChain integration)

        Returns:
            List of WebSearchResult objects
        """
        if not self.search_tool:
            logger.warning("tavily_client_not_available")
            return []

        try:
            logger.info("tavily_search_starting", query=query[:50])
            
            # LangChain TavilySearch.invoke returns a dict with 'results' key
            response = self.search_tool.invoke({"query": query})

            logger.info(
                "tavily_response_received",
                response_type=type(response).__name__,
                is_dict=isinstance(response, dict),
                response_keys=list(response.keys()) if isinstance(response, dict) else [],
            )

            # TavilySearch returns a dict with 'results' key containing the actual results
            if isinstance(response, dict):
                search_results = response.get("results", [])
                logger.info(
                    "tavily_results_extracted",
                    results_count=len(search_results),
                    response_keys=list(response.keys()),
                )
            elif isinstance(response, list):
                # Fallback: if it's a list, use it directly
                search_results = response
                logger.info("tavily_response_is_list", results_count=len(search_results))
            else:
                logger.error(
                    "unexpected_tavily_response_format",
                    response_type=type(response).__name__,
                    response_preview=str(response)[:500],
                )
                return []

            if not search_results:
                logger.warning("tavily_returned_no_results", query=query[:50])
                return []
            
            # Log first result structure for debugging
            if search_results and len(search_results) > 0:
                logger.info(
                    "tavily_first_result_structure",
                    result_keys=list(search_results[0].keys()) if isinstance(search_results[0], dict) else "Not a dict",
                    first_result_preview=str(search_results[0])[:300],
                )

            results = []
            for idx, result in enumerate(search_results[:num_results]):
                # Calculate relevance score (higher rank = higher score)
                score = 1.0 - (idx * 0.1)  # Decrease score by 0.1 per rank
                score = max(0.1, score)  # Minimum score of 0.1

                # LangChain TavilySearch returns dicts - check actual structure
                # It might return different keys, so try multiple possibilities
                if not isinstance(result, dict):
                    logger.warning(
                        "tavily_result_not_dict",
                        result_type=type(result).__name__,
                        result_preview=str(result)[:200],
                    )
                    continue
                
                # Log all keys in result for debugging
                if idx == 0:
                    logger.info("tavily_result_keys", keys=list(result.keys()))
                
                # Extract fields from the result - try multiple key variations
                title = (
                    result.get("title", "") 
                    or result.get("name", "") 
                    or result.get("title", "")
                    or ""
                )
                url = (
                    result.get("url", "") 
                    or result.get("link", "") 
                    or result.get("href", "")
                    or ""
                )
                snippet = (
                    result.get("content", "")
                    or result.get("snippet", "")
                    or result.get("description", "")
                    or result.get("text", "")
                    or result.get("body", "")
                    or ""
                )

                # If no title, try to extract from URL or use a default
                if not title and url:
                    # Try to extract domain name as title
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        title = parsed.netloc.replace("www.", "") if parsed.netloc else "Web Result"
                    except Exception:
                        title = "Web Result"

                if not url:
                    logger.warning(
                        "incomplete_tavily_result_no_url",
                        result_keys=list(result.keys()),
                    )
                    continue

                results.append(
                    WebSearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        score=score,
                    )
                )

            logger.info(
                "web_search_completed",
                query=query[:50],
                results_count=len(results),
            )
            return results

        except Exception as e:
            logger.error(
                "web_search_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return []

