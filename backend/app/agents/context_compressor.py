"""Context Compression Agent for Token-Aware Context Management."""

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.logging import get_logger
from app.graph.state import RAGState

logger = get_logger(__name__)
settings = get_settings()


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token for English)."""
    return len(text) // 4


def format_context(results: List[dict]) -> str:
    """Format reranked results into context string."""
    context_parts = []
    for i, result in enumerate(results, 1):
        content = result.get("content", "")
        doc_id = result.get("doc_id", "unknown")
        context_parts.append(f"[{i}] {content}\nSource: {doc_id}")
    return "\n\n".join(context_parts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def context_compression_node(state: RAGState) -> RAGState:
    """Compress context if too long, keeping most relevant parts."""
    if not settings.CONTEXT_COMPRESSION_ENABLED:
        logger.debug("context_compression_disabled")
        state["compressed_context"] = state["reranked_results"]
        return state

    try:
        if not state["reranked_results"]:
            logger.warning("no_results_to_compress")
            state["compressed_context"] = []
            return state

        # Format context and estimate tokens
        context_text = format_context(state["reranked_results"])
        token_count = estimate_tokens(context_text)

        # Check if compression is needed
        if token_count <= settings.MAX_CONTEXT_TOKENS:
            logger.debug(
                "context_within_limits",
                token_count=token_count,
                max_tokens=settings.MAX_CONTEXT_TOKENS,
            )
            state["compressed_context"] = state["reranked_results"]
            return state

        logger.info(
            "context_compression_needed",
            token_count=token_count,
            max_tokens=settings.MAX_CONTEXT_TOKENS,
            num_chunks=len(state["reranked_results"]),
        )

        # Keep top K chunks as-is, compress the rest
        top_k = settings.CONTEXT_COMPRESSION_TOP_K
        top_chunks = state["reranked_results"][:top_k]
        remaining_chunks = state["reranked_results"][top_k:]

        if not remaining_chunks:
            state["compressed_context"] = top_chunks
            return state

        # Compress remaining chunks
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.REQUEST_TIMEOUT,
            max_retries=settings.MAX_RETRIES,
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a context compression assistant. Summarize the following document chunks concisely while preserving key information relevant to answering questions.

Original query: {query}

Document chunks to compress:
{chunks}

Instructions:
1. Create a concise summary that preserves all important facts, concepts, and details
2. Maintain the original meaning and context
3. Keep specific numbers, names, dates, and technical terms
4. Combine related information from multiple chunks
5. Target length: approximately {target_tokens} tokens (roughly {target_chars} characters)

Provide a single compressed summary that captures the essential information:
"""
        )

        # Format remaining chunks for compression
        chunks_text = "\n\n---\n\n".join(
            [
                f"Chunk {i+1}:\n{chunk.get('content', '')}"
                for i, chunk in enumerate(remaining_chunks)
            ]
        )

        # Estimate target size (aim for 60% of max tokens to leave room for query)
        target_tokens = int(settings.MAX_CONTEXT_TOKENS * 0.6)
        target_chars = target_tokens * 4  # Rough estimate

        chain = prompt | llm

        compressed_summary = chain.invoke(
            {
                "query": state["query"],
                "chunks": chunks_text,
                "target_tokens": target_tokens,
                "target_chars": target_chars,
            }
        )

        # Extract content from response
        if hasattr(compressed_summary, "content"):
            compressed_text = compressed_summary.content
        else:
            compressed_text = str(compressed_summary)

        # Create compressed chunk entry
        compressed_chunk = {
            "content": compressed_text,
            "doc_id": "compressed",
            "chunk_idx": -1,
            "score": remaining_chunks[-1].get("score", 0.0) if remaining_chunks else 0.0,
            "metadata": {
                "compressed": True,
                "original_chunks": len(remaining_chunks),
            },
        }

        # Combine top chunks with compressed summary
        state["compressed_context"] = top_chunks + [compressed_chunk]

        logger.info(
            "context_compression_completed",
            original_chunks=len(state["reranked_results"]),
            compressed_chunks=len(state["compressed_context"]),
            original_tokens=token_count,
            compressed_tokens=estimate_tokens(format_context(state["compressed_context"])),
        )

        state["metadata"]["context_compression_completed"] = True

    except Exception as e:
        logger.error(
            "context_compression_failed", error=str(e), exc_info=True
        )
        # Fallback to original results
        state["compressed_context"] = state["reranked_results"]
        state["error"] = f"Context compression failed: {str(e)}"
        state["metadata"]["context_compression_failed"] = True

    return state
