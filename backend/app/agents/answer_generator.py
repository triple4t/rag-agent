"""Production LLM Answer Generation Agent."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger
from app.graph.state import RAGState

logger = get_logger(__name__)
settings = get_settings()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def answer_generation_node(state: RAGState) -> RAGState:
    """Generate answer using LangChain with production error handling."""
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.REQUEST_TIMEOUT,
            max_retries=settings.MAX_RETRIES,
        )

        # Format context from reranked results
        context_parts = []
        for i, result in enumerate(
            state["reranked_results"][: settings.TOP_K_RERANK], 1
        ):
            content = result.get("content", "")
            doc_id = result.get("doc_id", "unknown")
            context_parts.append(f"[{i}] {content}\nSource: {doc_id}")

        context_text = "\n\n".join(context_parts)

        # Truncate context if too long
        max_context_length = settings.MAX_CONTEXT_TOKENS * 4  # Rough char estimate
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "..."
            logger.warning("context_truncated")

        # Create prompt
        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant that answers questions based on the provided context from multiple documents.

Rules:
1. Answer using information from the provided context
2. For comparison/contrast questions, identify ideas/concepts from different sources and compare them systematically
3. For questions asking to compare across documents, analyze each source and identify similarities and differences
4. If the context doesn't contain enough information to fully answer, explain what you found and what might be missing
5. Cite sources using [1], [2], etc. when referencing specific chunks
6. Be thorough for comparison questions - identify similarities and differences across documents
7. If you cannot find relevant information, say "I cannot find this information in the provided documents" only as a last resort

Context:
{context}

Question: {question}

Answer:"""
        )

        # Create chain
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Invoke chain
        answer = chain.invoke(
            {
                "context": context_text,
                "question": state["query"],
            }
        )

        state["final_answer"] = answer
        state["metadata"]["answer_generated"] = True
        logger.info("answer_generated_successfully")

    except Exception as e:
        logger.error("answer_generation_failed", error=str(e), exc_info=True)
        raise LLMGenerationError(
            f"Answer generation failed: {str(e)}",
            details={"error_type": type(e).__name__},
        )

    return state

