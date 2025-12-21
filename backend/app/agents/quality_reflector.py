"""Quality Reflection Agent for Answer Scoring."""

import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.logging import get_logger
from app.graph.state import RAGState

logger = get_logger(__name__)
settings = get_settings()


def quality_reflection_node(state: RAGState) -> RAGState:
    """Score answer quality and reasoning using LangChain."""
    try:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.REQUEST_TIMEOUT,
            max_retries=settings.MAX_RETRIES,
        )

        parser = JsonOutputParser()

        prompt = ChatPromptTemplate.from_template(
            """
Evaluate this RAG answer on a scale of 0-1:

Question: {question}
Answer: {answer}

Criteria:
- Is the answer directly answering the question? (0.3 weight)
- Is it grounded in the provided context? (0.4 weight)
- Is it clear and well-structured? (0.3 weight)

Respond in JSON format with "score" (float) and "reasoning" (string).
{format_instructions}
"""
        )

        chain = prompt | llm | parser

        result = chain.invoke(
            {
                "question": state["query"],
                "answer": state["final_answer"],
                "format_instructions": parser.get_format_instructions(),
            }
        )

        state["quality_score"] = float(result.get("score", 0.5))
        state["reasoning"] = result.get("reasoning", "Evaluation completed")
        state["metadata"]["quality_evaluated"] = True

        logger.info(
            "quality_evaluation_completed",
            score=state["quality_score"],
        )

    except json.JSONDecodeError as e:
        logger.error("json_parse_error", error=str(e))
        state["quality_score"] = 0.5
        state["reasoning"] = "Evaluation failed: JSON parsing error"
    except Exception as e:
        logger.error("quality_evaluation_failed", error=str(e), exc_info=True)
        state["quality_score"] = 0.5
        state["reasoning"] = f"Evaluation failed: {str(e)}"

    return state

