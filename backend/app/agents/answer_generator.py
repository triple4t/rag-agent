"""Production LLM Answer Generation Agent."""

import base64
import re
from typing import List, Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.core.logging import get_logger
from app.graph.state import RAGState

logger = get_logger(__name__)
settings = get_settings()


def parse_image_data(image_data: str) -> Dict[str, Any]:
    """
    Parse image data from base64 string or data URL.
    
    Args:
        image_data: Base64-encoded image string or data URL (data:image/...;base64,...)
    
    Returns:
        Dict with 'type' and 'data' keys for LangChain message format
    """
    # Check if it's a data URL
    if image_data.startswith("data:image/"):
        # Extract base64 part from data URL
        match = re.match(r"data:image/([^;]+);base64,(.+)", image_data)
        if match:
            image_type = match.group(1)
            base64_data = match.group(2)
        else:
            # Fallback: try to extract just the base64 part
            base64_data = image_data.split(",", 1)[-1] if "," in image_data else image_data
            image_type = "png"  # Default
    else:
        # Assume it's already base64 encoded
        base64_data = image_data
        image_type = "png"  # Default, could be improved with detection
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/{image_type};base64,{base64_data}" if not base64_data.startswith("data:") else base64_data
        }
    }


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

        # Use compressed context if available, otherwise use reranked results
        context_source = (
            state.get("compressed_context")
            if state.get("compressed_context")
            else state["reranked_results"]
        )

        # Format context from results
        context_parts = []
        for i, result in enumerate(context_source[: settings.TOP_K_RERANK], 1):
            content = result.get("content", "")
            doc_id = result.get("doc_id", "unknown")
            # Check if this is a compressed chunk
            is_compressed = result.get("metadata", {}).get("compressed", False)
            source_label = f"Source: {doc_id}" + (" (compressed)" if is_compressed else "")
            context_parts.append(f"[{i}] {content}\n{source_label}")

        context_text = "\n\n".join(context_parts)

        # Truncate context if too long
        max_context_length = settings.MAX_CONTEXT_TOKENS * 4  # Rough char estimate
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "..."
            logger.warning("context_truncated")

        # Get conversation history if available
        conversation_history = state.get("conversation_history", [])
        conversation_context = ""
        if conversation_history:
            # Format recent conversation history (last 3 exchanges)
            recent_messages = conversation_history[-6:]  # Last 6 messages (3 exchanges)
            context_parts = []
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"User: {content}")
                elif role == "assistant":
                    context_parts.append(f"Assistant: {content}")
            if context_parts:
                conversation_context = "\n".join(context_parts)
        
        # Build the prompt template with optional conversation context
        base_prompt = """You are a helpful assistant that answers questions based on the provided context from multiple documents.

Rules:
1. Answer using information from the provided context
2. For comparison/contrast questions, identify ideas/concepts from different sources and compare them systematically
3. For questions asking to compare across documents, analyze each source and identify similarities and differences
4. If the context doesn't contain enough information to fully answer, explain what you found and what might be missing
5. Cite sources using [1], [2], etc. when referencing specific chunks
6. Be thorough for comparison questions - identify similarities and differences across documents
7. IMPORTANT: If images are provided, analyze them directly using your vision capabilities. 
   Describe what you see in the images in detail, including objects, text, layout, colors, and any other relevant details.
   If the question asks about images, prioritize analyzing the actual images over text descriptions.
8. IMPORTANT: If the context doesn't contain relevant information to answer the question, 
   clearly state: "The provided documents do not contain information about [topic]. 
   The documents appear to be about [what they actually contain based on the context]."
9. Only answer if the context is relevant to the question. If the context is about a completely 
   different topic, acknowledge the mismatch rather than trying to force an answer."""
        
        # Add conversation context section if available
        if conversation_context:
            base_prompt += f"""

9. IMPORTANT: This is a follow-up question in a conversation. Consider the previous conversation context:
{conversation_context}

When answering, you can reference the previous conversation if relevant. If the question is a follow-up (e.g., "is it worth it?", "can I enroll?"), interpret it in the context of what was discussed previously."""
        
        # Get images from state (images sent directly with query)
        images = state.get("images", [])
        
        # Check if query is about images and extract images from search results
        query_lower = state["query"].lower()
        is_image_query = any(keyword in query_lower for keyword in ["image", "images", "picture", "pictures", "photo", "photos", "what is in", "what's in", "explain what", "describe"])
        
        # Extract images from search results if query is about images
        if is_image_query and context_source:
            extracted_count = 0
            for result in context_source[: settings.TOP_K_RERANK]:
                metadata = result.get("metadata", {})
                image_data_url = metadata.get("image_data_url")
                if image_data_url and image_data_url not in images:
                    images.append(image_data_url)
                    extracted_count += 1
                    logger.info("extracted_image_from_search_result", doc_id=result.get("doc_id"), has_image=True)
            if extracted_count > 0:
                logger.info("images_extracted_from_search_results", count=extracted_count, total_images=len(images))
        
        has_images = len(images) > 0
        
        # If images are present, use vision API format
        if has_images:
            logger.info("processing_query_with_images", image_count=len(images))
            
            # Parse images for LangChain message format
            image_contents = [parse_image_data(img) for img in images]
            
            # Build messages with images
            messages = []
            
            # System message
            system_content = base_prompt
            if conversation_context:
                system_content += f"""

9. IMPORTANT: This is a follow-up question in a conversation. Consider the previous conversation context:
{conversation_context}

When answering, you can reference the previous conversation if relevant. If the question is a follow-up (e.g., "is it worth it?", "can I enroll?"), interpret it in the context of what was discussed previously."""
            
            messages.append(SystemMessage(content=system_content))
            
            # User message with text and images
            user_content_parts = []
            
            # Add context if available
            if context_text:
                context_note = ""
                if is_image_query and len(images) > len(state.get("images", [])):
                    context_note = "\n\nNote: Images from the uploaded documents are included below. Please analyze these images directly to answer the question."
                user_content_parts.append({
                    "type": "text",
                    "text": f"Context from documents:\n{context_text}{context_note}\n\nQuestion: {state['query']}\n\nAnswer:"
                })
            else:
                user_content_parts.append({
                    "type": "text",
                    "text": f"Question: {state['query']}\n\nAnswer:"
                })
            
            # Add images
            user_content_parts.extend(image_contents)
            
            messages.append(HumanMessage(content=user_content_parts))
            
            # Invoke LLM with messages
            response = llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
        else:
            # No images - use standard text-only approach
            # Complete the prompt template
            full_prompt_template = f"""{base_prompt}
Context:
{{context}}

Question: {{question}}

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(full_prompt_template)

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

