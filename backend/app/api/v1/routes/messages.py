"""Message endpoints for loading conversation history."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.dependencies.auth import get_current_active_user
from app.graph.router_graph import build_router_graph
from app.models.database import User, Chat
from app.search.hybrid_search import HybridSearchEngine
from app.search.reranker import RerankerAgent
from app.api.v1.routes import documents

router = APIRouter(prefix="/messages", tags=["messages"])

# Global router graph for accessing checkpointer
router_graph = None
checkpointer: MemorySaver | None = None


def _get_router_graph():
    """Get router graph with checkpointer."""
    global router_graph, checkpointer
    if router_graph is None:
        from app.api.v1.routes.queries import _initialize_system
        _initialize_system()
        # Get checkpointer from queries module
        from app.api.v1.routes.queries import checkpointer as cp
        checkpointer = cp
    return router_graph, checkpointer


@router.get("/chat/{chat_id}")
async def get_chat_messages(
    chat_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get conversation history for a specific chat."""
    # Get chat
    result = await db.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )
    
    # Load messages from database (persistent storage)
    from app.models.database import Message
    from sqlalchemy import asc
    
    messages_result = await db.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc())
    )
    db_messages = messages_result.scalars().all()
    
    # Convert database messages to API format
    formatted_messages = []
    for msg in db_messages:
        formatted_messages.append({
            "role": msg.role,
            "content": msg.content,
        })
    
    return {"messages": formatted_messages, "thread_id": chat.thread_id}

