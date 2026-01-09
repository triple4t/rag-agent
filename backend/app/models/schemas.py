"""Pydantic Schemas for API Requests/Responses."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr


# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Chat Schemas
class ChatBase(BaseModel):
    title: str


class ChatCreate(ChatBase):
    pass


class ChatResponse(ChatBase):
    id: int
    user_id: int
    thread_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatListResponse(BaseModel):
    chats: List[ChatResponse]
    total: int


# Message Schemas
class MessageBase(BaseModel):
    role: str
    content: str


class MessageCreate(MessageBase):
    chat_id: int
    sources: Optional[str] = None
    quality_score: Optional[float] = None


class MessageResponse(MessageBase):
    id: int
    chat_id: int
    sources: Optional[str] = None
    quality_score: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True

