# app/schemas/chat.py
from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    session_id: str = Field(..., description="Unique conversation session identifier")
    message: str = Field(..., min_length=1, description="Message content sent by user")


class ChatMessageResponse(BaseModel):
    session_id: str
    role: str  # user | assistant
    content: str
    sources: list[str] | None
    created_at: datetime