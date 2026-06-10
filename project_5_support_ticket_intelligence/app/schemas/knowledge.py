# app/schemas/knowledge.py
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class KnowledgeArticleBase(BaseModel):
    """Base schema sharing common fields for Knowledge Base Articles."""
    title: str = Field(
        ..., 
        min_length=3, 
        max_length=300, 
        description="Title of the FAQ/documentation article"
    )
    content: str = Field(
        ..., 
        min_length=10, 
        description="Full raw text content of the article"
    )
    source_url: str | None = Field(
        None, 
        description="Optional URL link to original online source document"
    )
    content_type: str = Field(
        "text", 
        description="Type of document (e.g. text, markdown, pdf)"
    )


class KnowledgeArticleCreate(KnowledgeArticleBase):
    """Schema for validating article creation requests."""
    pass


class KnowledgeArticleResponse(KnowledgeArticleBase):
    """Schema for serializing database response."""
    id: int
    status: str  # pending | indexed | failed
    created_at: datetime
    updated_at: datetime

    # Configures Pydantic to map SQLAlchemy attributes to JSON fields
    model_config = ConfigDict(from_attributes=True)