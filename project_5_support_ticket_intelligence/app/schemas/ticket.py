# app/schemas/ticket.py
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


from app.schemas.analysis import TicketAnalysisResponse


class TicketBase(BaseModel):
    """Base schema sharing common fields for Ticket."""
    subject: str = Field(..., min_length=3, max_length=200, description="Subject of the support ticket")
    message: str = Field(..., min_length=10, description="Detailed description of the issue")


class TicketCreate(TicketBase):
    """Schema for creating a new ticket (Request validation)."""
    pass


class TicketResponse(TicketBase):
    """Schema for returning a ticket's database representation (Response serialization)."""
    id: int
    status: str
    created_at: datetime
    updated_at: datetime
    analyses: list[TicketAnalysisResponse] = []

    # Tell Pydantic to read data even if it is an ORM (SQLAlchemy model) instead of a dictionary
    model_config = ConfigDict(from_attributes=True)