# app/schemas/analysis.py
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class TicketAnalysisResponse(BaseModel):
    """Schema for returning ticket analysis and predictions."""
    id: int
    ticket_id: int
    category: str
    priority: str
    sentiment: str
    confidence: float
    analysis_method: str
    
    # Heuristics for comparison/auditing
    rule_category: str | None
    rule_priority: str | None
    ml_category: str
    ml_priority: str
    rule_scores: dict | None
    
    # LLM responses (placeholder strings until Step 7)
    summary: str
    suggested_reply: str
    sources: list | None
    
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)