# app/api/tickets.py
from fastapi import APIRouter, Depends, HTTPException, Query, Security, status
from sqlalchemy.orm import Session
from app.auth import verify_api_key
from app.database import get_db
# Add new model and service imports
from app.models import Ticket, TicketAnalysis
from app.schemas.ticket import TicketCreate, TicketResponse
from app.schemas.analysis import TicketAnalysisResponse
from app.services.ticket_classifier import classifier
from app.services.ticket_rules import rule_classifier
from app.services.gemini_service import gemini_service  
from app.services.retriever import retriever_service

# Create router, protect all routes with the API Key dependency
router = APIRouter(
    prefix="/tickets",
    tags=["Tickets"],
    dependencies=[Depends(verify_api_key)]
)


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
def create_ticket(ticket_in: TicketCreate, db: Session = Depends(get_db)):
    """
    Creates a new support ticket in the database.
    """
    db_ticket = Ticket(
        subject=ticket_in.subject,
        message=ticket_in.message,
        status="open"
    )
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)
    return db_ticket


@router.get("", response_model=list[TicketResponse])
def list_tickets(
    status: str | None = Query(None, description="Filter tickets by status"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of records to return"),
    db: Session = Depends(get_db)
):
    """
    Retrieves a list of support tickets with optional pagination and status filtering.
    """
    query = db.query(Ticket)
    
    if status:
        query = query.filter(Ticket.status == status)
        
    tickets = query.offset(skip).limit(limit).all()
    return tickets


@router.get("/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a single support ticket by its database ID.
    """
    db_ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not db_ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID {ticket_id} not found"
        )
    return db_ticket


@router.post("/{ticket_id}/analyze", response_model=TicketAnalysisResponse, status_code=status.HTTP_201_CREATED)
def analyze_ticket(ticket_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a ticket, runs hybrid analysis, queries pgvector to find relevant
    FAQ policies, passes those policies as context to Gemini, and saves the results.
    """
    # 1. Fetch the ticket
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID {ticket_id} not found"
        )
    # 2. Run local ML and Rule classifiers
    ml_res = classifier.predict(ticket.subject, ticket.message)
    rule_res = rule_classifier.predict(ticket.subject, ticket.message)
    # 3. Retrieve relevant policies from the database (RAG)
    # We query using the ticket message
    matched_chunks = retriever_service.retrieve_similar_chunks(ticket.message, db, limit=2)
    
    # Format the retrieved chunks into a single text block
    context_text = ""
    sources_list = []
    for chunk in matched_chunks:
        # Retrieve source article title from the metadata JSON
        source_title = chunk.metadata_.get("source_title", "Unknown Source") if chunk.metadata_ else "Unknown Source"
        context_text += f"Document: {source_title}\nContent: {chunk.content}\n\n"
        
        # Track unique sources to save in the database
        if source_title not in sources_list:
            sources_list.append(source_title)
    # 4. Call Gemini using the retrieved policies
    llm_res = gemini_service.analyze_ticket(ticket.subject, ticket.message, context=context_text)
    # 5. Save the analysis report to the database
    db_analysis = TicketAnalysis(
        ticket_id=ticket.id,
        category=ml_res["category"],
        priority=ml_res["priority"],
        sentiment=llm_res.sentiment,
        confidence=ml_res["confidence"],
        analysis_method=ml_res["analysis_method"],
        
        rule_category=rule_res["category"],
        rule_priority=rule_res["priority"],
        ml_category=ml_res["category"],
        ml_priority=ml_res["priority"],
        rule_scores=rule_res["scores"],
        
        summary=llm_res.summary,
        suggested_reply=llm_res.suggested_reply,
        # Save the list of source article titles used to draft the reply
        sources=sources_list
    )
    
    db.add(db_analysis)
    ticket.status = "analyzed"
    db.commit()
    db.refresh(db_analysis)
    
    return db_analysis