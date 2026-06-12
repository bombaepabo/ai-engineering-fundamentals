# app/api/chat.py
from fastapi import APIRouter, Depends, status
from google.genai import types
from sqlalchemy.orm import Session

# --- Add this line here ---
from app.config import settings

from app.auth import verify_api_key
from app.database import get_db
from app.models import ChatLog, Ticket
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse
from app.services.retriever import retriever_service
from app.services.gemini_service import gemini_service

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)]
)


@router.post("", response_model=ChatMessageResponse, status_code=status.HTTP_201_CREATED)
def chat_interaction(request: ChatMessageRequest, db: Session = Depends(get_db)):
    """
    Stateful conversational chat endpoint.
    Retrieves history, runs RAG search, queries Gemini, and saves dialog logs.
    """
    # 1. Fetch conversation history from database
    history_logs = db.query(ChatLog).filter(
        ChatLog.session_id == request.session_id
    ).order_by(ChatLog.created_at.asc()).all()

    # 2. Retrieve relevant context for the current question (RAG)
    matched_chunks = retriever_service.retrieve_similar_chunks(request.message, db, limit=2)
    
    context_text = ""
    sources_list = []
    for chunk in matched_chunks:
        source_title = chunk.metadata_.get("source_title", "Unknown Source") if chunk.metadata_ else "Unknown Source"
        context_text += f"- {chunk.content} (Source: {source_title})\n"
        if source_title not in sources_list:
            sources_list.append(source_title)

    def get_ticket_status(ticket_id: int) -> str:
        """
        Retrieves the current status, category, priority, and AI summary 
        for a customer's support ticket from the database.
        
        Args:
            ticket_id: The integer ID of the ticket to look up.
        """
        try:
            ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
            if not ticket:
                return f"Ticket #{ticket_id} was not found in our database."
                
            status_info = f"Ticket #{ticket_id} status is '{ticket.status}'."
            if ticket.analyses:
                analysis = ticket.analyses[0]
                status_info += (
                    f" It is classified as a '{analysis.category}' "
                    f"with '{analysis.priority}' priority. "
                    f"AI Summary: {analysis.summary}"
                )
            return status_info
        except Exception as e:
            return f"Error querying ticket details: {str(e)}"

    # 3. Format the chat prompt including history and RAG context
    system_instruction = (
        "You are an expert customer support agent assistant. "
        "Answer the user's questions politely and professionally. "
        "Base your responses on the following official policy context if applicable:\n"
        f"{context_text}\n"
        "You also have access to the `get_ticket_status` tool to look up database details "
        "for a specific ticket ID. If a user asks about a specific ticket number, "
        "always use the tool to retrieve the information. "
        "If you do not know the answer, politely ask them to wait for a human agent."
    )

    # Format history messages for Gemini SDK
    messages = []
    for log in history_logs:
        messages.append(
            types.Content(
                role="user" if log.role == "user" else "model",
                parts=[types.Part.from_text(text=log.content)]
            )
        )
    # Add current message
    messages.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=request.message)]
        )
    )

    # 4. Generate response using Gemini client
    client = gemini_service._get_client()
    response = client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.5,
            tools=[get_ticket_status]
        )
    )
    response_text = response.text

    # 5. Save the user's message and the AI's reply to the database
    db_user_message = ChatLog(
        session_id=request.session_id,
        role="user",
        content=request.message,
        sources=[]
    )
    db_assistant_message = ChatLog(
        session_id=request.session_id,
        role="assistant",
        content=response_text,
        sources=sources_list
    )
    
    db.add(db_user_message)
    db.add(db_assistant_message)
    db.commit()
    db.refresh(db_assistant_message)

    return db_assistant_message