"""
SQLAlchemy ORM models — one class per database table.

Table overview:
    tickets            ← support tickets submitted by users
    ticket_analyses    ← ML/LLM analysis results for each ticket
    knowledge_articles ← knowledge base articles (help docs)
    knowledge_chunks   ← embedded chunks of articles (for RAG search)
    index_jobs         ← tracks the status of article indexing jobs
    chat_logs          ← conversation history for the chat endpoint

Relationships:
    ticket  1 ──► N  ticket_analyses
    article 1 ──► N  knowledge_chunks
    article 1 ──► N  index_jobs
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


# =============================================================================
# TICKETS
# =============================================================================

class Ticket(Base):
    """A support ticket submitted by a user."""

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="open",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationship: one ticket can have many analyses
    analyses: Mapped[list["TicketAnalysis"]] = relationship(
        back_populates="ticket",
        cascade="all, delete-orphan",
    )


class TicketAnalysis(Base):
    """
    The result of analyzing a ticket — contains ML predictions,
    LLM-generated summary, and suggested reply.

    Why a separate table? A ticket can be re-analyzed (e.g. after improving
    the model), so we store each analysis as its own record.
    """

    __tablename__ = "ticket_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=False,
    )

    # --- ML predictions ---
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    priority: Mapped[str] = mapped_column(String(20), nullable=False)
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    analysis_method: Mapped[str] = mapped_column(String(20), nullable=False)

    # --- Rule vs ML breakdown (for debugging/comparison) ---
    rule_category: Mapped[str | None] = mapped_column(String(50))
    rule_priority: Mapped[str | None] = mapped_column(String(20))
    ml_category: Mapped[str] = mapped_column(String(50), nullable=False)
    ml_priority: Mapped[str] = mapped_column(String(20), nullable=False)
    rule_scores: Mapped[dict | None] = mapped_column(JSON)

    # --- LLM outputs ---
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_reply: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationship back to ticket
    ticket: Mapped["Ticket"] = relationship(back_populates="analyses")


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

class KnowledgeArticle(Base):
    """
    A knowledge base article — could be an FAQ, troubleshooting guide,
    or policy document. The raw content is stored here; chunks are in
    a separate table for vector search.
    """

    __tablename__ = "knowledge_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="text",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",  # pending → indexed | failed
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    chunks: Mapped[list["KnowledgeChunk"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
    )
    index_jobs: Mapped[list["IndexJob"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
    )


class KnowledgeChunk(Base):
    """
    A chunk of a knowledge article with its vector embedding.

    This is the core of RAG: we split articles into small pieces,
    embed each piece as a vector, and store them here. When a user
    asks a question, we embed their question and find the most similar
    chunks using pgvector's cosine similarity search.

    The embedding column uses pgvector's Vector type — it's stored
    as a native PostgreSQL array with special indexing support.
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 768 dimensions — this is the output size of gemini-embedding-001
    embedding = mapped_column(Vector(768))

    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationship back to article
    article: Mapped["KnowledgeArticle"] = relationship(back_populates="chunks")

    # Cosine similarity index — makes vector searches fast
    __table_args__ = (
        Index(
            "ix_knowledge_chunks_embedding",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# =============================================================================
# INDEX JOBS
# =============================================================================

class IndexJob(Base):
    """
    Tracks the status of a knowledge article indexing job.

    Lifecycle: pending → processing → completed | failed

    Why track this? Indexing involves calling an external API (Gemini embeddings),
    which can fail. We need to know what succeeded and what needs retrying.
    """

    __tablename__ = "index_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",  # pending → processing → completed | failed
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationship back to article
    article: Mapped["KnowledgeArticle"] = relationship(back_populates="index_jobs")


# =============================================================================
# CHAT
# =============================================================================

class ChatLog(Base):
    """
    A single message in a chat conversation.

    We store both user messages and assistant responses.
    `session_id` groups messages into conversations — the client
    generates this ID and sends it with each request.
    """

    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
