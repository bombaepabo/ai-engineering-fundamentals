import time
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.auth import verify_api_key
from app.database import get_db
# Import models, chunking, and embedding services
from app.models import KnowledgeArticle, KnowledgeChunk, IndexJob
from app.schemas.knowledge import KnowledgeArticleCreate, KnowledgeArticleResponse
from app.services.chunking import split_text_into_chunks
from app.services.retriever import retriever_service
router = APIRouter(
    prefix="/knowledge",
    tags=["Knowledge Base"],
    dependencies=[Depends(verify_api_key)]
)


def index_article_task(article_id: int, db_session_factory):
    """
    Background task that splits an article into chunks,
    generates vector embeddings for each chunk, and saves them to pgvector.
    """
    db: Session = db_session_factory()
    try:
        # 1. Fetch Job and Article records
        job = db.query(IndexJob).filter(IndexJob.article_id == article_id).first()
        article = db.query(KnowledgeArticle).filter(KnowledgeArticle.id == article_id).first()
        
        if not article or not job:
            return
            
        job.status = "processing"
        db.commit()
        # 2. Chunk the raw article content
        chunks = split_text_into_chunks(article.content, chunk_size=150, overlap=20)
        
        # 3. Generate embeddings and save each chunk
        for idx, chunk_text in enumerate(chunks):
            embedding_vector = retriever_service.get_embedding(chunk_text)
            
            db_chunk = KnowledgeChunk(
                article_id=article.id,
                chunk_index=idx,
                content=chunk_text,
                embedding=embedding_vector,
                metadata_={"source_title": article.title}
            )
            db.add(db_chunk)
            
        # 4. Update status tracking
        job.status = "completed"
        article.status = "indexed"
        db.commit()
        print(f"Index Job Complete: Article #{article_id} parsed into {len(chunks)} vector chunks.")
        
    except Exception as e:
        db.rollback()
        # Track failures
        if job:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
        if article:
            article.status = "failed"
            db.commit()
        print(f"Index Job Failed for Article #{article_id}: {e}")
    finally:
        db.close()


@router.post("", response_model=KnowledgeArticleResponse, status_code=status.HTTP_201_CREATED)
def create_article(
    article_in: KnowledgeArticleCreate, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    """
    Saves an article and registers a background job to index it.
    """
    # 1. Save Article Metadata
    db_article = KnowledgeArticle(
        title=article_in.title,
        content=article_in.content,
        source_url=str(article_in.source_url) if article_in.source_url else None,
        content_type=article_in.content_type,
        status="pending"
    )
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    # 2. Create the IndexJob tracker
    db_job = IndexJob(
        article_id=db_article.id,
        status="pending"
    )
    db.add(db_job)
    db.commit()
    # 3. Queue the background task using the session factory
    from app.database import SessionLocal
    background_tasks.add_task(index_article_task, db_article.id, SessionLocal)
    return db_article


@router.get("", response_model=list[KnowledgeArticleResponse])
def list_articles(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Retrieves a list of help articles with pagination."""
    return db.query(KnowledgeArticle).offset(skip).limit(limit).all()


@router.get("/{article_id}", response_model=KnowledgeArticleResponse)
def get_article(article_id: int, db: Session = Depends(get_db)):
    """Retrieves a single article by ID."""
    article = db.query(KnowledgeArticle).filter(KnowledgeArticle.id == article_id).first()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article #{article_id} not found"
        )
    return article