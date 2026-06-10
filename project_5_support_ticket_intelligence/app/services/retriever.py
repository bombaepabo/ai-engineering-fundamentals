# app/services/retriever.py
from google import genai
# --- 1. Make sure to import types here ---
from google.genai import types
from sqlalchemy.orm import Session

from app.config import settings
from app.models import KnowledgeChunk


class RetrieverService:
    """
    Handles creating text embeddings via Gemini and querying
    the pgvector database for similar chunks.
    """
    def __init__(self):
        self.client = None

    def _get_client(self):
        if self.client is None:
            self.client = genai.Client(api_key=settings.gemini_api_key)
        return self.client

    def get_embedding(self, text: str) -> list[float]:
        """
        Calls Gemini Embedding API to turn a text string into a 768-dimension vector.
        """
        client = self._get_client()
        
        # --- 2. Add config here to force 768 dimensions ---
        response = client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )
        # Extract the list of floats
        return response.embeddings[0].values

    def retrieve_similar_chunks(self, query: str, db: Session, limit: int = 3) -> list[KnowledgeChunk]:
        """
        Embeds the query and runs a Cosine Similarity search on the pgvector index.
        """
        # 1. Get embedding for the query
        query_vector = self.get_embedding(query)

        # 2. Query DB and sort by Cosine Distance (lowest distance = highest similarity)
        chunks = db.query(KnowledgeChunk).order_by(
            KnowledgeChunk.embedding.cosine_distance(query_vector)
        ).limit(limit).all()

        return chunks


retriever_service = RetrieverService()