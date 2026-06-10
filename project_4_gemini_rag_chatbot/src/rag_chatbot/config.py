from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash-lite"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 6