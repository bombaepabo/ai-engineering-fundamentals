# app/services/chunking.py

def split_text_into_chunks(text: str, chunk_size: int = 150, overlap: int = 20) -> list[str]:
    """
    Splits raw text into smaller overlapping chunks of words.
    
    This ensures each chunk fits well within the embedding model's limits
    and preserves semantic context across chunk boundaries.
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
        
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Step forward by chunk_size minus overlap to maintain shared context
        start += (chunk_size - overlap)
        
    return chunks