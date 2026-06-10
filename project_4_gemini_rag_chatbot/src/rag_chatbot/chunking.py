from rag_chatbot.config import CHUNK_OVERLAP, CHUNK_SIZE
from rag_chatbot.documents import load_documents

def split_text(text,chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def create_chunks():
    """Create chunks with source metadata."""
    documents = load_documents()
    chunks = []

    for document in documents:
        text_chunks = split_text(document["text"])

        for index, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < 100:
                continue
            chunks.append(
                {
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"],
                    "chunk_index": index,
                }
            )

    return chunks

def main():
    chunks = create_chunks()

    print(f"Created {len(chunks)} chunks")

    for chunk in chunks[:3]:
        print("\nSource:", chunk["source"])
        print("Page:", chunk["page"])
        print("Chunk:", chunk["chunk_index"])
        print(chunk["text"][:300])


if __name__ == "__main__":
    main()