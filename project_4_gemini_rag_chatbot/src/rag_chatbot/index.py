import chromadb
from dotenv import load_dotenv

from rag_chatbot.chunking import create_chunks
from rag_chatbot.config import VECTOR_STORE_DIR
from rag_chatbot.embeddings import create_embedding

load_dotenv()


def build_index():
    """Build Chroma vector index from document chunks."""
    chunks = create_chunks()

    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = chroma_client.get_or_create_collection(name="documents")

    for index, chunk in enumerate(chunks):
        embedding = create_embedding(chunk["text"])

        collection.add(
            ids=[str(index)],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[
                {
                    "source": chunk["source"],
                    "page": chunk["page"] or "",
                    "chunk_index": chunk["chunk_index"],
                }
            ],
        )

        print(f"Indexed chunk {index + 1}/{len(chunks)}")

    print(f"Index saved to {VECTOR_STORE_DIR}")


def main():
    build_index()


if __name__ == "__main__":
    main()