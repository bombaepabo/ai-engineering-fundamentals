import chromadb
from dotenv import load_dotenv

from rag_chatbot.config import TOP_K, VECTOR_STORE_DIR
from rag_chatbot.embeddings import create_embedding

load_dotenv()


def retrieve_chunks(question, top_k=TOP_K):
    """Retrieve relevant chunks for a question."""
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = chroma_client.get_or_create_collection(name="documents")

    question_embedding = create_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    chunks = []

    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(
            {
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_index": metadata["chunk_index"],
            }
        )

    return chunks


def main():
    question = "What does the document say about camera settings?"
    chunks = retrieve_chunks(question)

    for chunk in chunks:
        print("\nSource:", chunk["source"])
        print("Page:", chunk["page"])
        print("Chunk:", chunk["chunk_index"])
        print(chunk["text"][:1500])

if __name__ == "__main__":
    main()