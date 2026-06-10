from dotenv import load_dotenv
from google import genai

from rag_chatbot.config import EMBEDDING_MODEL

load_dotenv()

client = genai.Client()


def create_embedding(text):
    """Create one embedding using Gemini."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


def main():
    embedding = create_embedding("This is a test sentence.")

    print("Embedding length:", len(embedding))
    print("First 5 values:", embedding[:5])


if __name__ == "__main__":
    main()