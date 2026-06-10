from dotenv import load_dotenv
from google import genai

from rag_chatbot.config import CHAT_MODEL
from rag_chatbot.retrieve import retrieve_chunks

load_dotenv()

client = genai.Client()


def build_context(chunks):
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"Source {index}: {chunk['source']} page {chunk['page']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def answer_question(question):
    """Answer a question using retrieved document context."""
    chunks = retrieve_chunks(question)
    context = build_context(chunks)

    prompt = f"""
Use the context below to answer the question.

Rules:
- Answer only using the provided context.
- If the answer is not in the context, say you do not know.
- Include source names in the answer.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )

    return response.text, chunks


def main():
    question = input("Ask a question: ")

    answer, chunks = answer_question(question)

    print("\nAnswer:")
    print(answer)

    print("\nRetrieved sources:")
    for chunk in chunks:
        print(f"- {chunk['source']} page {chunk['page']} chunk {chunk['chunk_index']}")


if __name__ == "__main__":
    main()