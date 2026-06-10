# Gemini RAG Chatbot 💬🔍

A local Retrieval-Augmented Generation (RAG) chatbot application. It reads local PDF and text documents, splits them into semantic chunks, indexes them in a local vector database, and uses Google Gemini to answer questions grounded strictly on the uploaded text.

## Tech Stack
* **LLM & Embeddings**: Google Gemini API via `google-genai` SDK (`gemini-2.5-flash-lite` & `text-embedding-004`).
* **Vector Store**: `ChromaDB` (local client).
* **Document Parsing**: `pypdf`.

## How to Run

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure API Key**:
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY="AIzaSy..."
   ```

3. **Ingest and Index Documents**:
   Place PDF/text files in `data/documents/`, then run the indexer to populate ChromaDB:
   ```bash
   uv run python -m rag_chatbot.index
   ```

4. **Run Conversational Chat**:
   Ask questions about your documents interactively:
   ```bash
   uv run python -m rag_chatbot.chat
   ```
