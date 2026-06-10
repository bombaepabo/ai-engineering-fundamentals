# AI Support Ticket Intelligence Platform 🚀🤖

A production-style FastAPI backend platform that automates customer support ticket management. It implements a hybrid classification pipeline (rules + local ML models), searches vector knowledge bases with pgvector, and drafts grounded replies using the Gemini API.

## Core Features
* **Hybrid Classification**: Local TF-IDF + Logistic Regression models classify Ticket Category, Priority, and Sentiment in under 2ms.
* **Semantic RAG Search**: Queries PostgreSQL using the `pgvector` extension to find relevant FAQ articles matching customer issues.
* **Grounded Replies**: Feeds the ticket and retrieved policies to Gemini to draft professional response emails and concise summaries.
* **Conversational Memory**: A stateful `/chat` endpoint with database-backed conversation tracking.
* **Observability**: Custom middleware injecting Request IDs and outputting structured JSON logs for cloud aggregation.

## How to Run

1. **Run Database Container**:
   ```bash
   docker compose up -d
   ```

2. **Run Migrations**:
   ```bash
   uv run alembic upgrade head
   ```

3. **Train Local ML Models**:
   ```bash
   uv run train_models.py
   ```

4. **Start Web Server**:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

5. **Run Tests**:
   ```bash
   uv run pytest
   ```
