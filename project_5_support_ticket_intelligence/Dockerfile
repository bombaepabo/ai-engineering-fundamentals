# Dockerfile
# Use a lightweight python image
FROM python:3.13-slim

WORKDIR /app

# Install build dependencies for PostgreSQL drivers (psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv directly from the official image (extremely fast installation)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy configuration files first to cache dependencies layer
COPY pyproject.toml uv.lock ./

# Install python dependencies globally inside the container
RUN uv pip install --system -r pyproject.toml

# Copy codebase, models, and migrations into the container
COPY app/ app/
COPY models/ models/
COPY alembic/ alembic/
COPY alembic.ini ./

# Expose port 8000
EXPOSE 8000

# Start the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]