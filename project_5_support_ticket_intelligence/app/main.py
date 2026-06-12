# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import health, tickets, knowledge, chat
from app.config import settings
# --- 1. Import Logging config and Middleware ---
from app.logging_config import setup_logging
from app.middleware import LoggingAndRequestIDMiddleware
from app.services.ticket_classifier import classifier

# --- 2. Initialize JSON Logger ---
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    # Download models from GCS if configured and missing
    try:
        from app.services.model_downloader import download_models_from_gcs
        download_models_from_gcs()
    except Exception as e:
        setup_logging()
        import logging
        logging.getLogger("app.api").warning(f"Model downloader failed to initialize: {e}")

    classifier.load_models()
    yield


app = FastAPI(
    title=settings.app_name,
    description="AI-Powered Support Ticket Intelligence Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

# --- 3. Mount the Middleware (must be added before registering routers) ---
app.add_middleware(LoggingAndRequestIDMiddleware)

# Register routers
app.include_router(health.router)
app.include_router(tickets.router)
app.include_router(knowledge.router)
app.include_router(chat.router)

# Mount static UI files (must be registered after routers to avoid overtaking api paths)
app.mount("/ui", StaticFiles(directory="app/ui", html=True), name="ui")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred."},
    )


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Support AI Ticket Intelligence Platform",
        "docs_url": "/docs",
        "environment": settings.environment,
    }