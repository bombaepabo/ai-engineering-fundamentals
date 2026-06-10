"""
API Key authentication dependency.

Secures endpoints by checking the incoming `X-API-Key` header against
the application configuration `settings.api_key`.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

# This tells FastAPI to extract the header "X-API-Key"
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    FastAPI dependency to verify the API key header.

    If valid, returns the API key string.
    If missing or invalid, raises an HTTP 401 Unauthorized exception.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing from header (X-API-Key)",
        )
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key provided",
        )
    return api_key
