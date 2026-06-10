"""
System health and status API endpoint.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db

router = APIRouter(tags=["System"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_db)):
    """
    Checks if the API and database connection are fully functional.
    """
    try:
        # Executing a lightweight SELECT 1 query to confirm DB is online
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {e}"

    return {
        "status": "online" if db_status == "healthy" else "degraded",
        "database": db_status,
    }
