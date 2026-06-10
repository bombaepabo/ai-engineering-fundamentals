"""
Database connection and session management.

Architecture:
    Engine (connection pool)
        └── SessionLocal (session factory)
              └── get_db() (FastAPI dependency — one session per request)

The key idea: each API request gets its own database session.
When the request finishes (success or error), the session is closed.
This is handled automatically by the `get_db` dependency.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import settings

# --- Engine ---
# The engine manages a POOL of database connections.
# We don't create a new connection for every query — that would be slow.
# Instead, the engine reuses connections from the pool.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # test connections before using them (handles DB restarts)
)

# --- Session Factory ---
# sessionmaker creates a "factory" — calling SessionLocal() gives us a new Session.
# autocommit=False: we control when to commit (explicit > implicit)
# autoflush=False: we control when to flush (prevents unexpected queries)
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# --- Declarative Base ---
# All our ORM models inherit from this. It's what connects our Python classes to DB tables.
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.

    Usage in an endpoint:
        @router.get("/things")
        def list_things(db: Session = Depends(get_db)):
            return db.query(Thing).all()

    The `yield` makes this a generator — FastAPI runs the code after yield
    when the request finishes, guaranteeing the session is always closed.
    This is called the "dependency injection with cleanup" pattern.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
