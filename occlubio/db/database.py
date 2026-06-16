"""SQLite for the MVP (zero-config). Switch to Postgres for production by setting
OCCLUBIO_DB=postgresql+psycopg://user:pass@host/db — no other code changes needed.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from occlubio.db.models import Base

DB_URL = os.environ.get("OCCLUBIO_DB", "sqlite:///occlubio.db")
_connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}

engine = create_engine(DB_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
