"""SQLite for the MVP (zero-config). Switch to Postgres for production by setting
OCCLUBIO_DB=postgresql+psycopg://user:pass@host/db — no other code changes needed.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from occlubio.db.models import Base

DB_URL = os.environ.get("OCCLUBIO_DB", "sqlite:///occlubio.db")
_connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}

engine = create_engine(DB_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def _migrate() -> None:
    """Tiny additive migration: add columns introduced after the first release.
    (A real deployment would use Alembic — see PLATFORM.md.)"""
    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("users")}
    if "role" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(16) DEFAULT 'user'"))


def init_db() -> None:
    Base.metadata.create_all(engine)
    _migrate()


def get_db():
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
