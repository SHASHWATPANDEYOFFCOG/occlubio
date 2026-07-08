from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from occlubio.db.models import Base

DB_URL = os.environ.get("OCCLUBIO_DB", "sqlite:///occlubio.db")
_connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}

engine = create_engine(DB_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def _add_column(table: str, name: str, ddl: str) -> None:
    insp = inspect(engine)
    if table not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns(table)}
    if name not in cols:
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))


def _migrate() -> None:
    _add_column("users", "role", "role VARCHAR(16) DEFAULT 'user'")
    _add_column("users", "full_name", "full_name VARCHAR(128) DEFAULT ''")
    _add_column("users", "roll_number", "roll_number VARCHAR(64)")
    _add_column("jobs", "window_end", "window_end DATETIME")
    _add_column("sightings", "appearances", "appearances INTEGER DEFAULT 1")
    _add_column("alerts", "video_start_s", "video_start_s FLOAT DEFAULT 0.0")
    _add_column("alerts", "video_end_s", "video_end_s FLOAT DEFAULT 0.0")


def init_db() -> None:
    Base.metadata.create_all(engine)
    _migrate()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
