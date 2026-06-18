"""Relational schema. The DB is the source of truth; the FAISS index is a derived,
rebuildable search structure (swap FAISS -> Milvus/Qdrant without touching this schema).

Embeddings are stored as raw float32 bytes here for the MVP. For production, encrypt this
column at rest (or store a *protected* template — architecture §1.12) since biometrics
cannot be reissued if leaked.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="user", index=True)  # user | authority
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    enrollment: Mapped["Enrollment"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class Session(Base):
    """Opaque bearer-token sessions (MVP stand-in for JWT; see PLATFORM.md)."""
    __tablename__ = "sessions"
    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Message(Base):
    """A notice passed from an authority to a participant (or broadcast to all)."""
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    sender_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    # recipient_id NULL == broadcast to every participant
    recipient_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Enrollment(Base):
    __tablename__ = "enrollments"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)  # 512 float32 template (encrypt in prod)
    dim: Mapped[int] = mapped_column(Integer, default=512)
    n_images: Mapped[int] = mapped_column(Integer, default=0)
    quality: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user: Mapped["User"] = relationship(back_populates="enrollment")


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), default="identify")
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|running|done|error
    input_path: Mapped[str] = mapped_column(String(512), default="")
    output_video: Mapped[str] = mapped_column(String(512), default="")
    report_path: Mapped[str] = mapped_column(String(512), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
